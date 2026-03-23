import gc
import time
from dataclasses import dataclass, field
from io import StringIO
from threading import Thread
from typing import BinaryIO, List, Optional, Union

import numpy as np
import torch
import whisper
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from transformers import AutoProcessor, pipeline

from app.asr_models.asr_model import ASRModel
from app.config import CONFIG
from app.utils import ResultWriter, WriteJSON, WriteSRT, WriteTSV, WriteTXT, WriteVTT


@dataclass
class Segment:
    """Segment dataclass compatible with the result writers (asdict support)."""

    id: int = 0
    seek: int = 0
    start: float = 0.0
    end: float = 0.0
    text: str = ""
    tokens: List[int] = field(default_factory=list)
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0
    words: Optional[list] = None
    temperature: Optional[float] = None


class OpenVINOWhisperASR(ASRModel):

    def load_model(self):
        model_id = self._get_model_id(CONFIG.MODEL_NAME)
        device = self._get_ov_device(CONFIG.DEVICE)

        ov_config = {}
        if device == "GPU":
            ov_config["PERFORMANCE_HINT"] = "LATENCY"

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=CONFIG.MODEL_PATH,
        )

        load_in_8bit = CONFIG.MODEL_QUANTIZATION == "int8"

        self.model = OVModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            export=True,
            load_in_8bit=load_in_8bit,
            device=device,
            ov_config=ov_config,
            cache_dir=CONFIG.MODEL_PATH,
        )

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=30,
        )

        Thread(target=self.monitor_idleness, daemon=True).start()

    def transcribe(
        self,
        audio,
        task: Union[str, None],
        language: Union[str, None],
        initial_prompt: Union[str, None],
        vad_filter: Union[bool, None],
        word_timestamps: Union[bool, None],
        options: Union[dict, None],
        output,
    ):
        self.last_activity_time = time.time()

        with self.model_lock:
            if self.model is None:
                self.load_model()

        generate_kwargs = {}
        if task:
            generate_kwargs["task"] = task
        if language:
            generate_kwargs["language"] = language
        if initial_prompt:
            prompt_ids = self.processor.get_prompt_ids(initial_prompt, return_tensors="np")
            generate_kwargs["prompt_ids"] = prompt_ids

        with self.model_lock:
            result = self.pipe(
                {"raw": audio, "sampling_rate": CONFIG.SAMPLE_RATE},
                return_timestamps=True,
                generate_kwargs=generate_kwargs,
            )

        # Convert pipeline output to segment format expected by writers
        segments = []
        text = result.get("text", "")
        chunks = result.get("chunks", [])

        if chunks:
            for i, chunk in enumerate(chunks):
                ts = chunk.get("timestamp", (0.0, 0.0))
                start = ts[0] if ts[0] is not None else 0.0
                end = ts[1] if ts[1] is not None else 0.0
                segments.append(Segment(id=i, start=start, end=end, text=chunk.get("text", "")))
        else:
            segments.append(Segment(id=0, start=0.0, end=0.0, text=text))

        detected_language = language
        if not detected_language and chunks:
            # Fall back to a short transcription to detect language
            detected_language = "en"

        formatted_result = {
            "language": detected_language or "en",
            "segments": segments,
            "text": text,
        }

        output_file = StringIO()
        self.write_result(formatted_result, output_file, output)
        output_file.seek(0)
        return output_file

    def language_detection(self, audio):
        self.last_activity_time = time.time()

        with self.model_lock:
            if self.model is None:
                self.load_model()

        # Pad/trim audio to 30 seconds
        audio = whisper.pad_or_trim(audio)

        input_features = self.processor(
            audio,
            sampling_rate=CONFIG.SAMPLE_RATE,
            return_tensors="pt",
        ).input_features

        with self.model_lock:
            # Generate a single token with no forced language to let the model detect it
            predicted_ids = self.model.generate(
                input_features,
                max_new_tokens=1,
                forced_decoder_ids=None,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Extract language token (last token in the generated sequence)
        lang_token_id = predicted_ids.sequences[0, -1].item()
        lang_token = self.processor.tokenizer.decode([lang_token_id])
        lang_code = lang_token.replace("<|", "").replace("|>", "").strip()

        # Compute confidence from logits
        scores = predicted_ids.scores[0][0]
        probs = torch.nn.functional.softmax(scores.float(), dim=-1)
        confidence = probs[lang_token_id].item()

        return lang_code, confidence

    def release_model(self):
        del self.model
        self.model = None
        self.processor = None
        self.pipe = None
        gc.collect()
        print("Model unloaded due to timeout")

    def write_result(self, result: dict, file: BinaryIO, output: Union[str, None]):
        if output == "srt":
            WriteSRT(ResultWriter).write_result(result, file=file)
        elif output == "vtt":
            WriteVTT(ResultWriter).write_result(result, file=file)
        elif output == "tsv":
            WriteTSV(ResultWriter).write_result(result, file=file)
        elif output == "json":
            WriteJSON(ResultWriter).write_result(result, file=file)
        else:
            WriteTXT(ResultWriter).write_result(result, file=file)

    @staticmethod
    def _get_model_id(model_name: str) -> str:
        """Map short model names to HuggingFace model IDs."""
        distil_models = {
            "distil-large-v2": "distil-whisper/distil-large-v2",
            "distil-large-v3": "distil-whisper/distil-large-v3",
            "distil-medium.en": "distil-whisper/distil-medium.en",
            "distil-small.en": "distil-whisper/distil-small.en",
        }
        if model_name in distil_models:
            return distil_models[model_name]
        if "/" in model_name:
            return model_name
        return f"openai/whisper-{model_name}"

    @staticmethod
    def _get_ov_device(device: str) -> str:
        """Map config device string to OpenVINO device name."""
        mapping = {"cpu": "CPU", "gpu": "GPU", "auto": "AUTO"}
        return mapping.get(device.lower(), "CPU")
