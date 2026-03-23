FROM onerahmet/ffmpeg:n7.1 AS ffmpeg

FROM swaggerapi/swagger-ui:v5.9.1 AS swagger-ui

FROM python:3.10-bookworm

LABEL org.opencontainers.image.source="https://github.com/ahmetoner/whisper-asr-webservice"

ENV POETRY_VENV=/app/.venv

# Copy FFmpeg binary, libraries, headers, and pkgconfig from the ffmpeg stage
# Required for building PyAV (av) which needs FFmpeg 7 dev files
COPY --from=ffmpeg /usr/local/bin/ffmpeg /usr/local/bin/ffmpeg
COPY --from=ffmpeg /usr/local/lib/ /usr/local/lib/
COPY --from=ffmpeg /usr/local/include/ /usr/local/include/
COPY --from=ffmpeg /usr/local/lib/pkgconfig/ /usr/local/lib/pkgconfig/

RUN ldconfig

RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==2.1.3

ENV PATH="${PATH}:${POETRY_VENV}/bin"
ENV PKG_CONFIG_PATH="/usr/local/lib/pkgconfig"

WORKDIR /app

COPY . .
COPY --from=swagger-ui /usr/share/nginx/html/swagger-ui.css swagger-ui-assets/swagger-ui.css
COPY --from=swagger-ui /usr/share/nginx/html/swagger-ui-bundle.js swagger-ui-assets/swagger-ui-bundle.js

RUN poetry config virtualenvs.in-project true
RUN poetry install --extras cpu

EXPOSE 9000

ENTRYPOINT ["whisper-asr-webservice"]
