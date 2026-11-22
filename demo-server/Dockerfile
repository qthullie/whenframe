# The demo site's text server (app.py), for any machine that runs containers or a Hugging Face Space (Docker).
# The text tower is downloaded when the image is built, so that the first visitor does not wait for it.
FROM python:3.12-slim

ARG WHENFRAME_REF=main
ENV WHENFRAME_CACHE=/opt/whenframe \
    WHENFRAME_ORIGINS=https://qthullie.github.io \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir "https://github.com/qthullie/whenframe/archive/${WHENFRAME_REF}.tar.gz" \
 && python -c "from whenframe.encoder import DEFAULT_MODEL, MODELS, fetch_files; s = MODELS[DEFAULT_MODEL]; fetch_files(s.repo, (s.tokenizer_file, s.text_file))" \
 && chmod -R a+rX /opt/whenframe

COPY app.py /app/app.py
EXPOSE 7860
CMD ["python", "/app/app.py", "--host", "0.0.0.0", "--port", "7860"]
