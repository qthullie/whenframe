# The demo site's text server (app.py), for any machine that runs containers or a Hugging Face Space (Docker).
# The text towers are downloaded when the image is built, so that the first visitor does not wait for them.
FROM python:3.12-slim

ARG WHENFRAME_REF=main
ENV WHENFRAME_CACHE=/opt/whenframe \
    WHENFRAME_ORIGINS=https://qthullie.github.io \
    PYTHONUNBUFFERED=1

COPY app.py /app/app.py
RUN pip install --no-cache-dir "https://github.com/qthullie/whenframe/archive/${WHENFRAME_REF}.tar.gz" \
 && cd /app && python -c "import app; app.text_encoders()" \
 && chmod -R a+rX /opt/whenframe
EXPOSE 7860
CMD ["python", "/app/app.py", "--host", "0.0.0.0", "--port", "7860"]
