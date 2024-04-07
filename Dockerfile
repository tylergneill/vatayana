FROM python:3.11-slim-bookworm
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN groupadd appgroup && \
    useradd -r -M -G appgroup sanskrit
COPY --chown=sanskrit:appgroup assets /app/assets
COPY --chown=sanskrit:appgroup templates /app/templates
COPY --chown=sanskrit:appgroup ./*.py /app/
USER sanskrit
ENV PORT=5020
CMD gunicorn --bind 0.0.0.0:$PORT flask_app:app
EXPOSE 5020