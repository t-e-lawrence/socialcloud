FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir -e .
ENV GENDER_DB_PATH=/data/gender_detection.db
EXPOSE 8000
CMD ["uvicorn", "socialcloud.gender_detection.main:app", "--host", "0.0.0.0", "--port", "8000"]
