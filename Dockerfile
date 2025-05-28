FROM python:3.12.7-slim-bookworm

WORKDIR /app/

# Move required resources
ADD requirements.txt .

RUN --mount=type=cache,target=/root/.cache \
    pip install -r requirements.txt

ADD model/ model
ADD server.py .

# Run flask api
# CMD ["sh", "-c", "uvicorn server:app --reload --host=0.0.0.0 --port=$PORT"]
CMD ["sh", "-c", "uvicorn server:app --reload --host=0.0.0.0"]
