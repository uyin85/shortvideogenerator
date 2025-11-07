FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        imagemagick \
        libmagickwand-dev && \
    rm -rf /var/lib/apt/lists/*

# Safely fix ImageMagick policy (works for v6 and v7)
RUN if [ -f /etc/ImageMagick-6/policy.xml ]; then \
        sed -i 's/<policy domain="path" rights="none" pattern="@*"/<policy domain="path" rights="read|write" pattern="@*"/g' /etc/ImageMagick-6/policy.xml; \
    elif [ -f /etc/ImageMagick-7/policy.xml ]; then \
        sed -i 's/<policy domain="path" rights="none" pattern="@*"/<policy domain="path" rights="read|write" pattern="@*"/g' /etc/ImageMagick-7/policy.xml; \
    else \
        echo "ImageMagick policy.xml not found — skipping policy fix."; \
    fi

# Set up app
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
