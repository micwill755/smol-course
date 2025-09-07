# Argilla Setup Guide

## What is Argilla?
Argilla is an open-source data annotation platform for AI/ML projects. It provides a web interface for human-in-the-loop data curation and validation.

## Quick Start with Docker (Recommended)

### 1. Start Argilla Container
```bash
docker run -d --name argilla -p 6900:6900 argilla/argilla-quickstart:latest
```

### 2. Verify It's Running
```bash
docker ps | grep argilla
```

### 3. Access Web Interface
- **URL**: http://localhost:6900
- **Username**: `admin`
- **Password**: `12345678`

### 4. API Credentials
- **API Key**: `admin.apikey`
- **API URL**: `http://localhost:6900`

## Alternative Setup Methods

### Option 1: Docker Compose
```bash
curl -O https://raw.githubusercontent.com/argilla-io/argilla/main/docker/docker-compose.yaml
docker-compose up -d
```

### Option 2: Python Package
```bash
pip install argilla[server]
python -m argilla server start
```

### Option 3: Hugging Face Spaces
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Create new Space with Argilla template
3. Deploy and use the provided URL

## Using Argilla in Scripts

```python
import argilla as rg

# Connect to Argilla
client = rg.Argilla(
    api_key="admin.apikey", 
    api_url="http://localhost:6900"
)

# Create dataset
dataset = rg.Dataset(name="my_dataset", settings=settings)
dataset.create()
```

## Stopping Argilla

```bash
# Stop container
docker stop argilla

# Remove container
docker rm argilla
```

## Troubleshooting

### Connection Issues
- Check if Docker is running: `docker ps`
- Verify port 6900 is available: `lsof -i :6900`
- Check container logs: `docker logs argilla`

### Authentication Issues
- Use API key: `admin.apikey` (not `argilla.apikey`)
- Ensure correct URL: `http://localhost:6900`

### Dataset Name Issues
- Use simple names (letters, numbers, spaces, hyphens, underscores only)
- Avoid special characters and file paths
