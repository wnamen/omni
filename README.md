# OmniParser API Service

This project provides a containerized OmniParser service with a test script for parsing UI elements from images.

## Quick Start

### 1. Start the Service

```bash
# Build and start the service
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs omniparser
```

### 2. Test the API

```bash
# Run the test script
python test_parse_api.py
```

The test script will:

- Wait for the service to be ready
- Test the health endpoint
- Parse the `mobile.png` image
- Save results to the root directory

## API Endpoints

### Health Check

```
GET /api/health
```

Returns service status and model loading information.

### Parse Image

```
POST /api/parse
Content-Type: application/json

{
  "image": "base64_encoded_image_data"
}
```

Returns parsed elements and a labeled image.

## Docker Compose Commands

```bash
# Start service in background
docker-compose up -d

# Stop service
docker-compose down

# Rebuild and restart
docker-compose up --build -d

# View logs
docker-compose logs -f omniparser

# Check resource usage
docker stats omniparser-container
```

## Requirements

- Docker and Docker Compose
- Python 3.7+ (for test script)
- `requests` library: `pip install requests`

## Troubleshooting

1. **Service not responding**: Check logs with `docker-compose logs omniparser`
2. **Models not loading**: The service may need several minutes to download and load models on first run
3. **Memory issues**: The service requires significant memory (4-8GB recommended)
4. **Connection refused**: Ensure the service is running with `docker-compose ps`

## Configuration

The service is configured to:

- Run on port 2171
- Auto-restart unless stopped
- Include health checks
- Mount a results directory for persistent output
