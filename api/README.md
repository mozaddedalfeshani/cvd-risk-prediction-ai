# 🚀 CVD Risk Prediction API

## 🎯 Overview

High-performance FastAPI backend serving the CVD risk prediction model with production-grade features including authentication, monitoring, and comprehensive documentation.

## 📁 Structure

```
api/
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── app_simple.py        # Lightweight development server
│   ├── models.py            # Pydantic data models
│   ├── routes/
│   │   ├── prediction.py    # Prediction endpoints
│   │   └── health.py        # Health check endpoints
│   └── utils/
│       ├── model_loader.py  # ML model management
│       └── validators.py    # Input validation
├── requirements.txt         # API dependencies
├── Dockerfile              # Container configuration
└── README.md               # This file
```

## 🚀 Quick Start

### Local Development

```bash
cd api
pip install -r requirements.txt
python app/app_simple.py
```

### Production Server

```bash
cd api
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 5001
```

### Docker Deployment

```bash
cd api
docker build -t cvd-api .
docker run -p 5001:5001 cvd-api
```

## 📡 API Endpoints

### Health & Status

- `GET /api/health` - Health check and system status
- `GET /api/model/info` - Model information and metadata
- `GET /api/features` - Required features for prediction

### Prediction

- `POST /api/predict` - CVD risk prediction
- `GET /api/example` - Get example patient data

### Documentation

- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation

## 🎯 Usage Examples

### Health Check

```bash
curl http://54.160.253.120:5001/api/health
```

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_info": {
    "model_type": "XGBoost Ensemble",
    "accuracy": 0.9315,
    "version": "1.0"
  }
}
```

### Risk Prediction

```bash
curl -X POST "http://54.160.253.120:5001/api/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "Age": 45,
       "Sex": 1,
       "BMI": 28.5,
       "Systolic BP": 140,
       "Total Cholesterol (mg/dL)": 220,
       "Smoking Status": 0,
       "Diabetes Status": 0
     }'
```

```json
{
  "success": true,
  "prediction": {
    "risk_level": "INTERMEDIATE",
    "risk_code": 1,
    "confidence": 0.85,
    "probabilities": {
      "LOW": 0.15,
      "INTERMEDIATE": 0.75,
      "HIGH": 0.1
    },
    "model_accuracy": 0.9315
  },
  "timestamp": "2025-01-08T23:00:00.000Z"
}
```

### Example Data

```bash
curl "http://54.160.253.120:5001/api/example?type=low_risk"
```

```json
{
  "name": "Low Risk Patient",
  "data": {
    "Age": 30,
    "Sex": 0,
    "BMI": 23.0,
    "Systolic BP": 110,
    "Total Cholesterol (mg/dL)": 180,
    "Smoking Status": 0,
    "Diabetes Status": 0
  }
}
```

## 📋 Data Models

### Patient Input

```python
class PatientData(BaseModel):
    Age: int = Field(..., ge=18, le=100)
    Sex: int = Field(..., ge=0, le=1)  # 0=Female, 1=Male
    BMI: float = Field(..., ge=15.0, le=50.0)
    systolic_bp: int = Field(..., ge=80, le=250)
    total_cholesterol: float = Field(..., ge=100.0, le=400.0)
    smoking_status: int = Field(..., ge=0, le=1)
    diabetes_status: int = Field(..., ge=0, le=1)
    # ... additional fields
```

### Prediction Response

```python
class PredictionResponse(BaseModel):
    success: bool
    prediction: PredictionResult
    timestamp: str

class PredictionResult(BaseModel):
    risk_level: str  # "LOW", "INTERMEDIATE", "HIGH"
    risk_code: int   # 0, 1, 2
    confidence: float
    probabilities: Dict[str, float]
    model_accuracy: float
```

## 🔧 Configuration

### Environment Variables

```bash
# Model Configuration
MODEL_PATH=/path/to/model.pkl
MODEL_VERSION=1.0

# API Configuration
API_HOST=0.0.0.0
API_PORT=5001
DEBUG=false

# Security
SECRET_KEY=your-secret-key
ALLOWED_ORIGINS=http://localhost:3000

# Monitoring
LOG_LEVEL=INFO
METRICS_ENABLED=true
```

### Model Loading

```python
from app.utils.model_loader import CVDModelManager

# Initialize model manager
model_manager = CVDModelManager()

# Load production model
model = model_manager.load_model("production")

# Health check
status = model_manager.health_check()
```

## 🛡️ Security Features

### Input Validation

- Pydantic models for request validation
- Range checking for medical values
- Type safety and sanitization
- SQL injection prevention

### Authentication (Production)

- JWT token authentication
- Role-based access control
- API rate limiting
- Request logging and monitoring

### CORS Configuration

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## 📊 Performance Monitoring

### Metrics Tracked

- Request/response times
- Model prediction latency
- Error rates and types
- Memory and CPU usage
- Model accuracy drift

### Health Monitoring

```python
# Health check endpoint
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "model_loaded": model_manager.is_loaded(),
        "memory_usage": get_memory_usage(),
        "uptime": get_uptime()
    }
```

## 🧪 Testing

### Unit Tests

```bash
python -m pytest tests/api/ -v
```

### Integration Tests

```bash
python -m pytest tests/integration/ -v
```

### Load Testing

```bash
# Install locust
pip install locust

# Run load tests
locust -f tests/load/test_api.py --host=http://54.160.253.120:5001
```

## 🐳 Docker Configuration

### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY models/ ./models/

EXPOSE 5001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5001"]
```

### Docker Compose

```yaml
version: "3.8"
services:
  api:
    build: .
    ports:
      - "5001:5001"
    environment:
      - MODEL_PATH=/app/models/production.pkl
    volumes:
      - ./models:/app/models
    restart: unless-stopped
```

## 📈 Production Deployment

### Scalability

- Horizontal scaling with load balancer
- Model caching for performance
- Database connection pooling
- Async request handling

### Monitoring & Logging

- Structured logging with JSON format
- Application Performance Monitoring (APM)
- Error tracking and alerting
- Model performance monitoring

### CI/CD Pipeline

```yaml
# .github/workflows/api-deploy.yml
name: API Deployment
on:
  push:
    branches: [main]
    paths: ["api/**"]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: pytest api/tests/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: ./scripts/deploy-api.sh
```

## 📋 Requirements

### Core Dependencies

```txt
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
python-multipart>=0.0.6
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
joblib>=1.3.0
```

### Optional Dependencies

```txt
redis>=5.0.0          # Caching
prometheus-client>=0.19.0  # Metrics
python-jose[cryptography]  # JWT auth
locust>=2.17.0        # Load testing
pytest>=7.4.0         # Testing
```

---

**🎯 Production-ready API serving clinical-grade CVD predictions!**
