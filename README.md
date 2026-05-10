# GridGuard ML — Energy Anomaly Detection

**Full-stack production MLOps project**
AWS S3 · Databricks · MLflow · XGBoost · FastAPI · Docker · ECR · GitHub Actions · Evidently AI · NIST AI RMF

---

## Architecture

```
[Python Generator]  →  S3 raw CSV
         ↓
[Databricks Auto Loader]  →  Bronze Delta Table
         ↓
[Silver Notebook]  →  Validated Delta Table (DQ gates)
         ↓
[Gold Notebook]  →  20 engineered features
         ↓
[MLflow Training]  →  XGBoost (70/15/15 split)  →  Unity Catalog Registry
         ↓
[download_model.py]  →  model/ folder
         ↓
[GitHub Actions]  →  AUC gate + overfitting gate  →  Docker  →  ECR
         ↓
[FastAPI /predict]  →  Prediction log  →  Evidently drift monitoring
         ↓
[Governance]  →  Model card + NIST AI RMF checklist  →  S3
```

---

## Quick Start

### 1. Setup
```bash
git clone https://github.com/<your-handle>/gridguard-ml.git
cd gridguard-ml
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

### 2. AWS Infrastructure
```bash
aws s3 mb s3://gridguard-ml-data --region us-east-1
aws s3api put-bucket-versioning --bucket gridguard-ml-data \
    --versioning-configuration Status=Enabled
aws ecr create-repository --repository-name gridguard-inference --region us-east-1
```

### 3. Databricks Setup
Run in Databricks SQL Editor:
```sql
CREATE CATALOG IF NOT EXISTS gridguard;
CREATE SCHEMA IF NOT EXISTS gridguard.energy;
```

### 4. Generate Data
```bash
python data/generate_data.py
```

### 5. Run Databricks Notebooks (in order)
Import notebooks from `databricks/notebooks/` into your workspace:
1. `01_bronze_ingestion.py`
2. `02_silver_transform.py`
3. `03_gold_features.py`
4. `04_model_training.py`  ← fill in DATABRICKS_HOST and TOKEN first

### 6. Download Model and Build Docker Image
```bash
cd inference

# Copy template and fill in your values
cp .env.template .env

# Download model from Databricks (reads credentials from .env)
pip install python-dotenv 'mlflow-skinny[databricks]'
python download_model.py

# Build image
DOCKER_BUILDKIT=1 docker build -t gridguard-inference:latest .

# Run locally
docker run -d \
  --name gridguard \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/model:/app/model \
  -v $(pwd)/logs:/app/logs \
  gridguard-inference:latest
```

### 7. Test the API
```bash
# Health check
curl http://localhost:8000/health

# Normal prediction (should return is_anomaly: false)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"station_id":"STATION_001","energy_kwh":450.0,"voltage_v":241.5,"current_a":10.4,"power_factor":0.92,"temperature_c":22.1,"hour":14,"dow":2,"month":6,"is_weekend":0,"is_peak":0,"energy_rolling_mean_24h":450.0,"energy_rolling_std_24h":45.0,"energy_rolling_mean_7d":430.0,"energy_rolling_max_24h":520.0,"voltage_rolling_mean_24h":240.0,"energy_lag_1h":448.0,"energy_lag_24h":442.0,"energy_delta_1h":2.0,"energy_vs_24h_mean":1.0,"z_score_24h":0.04}'
```

### 8. Push to ECR
```bash
ACCOUNT_ID=<YOUR_AWS_ACCOUNT_ID>
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin \
    $ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

docker tag gridguard-inference:latest \
    $ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/gridguard-inference:latest

docker push \
    $ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/gridguard-inference:latest
```

### 9. GitHub Actions CI/CD
Add these secrets to your GitHub repo (Settings → Secrets → Actions):
- `DATABRICKS_HOST`
- `DATABRICKS_TOKEN`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

Push to main to trigger the pipeline.

---

## Key Design Decisions & Fixes Applied

| Issue | Fix |
|---|---|
| Auto Loader EventBridge error | Removed `useNotifications=true` — uses directory listing mode |
| Unity Catalog `input_file_name()` error | Replaced with `_metadata.file_path` |
| Streaming `monotonically_increasing_id()` error | Removed — not supported in streaming |
| MLflow Legacy registry disabled | Uses `databricks-uc` registry URI + Unity Catalog 3-part model name |
| S3 auth error in Docker container | Model downloaded locally via `download_model.py`, loaded from mounted volume |
| Pydantic `model_` namespace warning | Added `model_config = {"protected_namespaces": ()}` to all schemas |
| Voltage validator blocking spike detection | Raised limit from 350V to 500V |
| Three-way train/val/test split | 70/15/15 stratified, early stopping on validation, test evaluated once |

---

*Built by Dibongo Ngoh — Senior AI/ML Architect*
 
