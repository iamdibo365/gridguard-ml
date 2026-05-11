"""
GridGuard ML — Model Card Generator
Auto-generates model card JSON and Markdown from MLflow run artifacts.
Called by GitHub Actions Job 4 after successful deployment.
"""
import mlflow
import boto3
import json
import os
import argparse
from datetime import datetime

MODEL_NAME = "gridguard.energy.anomaly_detector"
BUCKET     = os.getenv("S3_BUCKET", "gridguard-ml-data")

DATABRICKS_HOST  = os.getenv("DATABRICKS_HOST", "").rstrip("/")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")

os.environ["DATABRICKS_HOST"]  = DATABRICKS_HOST
os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")


def generate_model_card(model_version: str) -> dict:
    client = mlflow.MlflowClient()
    mv     = client.get_model_version(MODEL_NAME, model_version)
    run    = client.get_run(mv.run_id)
    m      = run.data.metrics
    p      = run.data.params

    # Download artifacts
    tmp_dir = f"/tmp/mlflow_artifacts_{model_version}"
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        client.download_artifacts(mv.run_id, "bias_evaluation.json",    tmp_dir)
        client.download_artifacts(mv.run_id, "feature_importance.json", tmp_dir)
        client.download_artifacts(mv.run_id, "confusion_matrix.json",   tmp_dir)

        with open(f"{tmp_dir}/bias_evaluation.json") as f:
            bias = json.load(f)
        with open(f"{tmp_dir}/feature_importance.json") as f:
            fi = json.load(f)
        with open(f"{tmp_dir}/confusion_matrix.json") as f:
            cm_data = json.load(f)
    except Exception as e:
        print(f"Warning: Could not download all artifacts: {e}")
        bias    = {}
        fi      = {}
        cm_data = {}

    top_features = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:5] if fi else []

    card = {
        "model_card_version": "1.1",
        "generated_at":       datetime.utcnow().isoformat(),
        "model_details": {
            "name":         MODEL_NAME,
            "version":      model_version,
            "type":         "Binary Classifier — Anomaly Detection",
            "framework":    "XGBoost",
            "intended_use": (
                "Detect anomalous energy consumption patterns in grid substations "
                "to support human review and investigation."
            ),
            "out_of_scope": [
                "Real-time safety-critical SCADA control decisions",
                "Substations outside the training distribution (industrial > 5MW)",
                "Data with fewer than 24 hours of historical readings (cold-start)"
            ],
        },
        "training_data": {
            "description":        "90 days of hourly energy sensor readings from 50 grid substations",
            "split_strategy":     p.get("split_strategy", "stratified_three_way_70_15_15"),
            "train_samples":      int(float(p.get("train_size", 0))),
            "val_samples":        int(float(p.get("val_size", 0))),
            "test_samples":       int(float(p.get("test_size", 0))),
            "anomaly_rate_train": float(p.get("anomaly_rate_train", 0)),
            "anomaly_rate_val":   float(p.get("anomaly_rate_val", 0)),
            "anomaly_rate_test":  float(p.get("anomaly_rate_test", 0)),
            "num_features":       20,
            "data_source":        f"s3://{BUCKET}/raw/energy_readings/",
            "note":               "Test set evaluated exactly once — primary reported metrics."
        },
        "performance_metrics": {
            "primary_eval_set": "test",
            "note": "Test set touched once. Validation used during training only.",
            "test_set": {
                "auc":       round(m.get("auc", 0), 4),
                "f1":        round(m.get("f1", 0), 4),
                "precision": round(m.get("precision", 0), 4),
                "recall":    round(m.get("recall", 0), 4),
            },
            "validation_set": {
                "auc":       round(m.get("val_auc", 0), 4),
                "f1":        round(m.get("val_f1", 0), 4),
                "precision": round(m.get("val_precision", 0), 4),
                "recall":    round(m.get("val_recall", 0), 4),
            },
            "val_test_auc_gap": round(m.get("val_test_auc_gap", 0), 4),
            "best_iteration":   int(float(p.get("best_iteration", 0))),
            "confusion_matrix": cm_data,
        },
        "bias_evaluation":  bias,
        "top_features":     [{"feature": k, "importance": round(v, 4)} for k, v in top_features],
        "limitations": [
            "Performance may degrade for novel anomaly types not seen during training",
            "Rolling features require 24h+ of history — cold-start stations may see elevated false positives",
            "Seasonal distribution shift not yet validated beyond Q1 2024 training window",
        ],
        "ethical_considerations": {
            "fairness":        "Model applied uniformly across all station types. No demographic data used.",
            "transparency":    "Feature importance logged per run. SHAP explanations available on request.",
            "human_oversight": "All anomaly alerts require human review before operational action."
        },
        "governance": {
            "owner":            "GridGuard ML Platform Team",
            "review_schedule":  "Quarterly model performance review",
            "nist_rmf_stage":   "Manage",
            "quality_gate":     "Test AUC >= 0.85",
            "overfitting_gate": "Val/Test AUC gap <= 0.05",
            "last_audit":       datetime.utcnow().strftime("%Y-%m-%d"),
            "model_registry":   f"databricks://models:/{MODEL_NAME}/{model_version}",
        }
    }
    return card


def save_artifacts(card: dict, version: str):
    os.makedirs("governance/output", exist_ok=True)

    json_path = f"governance/output/model_card_v{version}.json"
    with open(json_path, "w") as f:
        json.dump(card, f, indent=2)

    ts  = card["performance_metrics"]["test_set"]
    vs  = card["performance_metrics"]["validation_set"]
    gap = card["performance_metrics"]["val_test_auc_gap"]

    md = f"""# Model Card: {card['model_details']['name']} v{version}

**Generated:** {card['generated_at']}

## Model Details
- **Type:** {card['model_details']['type']}
- **Framework:** {card['model_details']['framework']}
- **Intended Use:** {card['model_details']['intended_use']}

## Data Split
| Set | Rows | Anomaly Rate |
|-----|------|-------------|
| Train (70%) | {card['training_data']['train_samples']:,} | {card['training_data']['anomaly_rate_train']:.4f} |
| Validation (15%) | {card['training_data']['val_samples']:,} | {card['training_data']['anomaly_rate_val']:.4f} |
| Test (15%) | {card['training_data']['test_samples']:,} | {card['training_data']['anomaly_rate_test']:.4f} |

## Performance — Test Set (Primary)
| Metric | Value |
|--------|-------|
| AUC | {ts['auc']} |
| F1 | {ts['f1']} |
| Precision | {ts['precision']} |
| Recall | {ts['recall']} |

## Performance — Validation Set (Reference)
| Metric | Value |
|--------|-------|
| AUC | {vs['auc']} |
| F1 | {vs['f1']} |

**Val/Test AUC Gap:** {gap}

## Limitations
""" + "\n".join(f"- {l}" for l in card['limitations']) + f"""

## Human Oversight
{card['ethical_considerations']['human_oversight']}

## Governance
- **Owner:** {card['governance']['owner']}
- **NIST RMF Stage:** {card['governance']['nist_rmf_stage']}
- **Quality Gate:** {card['governance']['quality_gate']}
- **Review Schedule:** {card['governance']['review_schedule']}
"""

    with open(f"governance/output/model_card_v{version}.md", "w") as f:
        f.write(md)

    # Upload to S3
    try:
        s3  = boto3.client("s3")
        key = f"governance/model_cards/model_card_v{version}.json"
        s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(card, indent=2))
        print(f"✓ Model card saved to S3: s3://{BUCKET}/{key}")
    except Exception as e:
        print(f"Warning: Could not upload to S3: {e}")

    print(f"✓ Model card saved locally: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", required=True)
    args = parser.parse_args()

    print(f"Generating model card for {MODEL_NAME} v{args.model_version}...")
    card = generate_model_card(args.model_version)
    save_artifacts(card, args.model_version)
    print("✓ Model card generation complete.")
