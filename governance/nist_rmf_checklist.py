"""
GridGuard ML — NIST AI RMF Compliance Checklist Generator
Maps all 4 RMF functions to concrete evidence artifacts in the pipeline.
Run automatically by GitHub Actions after each successful deployment.
"""
import json
import boto3
import os
from datetime import datetime

BUCKET = os.getenv("S3_BUCKET", "gridguard-ml-data")

checklist = {
    "framework":   "NIST AI Risk Management Framework 1.0",
    "system_name": "GridGuard ML — Energy Anomaly Detector",
    "generated":   datetime.utcnow().isoformat(),
    "functions": {
        "GOVERN": {
            "description": "Organizational accountability and culture of AI risk management",
            "controls": [
                {
                    "id":       "GV-1.1",
                    "control":  "AI risk policy defined",
                    "status":   "COMPLETE",
                    "evidence": "config/config.yaml — AUC gate (0.85), PSI threshold (0.2), SLO recall (0.90)"
                },
                {
                    "id":       "GV-1.2",
                    "control":  "Roles and responsibilities assigned",
                    "status":   "COMPLETE",
                    "evidence": "governance/output/model_card.json — governance.owner field"
                },
                {
                    "id":       "GV-2.1",
                    "control":  "Model risk tolerance documented",
                    "status":   "COMPLETE",
                    "evidence": "AUC threshold 0.85, PSI alert 0.2, SLO recall 0.90 in config.yaml"
                },
                {
                    "id":       "GV-4.1",
                    "control":  "Governance review cadence established",
                    "status":   "COMPLETE",
                    "evidence": "model_card.json — governance.review_schedule: Quarterly"
                },
            ]
        },
        "MAP": {
            "description": "Context, categorization and risk identification",
            "controls": [
                {
                    "id":       "MP-1.1",
                    "control":  "AI system categorized by risk level",
                    "status":   "COMPLETE",
                    "evidence": "model_card.json — human oversight required for all anomaly alerts"
                },
                {
                    "id":       "MP-2.1",
                    "control":  "Data sources and lineage documented",
                    "status":   "COMPLETE",
                    "evidence": "Databricks Unity Catalog — automatic lineage from bronze to gold"
                },
                {
                    "id":       "MP-3.1",
                    "control":  "Potential harms identified",
                    "status":   "COMPLETE",
                    "evidence": "model_card.json — limitations and ethical_considerations sections"
                },
                {
                    "id":       "MP-4.1",
                    "control":  "Intended and prohibited use cases documented",
                    "status":   "COMPLETE",
                    "evidence": "model_card.json — intended_use and out_of_scope fields"
                },
            ]
        },
        "MEASURE": {
            "description": "Analysis and assessment of AI risks",
            "controls": [
                {
                    "id":       "MS-1.1",
                    "control":  "Performance metrics established and tracked",
                    "status":   "COMPLETE",
                    "evidence": "MLflow — AUC, F1, precision, recall logged per run. Test set evaluated once."
                },
                {
                    "id":       "MS-2.1",
                    "control":  "Bias evaluation completed",
                    "status":   "COMPLETE",
                    "evidence": "bias_evaluation.json — recall/precision by anomaly type in MLflow artifacts"
                },
                {
                    "id":       "MS-2.5",
                    "control":  "Model explainability documented",
                    "status":   "PARTIAL",
                    "evidence": "feature_importance.json logged per run. SHAP analysis pending."
                },
                {
                    "id":       "MS-2.6",
                    "control":  "Overfitting prevention documented",
                    "status":   "COMPLETE",
                    "evidence": "Three-way 70/15/15 stratified split. Val/Test AUC gap gate <= 0.05 in CI/CD."
                },
                {
                    "id":       "MS-3.1",
                    "control":  "Monitoring plan in place",
                    "status":   "COMPLETE",
                    "evidence": "monitoring/drift_detector.py — PSI, KS tests weekly. SLO recall >= 0.90."
                },
            ]
        },
        "MANAGE": {
            "description": "Prioritize and address identified AI risks",
            "controls": [
                {
                    "id":       "MG-1.1",
                    "control":  "Incident response plan documented",
                    "status":   "COMPLETE",
                    "evidence": "governance/generate_system_card.py — failure_modes with detection and response"
                },
                {
                    "id":       "MG-2.2",
                    "control":  "Model rollback procedure tested",
                    "status":   "COMPLETE",
                    "evidence": "MLflow Unity Catalog Registry — champion alias reassignment enables instant rollback"
                },
                {
                    "id":       "MG-3.1",
                    "control":  "Retraining trigger defined",
                    "status":   "COMPLETE",
                    "evidence": "PSI > 0.2 or SLO breach documented as retraining trigger in drift_detector.py"
                },
                {
                    "id":       "MG-4.1",
                    "control":  "Deployment gating enforced",
                    "status":   "COMPLETE",
                    "evidence": "GitHub Actions — AUC gate + overfitting gate block deployment automatically"
                },
                {
                    "id":       "MG-4.2",
                    "control":  "Human oversight enforced at decision point",
                    "status":   "COMPLETE",
                    "evidence": "model_card.json — all anomaly alerts require human review before action"
                },
            ]
        }
    }
}

if __name__ == "__main__":
    os.makedirs("governance/output", exist_ok=True)

    with open("governance/output/nist_rmf_checklist.json", "w") as f:
        json.dump(checklist, f, indent=2)

    total    = sum(len(v["controls"]) for v in checklist["functions"].values())
    complete = sum(
        1 for v in checklist["functions"].values()
        for c in v["controls"] if c["status"] == "COMPLETE"
    )
    partial  = sum(
        1 for v in checklist["functions"].values()
        for c in v["controls"] if c["status"] == "PARTIAL"
    )
    print(f"✓ NIST AI RMF: {complete}/{total} COMPLETE | {partial} PARTIAL")

    # Upload to S3
    try:
        s3  = boto3.client("s3")
        key = "governance/nist_rmf_checklist.json"
        s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(checklist, indent=2))
        print(f"✓ Checklist saved to S3: s3://{BUCKET}/{key}")
    except Exception as e:
        print(f"Warning: Could not upload to S3: {e}")

    print("✓ NIST RMF checklist generation complete.")
