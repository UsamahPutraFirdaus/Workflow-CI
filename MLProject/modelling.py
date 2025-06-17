import argparse
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
import mlflow
import joblib
import os

# CLI Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--model_output', type=str, required=True)
parser.add_argument('--test_size', type=float, default=0.2)
parser.add_argument('--random_state', type=int, default=42)
args = parser.parse_args()

# Cek apakah dijalankan di GitHub Actions
if os.environ.get('CI') == 'true':
    print('ℹ️ CI detected. Forcing local MLflow tracking to ./mlruns')
    os.makedirs('mlruns', exist_ok=True)
    mlflow.set_tracking_uri('file:./mlruns')
    mlflow.set_experiment('diabetes-prediction-local')
    use_remote_tracking = False

# Jika tidak di CI, cek kredensial Dagshub
elif os.environ.get('DAGSHUB_TOKEN') and os.environ.get('DAGSHUB_USERNAME'):
    os.environ['MLFLOW_TRACKING_USERNAME'] = os.environ.get('DAGSHUB_USERNAME')
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.environ.get('DAGSHUB_TOKEN')
    mlflow.set_tracking_uri('https://dagshub.com/UsamahPutraFirdaus/diabetes-prediction-model.mlflow')
    mlflow.set_experiment('diabetes-prediction')
    use_remote_tracking = True
    print('✅ Using Remote MLflow tracking on Dagshub')

# Jika tidak ada kredensial Dagshub, fallback ke lokal
else:
    print('⚠️ Dagshub Credentials not found. Using local MLflow Tracking')
    os.makedirs('mlruns', exist_ok=True)
    mlflow.set_tracking_uri('file:./mlruns')
    mlflow.set_experiment('diabetes-prediction-local')
    use_remote_tracking = False

# Load dataset dari argumen CLI
data = pd.read_csv(args.data_path)
if not os.path.exists(args.data_path):
    raise FileNotFoundError(f"❌ File data tidak ditemukan: {args.data_path}")


X = data.drop('diabetes', axis=1)
y = data['diabetes']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.random_state
)

# Autologging
mlflow.sklearn.autolog(log_models=False)

with mlflow.start_run(run_name="Baseline Logistic Regression"):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Manual logging metrics tambahan
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
    mlflow.log_metric("log_loss", log_loss(y_test, model.predict_proba(X_test)))

    # Logging parameter tambahan
    mlflow.log_param("train_size", X_train.shape[0])
    mlflow.log_param("test_size", X_test.shape[0])
    mlflow.log_param("features_count", X.shape[1])
    mlflow.log_param("data_path", args.data_path)

    # Simpan dan log model
    os.makedirs(os.path.dirname(args.model_output), exist_ok=True)
    joblib.dump(model, args.model_output)
    mlflow.log_artifact(args.model_output)

    print("✅ Model disimpan:", args.model_output)

    # Register model (jika remote)
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/model"
    registered_model_name = "diabetes-prediction"

    if use_remote_tracking:
        try:
            mlflow.register_model(model_uri=model_uri, name=registered_model_name)
            print(f"✅ Model registered as '{registered_model_name}'")
        except Exception as e:
            print(f"❌ Failed to register model: {e}")

# Simpan URL experiment DagsHub
if use_remote_tracking:
    with open("DagsHub.txt", "w") as f:
        f.write("https://dagshub.com/UsamahPutraFirdaus/diabetes-prediction-model.mlflow/#/experiments/0")
    print("🔗 Hasil eksperimen dapat dilihat di: https://dagshub.com/UsamahPutraFirdaus/diabetes-prediction-model.mlflow/#/experiments/0")
    print("📁 Link juga disimpan di file: DagsHub.txt")

    print("📍 Untuk serve model via MLflow, jalankan:")
    print(f"🔗 mlflow models serve -m 'models:/{registered_model_name}/latest' --port 5000")
else:
    print("📍 Untuk serve model secara lokal, jalankan:")
    print(f"🔗 mlflow models serve -m '{model_uri}' --port 5000")

print("\n🎉 Pelatihan Selesai!")
