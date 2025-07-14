import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import pickle
import os
import boto3

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException
from prefect import task, flow
from prefect import get_run_logger
from dotenv import load_dotenv


@task(retries=2, retry_delay_seconds=5)
def load_data(path: str) -> pd.DataFrame:
    """Load csv file from data folder"""
    return pd.read_csv(path)


target_columns = ["fuel_consumption", "CO2_emissions"]


@task
def apply_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log transformation to target columns"""

    for target in target_columns:
        df[target] = np.log1p(df[target])

    return df


@task
def split_data(df: pd.DataFrame) -> tuple:
    """Split data into train and test sets"""

    X = df.drop(columns=["ship_id", "fuel_consumption", "CO2_emissions"])
    y = df[["fuel_consumption", "CO2_emissions"]]

    df_train, df_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return df_train, df_test, y_train, y_test


@task
def preprocess_data(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple:
    """Convert training and test DataFrames into vectorized arrays using DictVectorizer."""

    dv = DictVectorizer(sparse=False)
    train_dict = df_train.to_dict(orient="records")
    test_dict = df_test.to_dict(orient="records")

    # fit and transform
    X_train = dv.fit_transform(train_dict)
    X_test = dv.transform(test_dict)
    return X_train, X_test, dv


@task
def setup_localstack_s3(bucket_name: str, logger=None):
    s3 = boto3.client(
        service_name="s3",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
        endpoint_url="http://localhost:4566",
    )

    try:
        response = s3.list_buckets()
        bucket_names = [b["Name"] for b in response.get("Buckets", [])]

        if bucket_name not in bucket_names:
            s3.create_bucket(Bucket=bucket_name)
            if logger:
                logger.info(f"S3 bucket created: {bucket_name}")
        else:
            if logger:
                logger.info(f"S3 bucket already exists: {bucket_name}")
        return s3
    except Exception as e:
        if logger:
            logger.error(f"Error setting up LocalStack S3: {e}")
        raise


@task
def train_best_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    dv: DictVectorizer,
) -> None:
    """Train a model with best hyperparams"""

    logger = get_run_logger()
    try:
        logger.info("Training model started ....")
        with mlflow.start_run() as run:
            best_params = {"n_estimators": 50, "max_depth": 5, "learning_rate": 0.1}

            model = xgb.XGBRegressor(**best_params)
            mo_model = MultiOutputRegressor(model)
            mo_model.fit(X_train, y_train)

            y_pred = mo_model.predict(X_test)
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

            # log DictVectorizer
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            models_dir = os.path.join(base_dir, "models")
            os.makedirs(models_dir, exist_ok=True)

            model_path = os.path.join(models_dir, "model.pkl")
            dv_path = os.path.join(models_dir, "dv.pkl")

            # save the model locally for using docker later
            with open(model_path, "wb") as f:
                pickle.dump(mo_model, f)
            with open(dv_path, "wb") as f:
                pickle.dump(dv, f)

            mlflow.log_artifact(dv_path, artifact_path="preprocessor")
            mlflow.sklearn.log_model(
                mo_model,
                artifact_path="model",
                registered_model_name="ship-model-artifacts",
            )

            logger.info(f"Saved DictVectorizer at: {dv_path}")
            logger.info(f"Saved model at: {model_path}")

            # upload to localstack S3

            s3 = setup_localstack_s3("ship-model-artifacts", logger)
            s3.upload_file("models/model.pkl", "ship-model-artifacts", "model.pkl")
            logger.info("Model uploaded to S3 (LocalStack).")
            # log parameters and metrics
            mlflow.set_tag("model", "XGBRegressor")
            mlflow.set_tag("model_params", str(best_params))
            mlflow.set_tag("type", "multioutput_regression")
            mlflow.log_params(best_params)
            mlflow.log_metric("rmse", rmse)
            mlflow.sklearn.log_model(
                mo_model,
                artifact_path="model",
                registered_model_name="XGBRegressor_model",
            )
            logger.info("Training completed successfully.")
            return mo_model, run.info.run_id, rmse
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


@task(retries=2, retry_delay_seconds=5)
def register_best_model(run_id: str, model_name: str) -> str:
    """Always register a new version of the model for the given Mlflow run."""

    client = MlflowClient()
    logger = get_run_logger()

    # Try to get the registered model (if it exists)
    try:
        client.get_registered_model(model_name)
        logger.info(f'Model "{model_name}" already exists. Registering new version...')
    except RestException:
        # If not found, create it
        client.create_registered_model(model_name)
        logger.info(f'Model "{model_name}" created.')

    # Register new model version
    model_uri = f"runs:/{run_id}/model"
    model_version = client.create_model_version(
        name=model_name, source=model_uri, run_id=run_id
    )

    logger.info(f"Model registered: {model_name} (version {model_version.version})")
    return f"models:/{model_name}/{model_version.version}"


@flow
def main_flow(path: str = "data/ship_fuel_efficiency.csv") -> None:
    """The main training pipeline"""

    # MLflow settings
    # load from .env
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME"))

    # load
    df = load_data(path)

    # preprocess target columns
    df = apply_log_transform(df)

    # split
    df_train, df_test, y_train, y_test = split_data(df)

    # preproces before training
    X_train, X_test, dv = preprocess_data(df_train, df_test)

    # train
    model, run_id, rmse = train_best_model(X_train, X_test, y_train, y_test, dv)

    # register model
    model_name = "ship-model-artifacts"
    register_best_model(run_id, model_name)


if __name__ == "__main__":
    main_flow()
