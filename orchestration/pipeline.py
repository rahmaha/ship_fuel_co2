
import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import pickle
import os

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException
from prefect import task, flow
from prefect import get_run_logger


@task(retries=2, retry_delay_seconds=5)
def load_data(path: str) -> pd.DataFrame:
    """Load csv file from data folder"""
    return pd.read_csv(path)

target_columns = ['fuel_consumption', 'CO2_emissions']

@task
def apply_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log transformation to target columns"""

    for target in target_columns:
        df[target] = np.log1p(df[target])

    return df

@task 
def split_data(df: pd.DataFrame) -> tuple:
    """Split data into train and test sets """

    X = df.drop(columns=['ship_id', 'fuel_consumption', 'CO2_emissions'])
    y = df[['fuel_consumption', 'CO2_emissions']]

    df_train, df_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return df_train, df_test, y_train, y_test

@task
def preprocess_data(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple:
    """Convert training and test DataFrames into vectorized arrays using DictVectorizer."""

    dv = DictVectorizer(sparse=False)
    train_dict = df_train.to_dict(orient='records')
    test_dict = df_test.to_dict(orient='records')

    # fit and transform
    X_train = dv.fit_transform(train_dict)
    X_test = dv.transform(test_dict)
    return X_train, X_test, dv

@task
def train_best_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    dv: DictVectorizer
) -> None:
    """Train a model with best hyperparams """
    
    logger = get_run_logger()
    try: 
        logger.info("Training model started ....")
        with mlflow.start_run() as run:
            
            best_params = {
                'n_estimators': 50, 
                'max_depth': 5, 
                'learning_rate': 0.1
                }
            
            model = xgb.XGBRegressor(**best_params)
            mo_model = MultiOutputRegressor(model)
            mo_model.fit(X_train, y_train)

            y_pred = mo_model.predict(X_test)
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

            #log DictVectorizer
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            models_dir = os.path.join(base_dir, "models")
            os.makedirs(models_dir, exist_ok=True)

            dv_path = os.path.join(models_dir, "dv.pkl")
            with open(dv_path, "wb") as f:
                pickle.dump(dv, f)
            mlflow.log_artifact(dv_path, artifact_path='preprocessor')
            logger.info(f"Saved DictVectorizer at: {dv_path}")

            # log parameters and metrics
            mlflow.set_tag("model", "XGBRegressor")
            mlflow.set_tag("model_params", str(best_params))
            mlflow.set_tag("type", "multioutput_regression")
            mlflow.log_params(best_params)
            mlflow.log_metric(f'rmse', rmse)
            mlflow.sklearn.log_model(
                mo_model,
                artifact_path="model",
                registered_model_name="XGBRegressor_model"
            )
            logger.info('Training completed successfully.')
            return mo_model, run.info.run_id, rmse
    except Exception as e:
        logger.error(f'Training failed: {e}')
        raise

@task(retries=2, retry_delay_seconds=5)
def register_best_model(run_id: str, model_name: str) -> str:
    """Register the model from a specific Mlflow run"""

    client = MlflowClient()
    try:
        client.create_registered_model(model_name)
    except RestException:
        print(f"Model '{model_name}' already exists. Skipping creation.")

    model_uri = f"runs:/{run_id}/model"
    model_version = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id
    )

    print(f'Model registered: {model_name} (version {model_version.version})')
    return f'models:/{model_name}/{model_version.version}'

@flow
def main_flow(path: str = "data/ship_fuel_efficiency.csv") -> None:
    """The main training pipeline"""

    # MLflow settings
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ship_fuel_co2_mlops")

    #load 
    df = load_data(path)

    # preprocess target columns
    df[target_columns] = apply_log_transform(df[target_columns])

    # split
    df_train, df_test, y_train, y_test = split_data(df)

    # preproces before training
    X_train, X_test, dv = preprocess_data(df_train, df_test)

    # train
    model, run_id, rmse = train_best_model(X_train, X_test, y_train, y_test, dv)

    # register model
    model_name = 'ship_fuel_co2_predictor'
    register_best_model(run_id, model_name)


if __name__ == "__main__":
    main_flow()