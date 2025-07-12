from evidently import DataDefinition
from evidently import Dataset
from evidently import Report
from evidently import Regression
from evidently.presets import DataDriftPreset
from evidently.metrics import RMSE
import pandas as pd
import sys
import os
from prefect import flow, task, get_run_logger
from datetime import datetime

# to mport main_flow
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from orchestration.pipeline import main_flow

@task
def generate_report(
    ref_path='data/reference.csv',
    cur_path='data/current_with_preds.csv',
    html_path='monitoring/dashboard/report.html',
    json_path='monitoring/dashboard/report.json'
):
    # Load datasets
    ref_data = pd.read_csv(ref_path)
    cur_data = pd.read_csv(cur_path)

    # Create "pred" columns in reference to match Evidently format
    ref_data['fuel_consumption_pred'] = ref_data['fuel_consumption']
    ref_data['CO2_emissions_pred'] = ref_data['CO2_emissions']

    # Define regression tasks
    definition = DataDefinition(
        regression=[
            Regression(
                target="fuel_consumption",
                prediction="fuel_consumption_pred",
                is_default=True
            ),
            Regression(
                target="CO2_emissions",
                prediction="CO2_emissions_pred",
                name="co2_model"
            )
        ]
    )

    # Convert to Evidently Dataset
    ref_dataset = Dataset.from_pandas(ref_data, data_definition=definition)
    cur_dataset = Dataset.from_pandas(cur_data, data_definition=definition)

    # Create report with metrics
    report = Report(metrics=[
        DataDriftPreset(),
        RMSE(target_column="fuel_consumption", prediction_column="fuel_consumption_pred"),
        RMSE(target_column="CO2_emissions", prediction_column="CO2_emissions_pred")
    ])

    # Run report
    evaluation = report.run(reference_data=ref_dataset, current_data=cur_dataset)

    # Save dashboard
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    evaluation.save_html(html_path)
    with open(json_path, "w") as f:
        f.write(evaluation.json())

    return evaluation

# ----------------------
# Step 2: Extract metrics
# ----------------------

@task
def extract_metrics(results_dict):
    rmse = None
    drift_count = 0

    for metric in results_dict["metrics"]:
        if "RMSE" in metric.get("metric_id", ""):
            rmse = metric.get("value", None)
        elif "DriftedColumnsCount" in metric.get("metric_id", ""):
            drift_count = metric.get("value", {}).get("count", 0)
    return rmse, drift_count

# ----------------------
# Step 3: Conditional retraining
# ----------------------

@task
def evaluate_and_trigger_rerun(rmse, drift_count, rmse_threshold=1000, drift_threshold=3):
    retrain = False

    if rmse is not None and rmse > rmse_threshold:
        print(f" RMSE too high! ({rmse:.2f})")
        retrain = True

    if drift_count > drift_threshold:
        print(f"Too many drifted columns! ({drift_count})")
        retrain = True

    if retrain:
        print("Triggering ML pipeline...")
        main_flow()
    else:
        print("No retraining needed.")

    return retrain

# ----------------------
# Step 4: Logging
# ----------------------

@task
def log_monitoring_event(rmse, drift_count, retrain, log_path="monitoring/logs/monitoring_log.csv"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as log_file:
        log_file.write(f"{datetime.now()},{rmse},{drift_count},{retrain}\n")

    print(f"[LOGGED] RMSE: {rmse}, Drifted: {drift_count}, Retrain Triggered: {retrain}")

@flow(name="monitoring-flow")
def monitoring_flow():
    logger = get_run_logger()
    logger.info('Running monitoring flow....')

    evaluation = generate_report()
    results_dict = evaluation.dict()
    rmse, drift_count = extract_metrics(results_dict)
    print(f"RMSE: {rmse}")
    print(f"Drifted columns: {drift_count}")

    retrain = evaluate_and_trigger_rerun(rmse, drift_count)
    log_monitoring_event(rmse, drift_count, retrain)