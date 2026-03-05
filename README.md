# tourPlanner

from azure.ai.ml import MLClient, command, Input
from azure.identity import DefaultAzureCredential
from pathlib import Path

# ===== SETTINGS (change only if needed) =====

DATASET = "azureml:poc_orientation_estimation:1"
ENVIRONMENT = "orientation-env:1"
COMPUTE = "gpu-compute"
EXPERIMENT = "poc_orientation_estimation"

CONFIG_FILE = "configs/resnet18.yaml"

# ===========================================


def main():
    ml_client = MLClient.from_config(
        credential=DefaultAzureCredential()
    )

    job = command(
        code=".",
        command=f"python -m src.cli.train --config {CONFIG_FILE} --data-root ${{inputs.data}}",
        inputs={
            "data": Input(type="uri_folder", path=DATASET)
        },
        environment=ENVIRONMENT,
        compute=COMPUTE,
        experiment_name=EXPERIMENT,
        display_name=Path(CONFIG_FILE).stem,
    )

    submitted_job = ml_client.jobs.create_or_update(job)

    print("Job submitted:", submitted_job.name)
    print("Studio URL:", submitted_job.studio_url)


if __name__ == "__main__":
    main()


xxx