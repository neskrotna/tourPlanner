# tourPlanner

from azure.ai.ml import MLClient, command, Input
from azure.identity import DefaultAzureCredential
from pathlib import Path
import argparse


DATASET = "azureml:poc_orientation_estimation:1"
ENVIRONMENT = "orientation-env:1"
COMPUTE = "gpu-cluster-nc6sv3"
EXPERIMENT = "poc_orientation_estimation"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Submit training job to Azure ML")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument("--display-name", type=str, default=None, help="Optional display name for the job")
    return p.parse_args()


def main():

    print("\n==============================")
    print("Azure ML Job Submission Start")
    print("==============================\n")

    args = parse_args()

    print("Arguments received:")
    print("Config:", args.config)
    print("Display name:", args.display_name)

    cfg_path = Path(args.config).resolve()
    print("\nResolved config path:", cfg_path)

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    display_name = args.display_name or cfg_path.stem
    print("Final display name:", display_name)

    print("\nConnecting to Azure ML workspace...")
    ml_client = MLClient.from_config(
        credential=DefaultAzureCredential()
    )
    print("Connected to Azure ML workspace")

    print("\nJob configuration:")
    print("Dataset:", DATASET)
    print("Environment:", ENVIRONMENT)
    print("Compute:", COMPUTE)
    print("Experiment:", EXPERIMENT)

    print("\nCreating job definition...")

    job = command(
        code="src",
        command="python -m cli.train --config ${{inputs.config}} --data-root ${{inputs.data}}",
        inputs={
            "data": Input(type="uri_folder", path=DATASET),
            "config": Input(type="uri_file", path=str(cfg_path))
        },
        environment=ENVIRONMENT,
        compute=COMPUTE,
        experiment_name=EXPERIMENT,
        display_name=display_name,
    )

    print("Job definition created")
    print("Source code folder to upload: src/")
    print("Config file passed as input:", cfg_path)

    print("\nSubmitting job to Azure ML...")

    submitted_job = ml_client.jobs.create_or_update(job)

    print("\n==============================")
    print("Job submitted successfully")
    print("==============================")

    print("Job name:", submitted_job.name)
    print("Studio URL:", submitted_job.studio_url)


if __name__ == "__main__":
    main()

xxx