# tourPlanner

import argparse
from pathlib import Path

from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential


def get_ml_client() -> MLClient:
    """
    Uses your existing 'az login' session (DefaultAzureCredential).
    AzureML context comes from CLI defaults set via:
      az account set --subscription ...
      az configure --defaults group=... workspace=... location=...
    """
    cred = DefaultAzureCredential(exclude_interactive_browser_credential=False)
    return MLClient.from_config(credential=cred)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Submit AzureML training job")

    p.add_argument(
        "--code-dir",
        type=str,
        default=".",
        help="Repo root that AzureML uploads as job code.",
    )

    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model config yaml inside repo, e.g. configs/binary/resnet18.yaml",
    )

    p.add_argument(
        "--data-asset",
        type=str,
        required=True,
        help="AzureML data asset reference, e.g. azureml:poc_orientation_estimation:1",
    )

    p.add_argument(
        "--compute",
        type=str,
        required=True,
        help="Compute target name, e.g. gpu-compute",
    )

    p.add_argument(
        "--environment",
        type=str,
        required=True,
        help="AzureML environment name:version, e.g. orientation-env:1",
    )

    p.add_argument(
        "--experiment-name",
        type=str,
        default="poc_orientation_estimation",
        help="AzureML experiment name.",
    )

    p.add_argument(
        "--display-name",
        type=str,
        default=None,
        help="Optional display name in AzureML UI.",
    )

    p.add_argument(
        "--task",
        type=str,
        default="train-binary",
        help="Optional tag for grouping runs.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    ml_client = get_ml_client()

    code_dir = Path(args.code_dir).resolve()
    config_path = args.config.replace("\\", "/")

    # Data asset mounted as a folder input called "data"
    data_input = Input(type="uri_folder", path=args.data_asset)

    # Your training entrypoint. Uses --data-root so your code rewrites cfg dataset paths.
    # Keep it simple and consistent with your local calls.
    job = command(
        code=str(code_dir),
        command=(
            "python -m src.cli.train "
            f"--config {config_path} "
            "--data-root ${{inputs.data}}"
        ),
        inputs={
            "data": data_input,
        },
        environment=args.environment,
        compute=args.compute,
        experiment_name=args.experiment_name,
        display_name=args.display_name or f"{args.task}-{Path(config_path).stem}",
        tags={
            "task": args.task,
            "config": config_path,
            "dataset": args.data_asset,
            "compute": args.compute,
            "env": args.environment,
        },
    )

    submitted = ml_client.jobs.create_or_update(job)

    print("Submitted job:", submitted.name)
    print("Studio URL:", submitted.studio_url)


if __name__ == "__main__":
    main()

..
pip install azure-ai-ml azure-identity

xxx