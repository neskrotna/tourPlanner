# tourPlanne

..Get a SAS token

$ACCOUNT = "<STORAGE_ACCOUNT_NAME_FROM_ABOVE>"
$CONTAINER = "<CONTAINER_NAME_FROM_ABOVE>"

az storage container generate-sas `
  --account-name $ACCOUNT `
  --name $CONTAINER `
  --permissions acdlrw `
  --expiry (Get-Date).AddHours(6).ToString("yyyy-MM-ddTHH:mmZ") `
  --auth-mode login `
  -o tsv

..Copy the SAS output and store it:

$SAS = "<PASTE_SAS_HERE>"

..upload only the needed parts:

az storage blob upload-batch --account-name $ACCOUNT --destination $CONTAINER --destination-path "azureml-datasets/orientation_poc_v1/crops"  --source "data/crops"  --sas-token $SAS
az storage blob upload-batch --account-name $ACCOUNT --destination $CONTAINER --destination-path "azureml-datasets/orientation_poc_v1/meta"   --source "data/meta"   --sas-token $SAS
az storage blob upload-batch --account-name $ACCOUNT --destination $CONTAINER --destination-path "azureml-datasets/orientation_poc_v1/splits" --source "data/splits" --sas-token $SAS

..Register the uploaded folder as an AzureML Data Asset 

$schema: https://azuremlschemas.azureedge.net/latest/data.schema.json
name: orientation-poc-v1
version: 1
type: uri_folder
path: azureml://datastores/workspaceblobstore/paths/azureml-datasets/orientation_poc_v1/
description: Crops + manifests + splits for binary orientation PoC

..Then register

az ml data create -f azureml/data.yml

..Check:

az ml data show -n orientation-poc-v1 -v 1

..Create an AzureML Environment from requirements.txt

$schema: https://azuremlschemas.azureedge.net/latest/environment.schema.json
name: orientation-poc-env
version: 1
image: mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04
conda_file: azureml/conda.yml
description: Environment for orientation estimation training

..Create azureml/conda.yml

name: orientation-poc
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - pip:
      - torch>=2.2.0
      - torchvision>=0.17.0
      - timm>=1.0.7
      - numpy>=1.26.0
      - pandas>=2.2.0
      - pillow>=10.2.0
      - opencv-python>=4.9.0.80
      - scikit-learn>=1.4.0
      - matplotlib>=3.8.0
      - tqdm>=4.66.0
      - pyyaml>=6.0.1
      - python-dotenv>=1.0.1

..Create the env

az ml environment create -f azureml/env.yml

..Create the Job YAML

$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
type: command

name: orientation-train-resnet18
display_name: orientation_train_resnet18

code: .

environment: azureml:orientation-poc-env:1

compute: azureml:gpu-compute   # your existing compute name

inputs:
  data_dir:
    type: uri_folder
    path: azureml:orientation-poc-v1:1
  config_path:
    type: uri_file
    path: configs/binary/resnet18.yaml

command: >-
  python -m src.cli.train
  --config ${{inputs.config_path}}
  --data-root ${{inputs.data_dir}}

outputs:
  outputs_dir:
    type: uri_folder
    mode: rw_mount

..Submit the job

az ml job create -f azureml/job_train.yml --stream

..To see jobs later

az ml job list --max-results 20

..To open in Studio:

az ml job show --name <JOB_NAME>

..Download outputs (model checkpoints, metrics, etc.)

az ml job download --name <JOB_NAME> --output-name outputs_dir --download-path ./azureml_outputs

xxx