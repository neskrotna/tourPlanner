# tourPlanner

python -c "import yaml; from src.models.factory import create_model, get_num_params; cfg=yaml.safe_load(open('configs/binary/resnet18.yaml','r',encoding='utf-8')); m=create_model(cfg); print('OK', type(m).__name__, get_num_params(m))"

xxx