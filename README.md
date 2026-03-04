# tourPlanner

python scripts\prepare_splits_binary.py --manifest_csv data/meta/crops_manifest_binary_merged.csv --output_split_json data/splits/split_binary_merged_v1.json --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --seed 42

python -c "import pandas as pd, os; df=pd.read_csv('data/meta/crops_manifest_binary_merged.csv'); missing=[p for p in df.crop_path.head(200).tolist() if not os.path.exists(p)]; print('missing in first 200:', len(missing)); print(missing[:5])"