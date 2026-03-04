# tourPlanner

python -c "import pandas as pd; df=pd.read_csv('data/meta/crops_manifest_binary_merged.csv'); df.loc[df['label_name'].str.contains('rotated'), 'label_name']='rotated'; df.loc[df['label_name']=='rotated', 'label_id']=1; df.loc[df['label_name']=='normal', 'label_id']=0; df.to_csv('data/meta/crops_manifest_binary_merged.csv', index=False); print(df['label_name'].value_counts())"

--
python -c "from pathlib import Path; from src.data.packcrop_dataset import PackCropDataset; ds=PackCropDataset(Path('data/meta/crops_manifest_binary_merged.csv'), Path('data/meta/classes_binary.json'), Path('data/splits/split_binary_v1.json'), split='train'); print(len(ds), ds.class_counts()); x,y,meta=ds[0]; print(x.shape, y.item(), meta['crop_path'])"

--
python scripts/prepare_splits_binary.py --manifest_csv data/meta/crops_manifest_binary_merged.csv --output_split_json data/splits/split_binary_merged_v1.json --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --seed 42
xxx