import orjson
import pandas as pd
from tqdm import tqdm

from mmmm.data import get_target_tax
from mmmm.data.defs import ORIGIN_VL_DATA_ROOT, PROCESSED_VL_DATA_ROOT
from mmmm.data.target_tax import TargetCategory

dataset_dir = ORIGIN_VL_DATA_ROOT / 'CT-RATE/dataset'

labels = [
    ('Arterial wall calcification', 'calcification'),
    ('Cardiomegaly', 'cardiomegaly'),
    ('Pericardial effusion', 'pericardial effusion'),
    ('Coronary artery wall calcification', 'calcification'),  # too hard to accurately taxonomize
    ('Hiatal hernia', 'hiatal hernia'),
    ('Lymphadenopathy', 'lymphadenopathy'),
    ('Emphysema', 'pulmonary emphysema'),
    ('Atelectasis', 'atelectasis'),
    ('Lung nodule', 'lung nodule'),
    ('Lung opacity', 'pulmonary opacification'),
    # 'Pulmonary fibrotic sequela',
    ('Pleural effusion', 'pleural effusion'),
    # 'Mosaic attenuation pattern',
    ('Peribronchial thickening', 'peribronchial cuffing'),
    ('Consolidation', 'pulmonary consolidation'),
    ('Bronchiectasis', 'bronchiectasis'),
    ('Interlobular septal thickening', 'interlobular septal thickening'),
]

tax = get_target_tax()
for _, name in labels:
    target = tax.get(name)
    assert target is not None, name
    assert target.category == TargetCategory.ANOMALY, name

def cmp_series(a: pd.Series, b: pd.Series) -> pd.Series:
    return ((a == b) | (a.isna() & b.isna())).all()

def main():
    output_dir = PROCESSED_VL_DATA_ROOT / 'CT-RATE'
    output_dir.mkdir(exist_ok=True, parents=True)
    for split in ['train', 'validate']:
        report_path = dataset_dir / 'radiology_text_reports' / (('train' if split == 'train' else 'validation') + '_reports.csv')
        report_df = pd.read_csv(report_path, index_col='VolumeName')
        label_path = dataset_dir / 'multi_abnormality_labels' / (('train' if split == 'train' else 'valid') + '_predicted_labels.csv')
        label_df = pd.read_csv(label_path, index_col='VolumeName')
        image_dir = dataset_dir / ('train' if split == 'train' else 'valid')
        data = []
        for patient_dir in tqdm(list(image_dir.iterdir())):
            for study_dir in patient_dir.iterdir():
                study = {
                    'key': study_dir.name,
                    'image': [],
                    'modality': [],
                    'anomaly_pos': [],
                    'anomaly_neg': [],
                }
                ref_report = None
                ref_label = None
                for image_path in study_dir.glob('*.nii.gz'):
                    study['image'].append(str(image_path))
                    study['modality'].append('CT')
                    if ref_report is None:
                        ref_report = report_df.loc[image_path.name]
                        ref_label = label_df.loc[image_path.name]
                    else:
                        assert cmp_series(ref_report, report_df.loc[image_path.name]), image_path.name
                        assert cmp_series(ref_label, label_df.loc[image_path.name]), image_path.name
                if pd.isna(findings := ref_report['Findings_EN']):
                    continue
                study['findings'] = findings.strip()
                if not pd.isna(impression := ref_report['Impressions_EN']):
                    study['impression'] = impression.strip()
                for anomaly_key, anomaly_name in labels:
                    if ref_label[anomaly_key]:
                        study['anomaly_pos'].append(anomaly_name)
                    else:
                        study['anomaly_neg'].append(anomaly_name)
                data.append(study)
        (output_dir / f'{split}-raw.json').write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))

if __name__ == '__main__':
    main()
