LLAMA3_PATH = '/data/llama3/Meta-Llama-3-70B-Instruct-hf'
CHEXBERT_PATH = '/data/chexbert/chexbert.pth'
NORMALIZER_PATH = 'third-party/CXR-Report-Metric/CXRMetric/normalizer.pkl'
COMPOSITE_METRIC_V0_PATH = 'third-party/CXR-Report-Metric/CXRMetric/composite_metric_model.pkl'
COMPOSITE_METRIC_V1_PATH = 'third-party/CXR-Report-Metric/CXRMetric/radcliq-v1.pkl'

LLAMA_SYSTEM_PROMPT = '''
You are an AI assistant with expertise in radiology.
'''

LLAMA_ZEROSHOT_USER_PROMPT = '''
You are given the question, ground truth and prediction of a medical visual question answering in a clinical diagnosis scenario. Your task is to evaluate the correctness of the prediction based on the question and ground truth in terms of medical knowledge.
You should take both precision (i.e. the fraction of correct contents among the predicted contents) and recall (i.e. the fraction of correct content that were predicted) into account.
You should only focus on the contents directly answering the question. Other contents, such as further interpretation and derivation and acknowledgment of the uncertainty and need for further analysis, should be ignored and must not affect your judgement.
You should be strict and conservative. If you are not sure about the correctness of the prediction, you should give a low score.
You should provide a concise analysis and a score from 0 to 10 to summarize your evaluation. The output format is 'Analysis: ... Score: ...'. Do not output anything else.
Question: "{question}"
Ground truth: "{answer}"
Prediction: "{prediction}"
'''

LLAMA_FINETUNED_USER_PROMPT = '''
Your task is to evaluate the correctness of the prediction based on the question and ground truth in a clinical diagnosis scenario.
Question: "{question}"
Ground truth: "{answer}"
Prediction: "{prediction}"
Is the prediction correct? Provide a concise analysis and give an integer score of 0 or 1. Answer in the format "Analysis: ... Score: ...".
'''

CHEXPERT_CONDITIONS = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
              'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
              'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
              'Support Devices', 'No Finding']

RADBERT_CONDITIONS = [
    'Medical material','Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion',
    'Coronary artery wall calcification', 'Hiatal hernia','Lymphadenopathy', 'Emphysema', 'Atelectasis',
    'Lung nodule','Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion',
    'Mosaic attenuation pattern','Peribronchial thickening', 'Consolidation', 'Bronchiectasis',
    'Interlobular septal thickening'
]

CHEXPERT_5 = [1, 4, 5, 7, 9]