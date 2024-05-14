LLAMA3_PATH = '/data/llama3/Meta-Llama-3-8B-Instruct-hf'
CHEXBERT_PATH = '/data/chexbert/chexbert.pth'
RADCLIQ_PATH = 'third-party/CXR-Report-Metric/CXRMetric/radcliq-v1.pkl'


LLAMA_SYSTEM_PROMPT = '''
You are an AI assistant specialized in medical topics. 
'''

LLAMA_USER_PROMPT = '''
You are given the question, ground truth and prediction of a medical visual question answering task. Your task is to evaluate the correctness of the prediction based on the question and ground truth in terms of medical knowledge, considering both precision (i.e. the fraction of relevant contents among the predicted contents) and recall (i.e. the fraction of relevant content that were predicted). You should provide a concise analysis and a score from 0 to 10 to summarize your evaluation. The output format is 'Analysis: ... Score: ...'. Do not output anything else.
question: {question}
ground truth: {answer}
prediction: {prediction}
'''


FEW_SHOT_PROMPTS = {
    'MIMIC-CXR': '''
You are a helpful medical assistant. You will be provided with a chest radiograph and you should provide a radiology report for the image.
Here are some examples of radiology reports:
1. Findings: There is no focal consolidation, pleural effusion or pneumothorax.  Bilateral nodular opacities that most likely represent nipple shadows. The cardiomediastinal silhouette is normal.  Clips project over the left lung, potentially within the breast. The imaged upper abdomen is unremarkable. Chronic deformity of the posterior left sixth and seventh ribs are noted. Impression: No acute cardiopulmonary process.
2. Findings: PA and lateral views of the chest.  Lung volumes are low.  Overlying soft tissue causes haziness throughout the lungs.  There is no focal consolidation, pleural effusion, or pneumothorax.  There is mild pulmonary vascular congestion.  There is mild cardiomegaly. Impression: Mild cardiomegaly and pulmonary vascular congestion.
3. Findings: Frontal and lateral views of the chest.  Moderate cardiomegaly and mediastinal contours are stable.  Prominence of the pulmonary vascular markings is consistent with mild congestion.  Lungs are hyperinflated, suggestive of COPD. No focal consolidation, pleural effusion, or pneumothorax. Impression: 1.  No focal consolidation.   2.  Mild pulmonary vascular congestion.    3.  Lung hyperinflation.
Now please provide a radiology report for the provided image.
'''
}