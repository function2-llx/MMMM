LLAMA3_PATH = '/data/llama3/Meta-Llama-3-70B-Instruct-hf'
CHEXBERT_PATH = '/data/chexbert/chexbert.pth'
NORMALIZER_PATH = 'third-party/CXR-Report-Metric/CXRMetric/normalizer.pkl'
COMPOSITE_METRIC_V0_PATH = 'third-party/CXR-Report-Metric/CXRMetric/composite_metric_model.pkl'
COMPOSITE_METRIC_V1_PATH = 'third-party/CXR-Report-Metric/CXRMetric/radcliq-v1.pkl'


LLAMA_SYSTEM_PROMPT = '''
You are an AI assistant specialized in medical topics. 
'''

LLAMA_USER_PROMPT = '''
You are given the question, ground truth and prediction of a medical visual question answering in a clinical diagnosis scenario. Your task is to evaluate the correctness of the prediction based on the question and ground truth in terms of medical knowledge.
You should take both precision (i.e. the fraction of correct contents among the predicted contents) and recall (i.e. the fraction of correct content that were predicted) into account.
You should only focus on the contents directly answering the question. Other contents, such as further interpretation and derivation and acknowledgment of the uncertainty and need for further analysis, should be ignored and must not affect your judgement.
You should be strict and conservative. If you are not sure about the correctness of the prediction, you should give a low rating.
You should provide a concise analysis and a rating from 0 to 10 to summarize your evaluation. The output format is 'Analysis: ... Rating: ...'. Do not output anything else.
Question: {question}
Ground truth: {answer}
Prediction: {prediction}
'''