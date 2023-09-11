###################################################################################################################
# Deploy model fine_tuned model from s3
# @asabay 9/11/2023
###################################################################################################################
import json
from sagemaker.huggingface.model import HuggingFaceModel
from preprocess import role_name
import os

sm_instance_type='ml.g5.4xlarge'
sm_model = 's3://mlpipes-md-assistant/md-assistant-2023-09-09-03-40-27-2023-09-09-10-40-27-308/output/model.tar.gz'
number_of_gpu = 1
health_check_timeout = 1000
HF_API_TOKEN = os.getenv('HF_API_TOKEN')

# Model and Endpoint configuration
config = {
  'HF_MODEL_ID': "meta-llama/Llama-2-7b-chat-hf",           # model_id from hf.co/models
  'SM_NUM_GPUS': json.dumps(number_of_gpu),                 # Number of GPU used per replica
  'MAX_INPUT_LENGTH': json.dumps(2048),                     # Max length of input text
  'MAX_TOTAL_TOKENS': json.dumps(4096),                     # Max length of the generation (including input text)
  'MAX_BATCH_TOTAL_TOKENS': json.dumps(8192),               # Limits the number of tokens that can be processed in parallel during the generation
  'HUGGING_FACE_HUB_TOKEN': os.getenv('HF_API_TOKEN'),
  # ,'HF_MODEL_QUANTIZE': "bitsandbytes",                   # comment in to quantize
}

# check if token is set
assert config['HUGGING_FACE_HUB_TOKEN'] == HF_API_TOKEN, "Please set your Hugging Face Hub token"

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
   model_data=sm_model,                                     # path to your trained SageMaker model
   role=role_name,                                          # IAM role with permissions to create an endpoint
   transformers_version="4.28",                             # Transformers version used
   pytorch_version="2.0",                                   # PyTorch version used
   py_version='py310',                                      # Python version used
   env=config
)

# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
   initial_instance_count=1,
   instance_type=sm_instance_type,
   container_startup_health_check_timeout=health_check_timeout # 10 minutes to be able to load the model
)

# test here

# delete endpoint
print('deleting endpoint... test done')
huggingface_model.delete_model()
predictor.delete_endpoint()