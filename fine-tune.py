#############################################################################
# Fine-tuning script for md-assistant model
# Adapted from @philschmid hugginface-llama-2-samples
#############################################################################

import preprocess
import time
# import wandb
import os
from sagemaker.huggingface import HuggingFace
from huggingface_hub import HfFolder
from preprocess import (init_sagemaker,
                        sagemaker_session_bucket,
                        model_id,
                        role_name
                    )

# sagemaker session bucket -> used for uploading data, models and logs
# sagemaker will automatically create this bucket if it not exists

sm_sess, sm_role = init_sagemaker(role_name, sagemaker_session_bucket)
print(f"sagemaker role arn: {sm_role}")
print(f"sagemaker bucket: {sm_sess.default_bucket()}")
print(f"sagemaker session region: {sm_sess.boto_region_name}")

# s3 uri where our checkpoints will be uploaded during training
base_job_name = "md-assistant"
checkpoint_in_bucket = 'checkpoints'
bucket = f's3://{sm_sess.default_bucket()}'
checkpoint_s3_bucket="s3://{}/{}-/{}".format(sm_sess.default_bucket(), base_job_name, checkpoint_in_bucket)

# training machine instance types, uncomment/enable only 1
#sm_instance_type = 'ml.p4d.24xlarge'                 # 96 vcpu, 1152G mem, 8 GPU, 320G total gpu mem, A100
#sm_instance_type = 'ml.p3.16xlarge'                 # 64 vcpu, 488G mem, 8 gpu, 128G total gpu mem, V100
#sm_instance_type = 'ml.g5.24xlarge'                  # 96 vcpu, 384G mem, 4 gpu, 96G total gpu mem, A10G
#sm_instance_type = 'ml.g5.12xlarge'                  # 48 vcpu, 192G mem, 4 gpu, 96G total gpu mem, A10G
sm_instance_type = 'ml.g5.4xlarge'                   # 16 cvpu, 64G mem, 1 gpu, 24G total gpu mem, A10G

# define Training Job Name 
job_name = base_job_name + f'-{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}'
sm_output_dir = '/opt/ml/checkpoints'
wandb_api_key = os.getenv('WB_API_KEY')

# hyperparameters, which are passed into the training job
hyperparameters ={
  'model_id': model_id,                             # pre-trained model
  'dataset_path': '/opt/ml/input/data/training',    # path where sagemaker will save training dataset
  'epochs': 3,                                      # number of training epochs
  'per_device_train_batch_size': 2,                 # batch size for training
  'lr': 2e-4,                                       # learning rate used during training
  'hf_token': HfFolder.get_token(),                 # huggingface token to access llama 2
  'merge_weights': True,                            # wether to merge LoRA into the model (needs more memory)
  'output_dir': sm_output_dir,                       # checkpoint dir when using spot instances
  'wandb_api_key': wandb_api_key,                    # pass to training script for wandb login
  'wandb_project': base_job_name
}

# create the Estimator
huggingface_estimator = HuggingFace(
    entry_point          = 'run_clm.py',            # train script
    source_dir           = 'scripts',               # directory which includes all the files needed for training
    output_path          = bucket,                  # S3 location for saving the training result (model artifacts and output files)
    instance_type        = sm_instance_type,        # instances type used for the training job
    instance_count       = 1,                       # the number of instances used for training
    base_job_name        = job_name,                # the name of the training job
    role                 = sm_role,                 # Iam role used in training job to access AWS ressources, e.g. S3
    use_spot_instances   = True,                    # aws spot instance **kwargs
    max_wait             = 144000,                  # aws spot instance **kwargs in seconds
    max_run              = 143000,                  # aws spot instance **kwargs in seconds
    checkpoint_s3_uri    = checkpoint_s3_bucket,    # s3 uri where our checkpoints will be uploaded during training
    checkpoint_local_path= sm_output_dir,           # algorithm writes checkpoints to this local path then up to s3
    volume_size          = 300,                     # the size of the EBS volume in GB
    transformers_version = '4.28',                  # the transformers version used in the training job
    pytorch_version      = '2.0',                   # the pytorch_version version used in the training job
    py_version           = 'py310',                 # the python version used in the training job
    hyperparameters      =  hyperparameters,        # the hyperparameters passed to the training job
    environment          = { "HUGGINGFACE_HUB_CACHE": "/tmp/.cache" }, # set env variable to cache models in /tmp
)

# define a data input dictonary with our uploaded s3 uris
training_input_path = f's3://{sm_sess.default_bucket()}/processed/llama/md_dialouge/train'
data = {'training': training_input_path}
print(training_input_path)
print(f'##########output_path = {bucket}###############')
print(f'**********checkpoint s3 uri = {huggingface_estimator.checkpoint_s3_uri}**************************')
# starting the train job with our uploaded datasets as input
huggingface_estimator.fit(data, wait=True)






