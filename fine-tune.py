
import preprocess
import time
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
base_job_name = "md-assistant-"
checkpoint_s3_uri = f's3://{sm_sess.default_bucket()}/{base_job_name}/checkpoints'

# training machine instance type
sm_instance_type = 'ml.p4d.24xlarge'

# define Training Job Name 
job_name = base_job_name + f'{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}'

# hyperparameters, which are passed into the training job
hyperparameters ={
  'model_id': model_id,                             # pre-trained model
  'dataset_path': '/opt/ml/input/data/training',    # path where sagemaker will save training dataset
  'epochs': 3,                                      # number of training epochs
  'per_device_train_batch_size': 2,                 # batch size for training
  'lr': 2e-4,                                       # learning rate used during training
  'hf_token': HfFolder.get_token(),                 # huggingface token to access llama 2
  'merge_weights': True,                            # wether to merge LoRA into the model (needs more memory)
}

# create the Estimator
huggingface_estimator = HuggingFace(
    entry_point          = 'run_clm.py',      # train script
    source_dir           = 'scripts',         # directory which includes all the files needed for training
    instance_type        = sm_instance_type,   # instances type used for the training job
    instance_count       = 1,                 # the number of instances used for training
    base_job_name        = job_name,          # the name of the training job
    role                 = sm_role,           # Iam role used in training job to access AWS ressources, e.g. S3
    use_spot_instances   = True,              # aws spot instance **kwargs
    max_wait             = 40000,             # aws spot instance **kwargs
    max_run              = 36000,             # aws spot instance **kwargs
    checkpoint_s3_uri    = checkpoint_s3_uri, # s3 uri where our checkpoints will be uploaded during training
    volume_size          = 300,               # the size of the EBS volume in GB
    transformers_version = '4.28',            # the transformers version used in the training job
    pytorch_version      = '2.0',             # the pytorch_version version used in the training job
    py_version           = 'py310',           # the python version used in the training job
    hyperparameters      =  hyperparameters,  # the hyperparameters passed to the training job
    environment          = { "HUGGINGFACE_HUB_CACHE": "/tmp/.cache" }, # set env variable to cache models in /tmp
)

# define a data input dictonary with our uploaded s3 uris
training_input_path = f's3://{sm_sess.default_bucket()}/processed/llama/md_dialouge/train'
data = {'training': training_input_path}
print(training_input_path)

# starting the train job with our uploaded datasets as input
huggingface_estimator.fit(data, wait=True)






