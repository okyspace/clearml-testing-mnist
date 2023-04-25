PROJECT=ProjectA/Pipelines
TASK=mnist-pipeline
QUEUE=queue-a
DATASET_PROJECT=ProjectA/Datasets
DATASET_NAME=mnist-data
DATASET_INPUT=../images
DATASET_OUTPUT="s3://minio.apps-crc.testing:80/proj-a/datasets"
MODEL_OUTPUT="s3://minio.apps-crc.testing:80/proj-a/models"
S3_ACCESS=SZGMY3YZOHB53G7V0MPA
S3_SECRET=4wYKmhIUr5ekGz0lATUdg6oD8IucIkUftHBwV0MOeX5bTrxZBs
S3_REGION=us-east-1

PROCESS_PROJECT=ProjectA/Experiments
PROCESS_TASK=process-mnist
PROCESS_QUEUE=queue-a
CONTAINER_ARGS='--env AWS_ACCESS_KEY_ID=SZGMY3YZOHB53G7V0MPA --env AWS_SECRET_ACCESS_KEY=4wYKmhIUr5ekGz0lATUdg6oD8IucIkUftHBwV0MOeX5bTrxZBs'
IMAGE=default-route-openshift-image-registry.apps-crc.testing/clearml-agent/ubuntu:focal-sessions
OUTPUT="s3://minio.apps-crc.testing:80/clearml-datasets"

EXPERIMENT_PROJECT=ProjectA/Experiments
EXPERIMENT_TASK=train-mnist
EXPERIMENT_OUTPUT=s3://minio.apps-crc.testing:80/clearml-models
EXPERIMENT_IMAGE=default-route-openshift-image-registry.apps-crc.testing/clearml-agent/ubuntu:focal-sessions
EXPERIMENT_QUEUE=queue-a
EXPERIMENT_CONTAINER_ARGS='--env AWS_ACCESS_KEY_ID=SZGMY3YZOHB53G7V0MPA --env AWS_SECRET_ACCESS_KEY=4wYKmhIUr5ekGz0lATUdg6oD8IucIkUftHBwV0MOeX5bTrxZBs'
EXPERIMENT_WEIGHTS=mnist.pt
EPOCHS=1

python3 pipeline_controller.py \
	--pipeline-project $PROJECT \
	--pipeline-task $TASK \
	--queue $QUEUE \
	--dataset-project $DATASET_PROJECT \
	--dataset-name $DATASET_NAME \
	--dataset-input $DATASET_INPUT \
	--dataset-output $DATASET_OUTPUT \
	--model-output $MODEL_OUTPUT \
	--s3-access $S3_ACCESS \
	--s3-secret $S3_SECRET \
	--s3-region $S3_REGION \
	--process-project $PROCESS_PROJECT \
	--process-task $PROCESS_TASK \
	--process-queue $PROCESS_QUEUE \
	--container-args "$CONTAINER_ARGS" \
	--image $IMAGE \
	--output $OUTPUT \
	--experiment-project $EXPERIMENT_PROJECT \
	--experiment-task $EXPERIMENT_TASK \
	--experiment-output $EXPERIMENT_OUTPUT \
	--experiment-image $EXPERIMENT_IMAGE \
	--experiment-queue $EXPERIMENT_QUEUE \
	--experiment-container-args "$EXPERIMENT_CONTAINER_ARGS" \
	--experiment-weights $EXPERIMENT_WEIGHTS \
	--epochs $EPOCHS
