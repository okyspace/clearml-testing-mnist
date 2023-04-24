PROJECT=PublicPipelines
TASK=mnist-pipeline
QUEUE=queue-cpu-only
DATASET_PROJECT=PublicDatasets
DATASET_NAME=mnist-data
DATASET_INPUT=../images
DATASET_OUTPUT="s3://minio.apps-crc.testing:80/clearml-datasets"
MODEL_OUTPUT="s3://minio.apps-crc.testing:80/clearml-models"
S3_ACCESS=S8LDGKW3WMJUVFZNBFEH
S3_SECRET=LQLHnwMHa1OAzy5NhY7av86l0mIaw8eXkUOATHS9qxHEZi6KTm
S3_REGION=us-east-1

PROCESS_PROJECT=PublicProjects
PROCESS_TASK=process-mnist
PROCESS_QUEUE=queue-cpu-only
CONTAINER_ARGS='--env AWS_ACCESS_KEY_ID=34VtVODme9ZwMQMK --env AWS_SECRET_ACCESS_KEY=m0RIai6tzDpyTttKPg7OvSQjdcbemK0h'
IMAGE=default-route-openshift-image-registry.apps-crc.testing/clearml-agent/ubuntu:focal-sessions
OUTPUT="s3://minio.apps-crc.testing:80/clearml-datasets"

EXPERIMENT_PROJECT=PublicProjects
EXPERIMENT_TASK=train-mnist
EXPERIMENT_OUTPUT=s3://minio.apps-crc.testing:80/clearml-models
EXPERIMENT_IMAGE=default-route-openshift-image-registry.apps-crc.testing/clearml-agent/ubuntu:focal-sessions
EXPERIMENT_QUEUE=queue-cpu-only
EXPERIMENT_CONTAINER_ARGS='--env AWS_ACCESS_KEY_ID=34VtVODme9ZwMQMK --env AWS_SECRET_ACCESS_KEY=m0RIai6tzDpyTttKPg7OvSQjdcbemK0h'
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
