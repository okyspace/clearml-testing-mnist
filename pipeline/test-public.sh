PROJECT=PublicPipelines
TASK=mnist-pipeline
QUEUE=queue-public
DATASET_PROJECT=PublicDatasets
DATASET_NAME=mnist-data
DATASET_INPUT=../images
DATASET_OUTPUT="s3://minio.apps-crc.testing:80/clearml-public-datasets"
MODEL_OUTPUT="s3://minio.apps-crc.testing:80/clearml-public-models"
S3_ACCESS=K8HYLZM3F1X8KFHQX69N
S3_SECRET=snC6jiuMSQYGCAXTpEYFAJKor1hqWk91wWHp5fbx9FcD9Y9ClW
S3_REGION=us-east-1

PROCESS_PROJECT=PublicProjects
PROCESS_TASK=process-mnist
PROCESS_QUEUE=queue-public
CONTAINER_ARGS='--env AWS_ACCESS_KEY_ID=K8HYLZM3F1X8KFHQX69N --env AWS_SECRET_ACCESS_KEY=snC6jiuMSQYGCAXTpEYFAJKor1hqWk91wWHp5fbx9FcD9Y9ClW'
IMAGE=default-route-openshift-image-registry.apps-crc.testing/clearml-agent/ubuntu:focal-sessions
OUTPUT="s3://minio.apps-crc.testing:80/clearml-public-datasets"

EXPERIMENT_PROJECT=PublicProjects
EXPERIMENT_TASK=train-mnist
EXPERIMENT_OUTPUT=s3://minio.apps-crc.testing:80/clearml-public-models
EXPERIMENT_IMAGE=default-route-openshift-image-registry.apps-crc.testing/clearml-agent/ubuntu:focal-sessions
EXPERIMENT_QUEUE=queue-public
EXPERIMENT_CONTAINER_ARGS='--env AWS_ACCESS_KEY_ID=K8HYLZM3F1X8KFHQX69N --env AWS_SECRET_ACCESS_KEY=snC6jiuMSQYGCAXTpEYFAJKor1hqWk91wWHp5fbx9FcD9Y9ClW'
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
