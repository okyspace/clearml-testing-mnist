PROJECT='[Admin] Project-A'
TASK=mnist-pipeline
QUEUE=queue-2cpu-4GRAM

MODEL_OUTPUT="s3://s3.apps-crc.testing:443/clearml-models"

PROCESS_PROJECT=PublicProjects
PROCESS_TASK=process-mnist
PROCESS_QUEUE=queue-2cpu-4GRAM
CONTAINER_ARGS=''
DATASET=0651fa9ab0e143a99f7bf4205e60067b
IMAGE="docker.io/okydocker/pytorch:1.13.1-cuda11.6-cudnn8-runtime"
OUTPUT="s3://s3.apps-crc.testing:443/clearml-datasets"

EXPERIMENT_PROJECT='[Admin] Project-A'
EXPERIMENT_TASK=train-mnist
EXPERIMENT_OUTPUT=s3://s3.apps-crc.testing:443/clearml-models
EXPERIMENT_IMAGE="docker.io/okydocker/pytorch:1.13.1-cuda11.6-cudnn8-runtime"
EXPERIMENT_QUEUE=queue-2cpu-4GRAM
EXPERIMENT_CONTAINER_ARGS=''
EXPERIMENT_WEIGHTS=mnist.pt
EPOCHS=2

python3 pipeline_controller.py \
	--pipeline-project "$PROJECT" \
	--pipeline-task $TASK \
	--queue $QUEUE \
	--model-output $MODEL_OUTPUT \
	--process-project "$PROCESS_PROJECT" \
	--process-task $PROCESS_TASK \
	--process-queue $PROCESS_QUEUE \
	--process-dataset-id $DATASET \
	--container-args "$CONTAINER_ARGS" \
	--image $IMAGE \
	--output $OUTPUT \
	--experiment-project "$EXPERIMENT_PROJECT "\
	--experiment-task $EXPERIMENT_TASK \
	--experiment-output $EXPERIMENT_OUTPUT \
	--experiment-image $EXPERIMENT_IMAGE \
	--experiment-queue $EXPERIMENT_QUEUE \
	--experiment-container-args "$EXPERIMENT_CONTAINER_ARGS" \
	--experiment-weights $EXPERIMENT_WEIGHTS \
	--epochs $EPOCHS
