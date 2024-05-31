PROJECT='Project-Alpha'
TASK=mnist-pipeline
QUEUE=queue-project-alpha-2cpu-2Gram

MODEL_OUTPUT="s3://s3.apps-crc.testing:80/project-alpha/models"

PROCESS_PROJECT=Project-Alpha
PROCESS_TASK=process-mnist-dataset
PROCESS_QUEUE=queue-project-alpha-2cpu-2Gram
CONTAINER_ARGS=''
DATASET=fb1cc9b35b0046f2ad590244d298f076
IMAGE="docker.io/okydocker/pytorch:1.13.1-cuda11.6-cudnn8-runtime"
OUTPUT="s3://s3.apps-crc.testing:80/project-alpha/datasets"

EXPERIMENT_PROJECT='Project-Alpha'
EXPERIMENT_TASK=train-mnist
EXPERIMENT_OUTPUT=s3://s3.apps-crc.testing:80/project-alpha/models
EXPERIMENT_IMAGE="docker.io/okydocker/pytorch:1.13.1-cuda11.6-cudnn8-runtime"
EXPERIMENT_QUEUE=queue-project-alpha-2cpu-2Gram
EXPERIMENT_CONTAINER_ARGS=''
EXPERIMENT_WEIGHTS=mnist.pt
EPOCHS=2

python3 pipeline_controller.py \
	--pipeline-project "$PROJECT" \
	--pipeline-task "$TASK" \
	--queue "$QUEUE" \
	--model-output "$MODEL_OUTPUT" \
	--process-project "$PROCESS_PROJECT" \
	--process-task "$PROCESS_TASK" \
	--process-queue "$PROCESS_QUEUE" \
	--process-dataset-id "$DATASET"\
	--container-args "$CONTAINER_ARGS" \
	--image "$IMAGE" \
	--output "$OUTPUT" \
	--experiment-project "$EXPERIMENT_PROJECT "\
	--experiment-task "$EXPERIMENT_TASK" \
	--experiment-output "$EXPERIMENT_OUTPUT" \
	--experiment-image "$EXPERIMENT_IMAGE"\
	--experiment-queue "$EXPERIMENT_QUEUE" \
	--experiment-container-args "$EXPERIMENT_CONTAINER_ARGS" \
	--experiment-weights "$EXPERIMENT_WEIGHTS" \
	--epochs "$EPOCHS"
