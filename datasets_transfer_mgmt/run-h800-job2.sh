# clearml settings
CLEARML_PROJECT="project-aieng"
CLEARML_TASK="mnist-h800-job2"
CLEARML_QUEUE="queue-project-h800-1cpu-2gbram"
CLEARML_OUTPUT="s3://s3.apps-crc.testing:80/clearml-data/models"
CLEARML_IMAGE="docker.io/python:3.11"

# experiment settings
BATCH_SIZE=8
TEST_BATCH_SIZE=8
EPOCHS=2
LEARNING_RATE=0.1
DATASET_ID=b6154cb53b984b84a94562a73c67f947
MODEL_FILENAME="mnist.pt"

# train a model
python train.py \
	--clearml-project "$CLEARML_PROJECT" \
	--clearml-task "$CLEARML_TASK" \
	--clearml-queue "$CLEARML_QUEUE" \
	--clearml-output "$CLEARML_OUTPUT" \
	--clearml-image "$CLEARML_IMAGE" \
	--clearml-dataset-id "$DATASET_ID" \
	--batch-size "$BATCH_SIZE" \
	--test-batch-size "$TEST_BATCH_SIZE" \
	--epochs "$EPOCHS" \
	--learning-rate "$LEARNING_RATE" \
	--model-filename "$MODEL_FILENAME" \
