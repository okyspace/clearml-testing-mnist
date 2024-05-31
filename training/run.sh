# clearml settings
CLEARML_PROJECT="Project-Alpha"
CLEARML_TASK="mnist"
# queue name can be found in clearml ui
# CLEARML_QUEUE="queue-project-alpha-4cpu-8Gram"
CLEARML_QUEUE="queue-project-alpha-1gpu"
# store model in own project bucket
CLEARML_OUTPUT="s3://s3.apps-crc.testing:80/project-alpha/models"
# image for ml training
CLEARML_IMAGE="docker.io/okydocker/pytorch:1.13.1-cuda11.6-cudnn8-runtime"

# experiment settings
BATCH_SIZE=16
TEST_BATCH_SIZE=16
EPOCHS=8
LEARNING_RATE=0.1
DATASET_ID=7a95faebeb344a8094d7b62edf4798f3
MODEL_FILENAME="mnist.pt"

# train a model
python train.py \
	--clearml-project "$CLEARML_PROJECT" \
	--clearml-task "$CLEARML_TASK" \
	--clearml-queue "$CLEARML_QUEUE" \
	--clearml-output "$CLEARML_OUTPUT" \
	--clearml-image "$CLEARML_IMAGE" \
	--batch-size "$BATCH_SIZE" \
	--test-batch-size "$TEST_BATCH_SIZE" \
	--epochs "$EPOCHS" \
	--learning-rate "$LEARNING_RATE" \
	--dataset-id "$DATASET_ID" \
	--model-filename "$MODEL_FILENAME" \
