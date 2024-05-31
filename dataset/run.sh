DATA_PATH="../images"
S3_OUTPUT="s3://s3.apps-crc.testing:80/clearml-datasets"
PROJECT_NAME="Public-Datasets"
TASK_NAME="mnist"

# create dataset
python create_datasets.py \
	--project "$PROJECT_NAME" \
	--task "$TASK_NAME" \
	--s3-output "$S3_OUTPUT" \
	--data-path "$DATA_PATH"
