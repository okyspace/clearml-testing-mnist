DATA_PATH="../images"
S3_OUTPUT="s3://s3.apps-crc.testing:443/clearml-datasets"
PROJECT_NAME="DataVerse"
TASK_NAME="mnist"

# create dataset
python create_ds.py \
	--project "$PROJECT_NAME" \
	--task "$TASK_NAME" \
	--s3-output "$S3_OUTPUT" \
	--data-path "$DATA_PATH"
