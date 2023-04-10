DATA_PATH="images"
S3_DEFAULT_REGION="us-east-1"
PUBLIC_S3_ACCESS=iO1CDmdJRwUFbDfx
PUBLIC_S3_SECRET=blwv8gdpGnUbRecIcaWDXfmn8SyBeOeb
PUBLIC_S3_OUTPUT="s3://minio.apps-crc.testing:80/clearml-datasets"
PUBLIC_CLEARML_ACCESS_KEY=AUQMABE5729LCYOY19XS
PUBLIC_CLEARML_SECRET_KEY=FPvvyiOlTX7vhQNAYECTafwhapYMfA2ZcB6UkQbJ2j5MyDElQl
PROJECT_NAME="Public Datasets"
TASK_NAME="mnist"

# create dataset
python create_ds.py \
	--project "$PROJECT_NAME" \
	--task "$TASK_NAME" \
	--s3-access "$PUBLIC_S3_ACCESS" \
	--s3-secret "$PUBLIC_S3_SECRET" \
	--s3-region "$S3_DEFAULT_REGION" \
	--s3-output "$PUBLIC_S3_OUTPUT" \
	--data-path "$DATA_PATH"

