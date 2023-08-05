DATA_PATH="../images"
S3_DEFAULT_REGION="us-east-1"
PUBLIC_S3_ACCESS=WkflU71mTlwC3aFT
PUBLIC_S3_SECRET=Ean4vWI7UGdrkUE3H3Mjt1ZRWquRh2QD
PUBLIC_S3_OUTPUT="s3://s3.apps-crc.testing:80/clearml-public-datasets"
PUBLIC_CLEARML_ACCESS_KEY=S8LDGKW3WMJUVFZNBFEH
PUBLIC_CLEARML_SECRET_KEY=LQLHnwMHa1OAzy5NhY7av86l0mIaw8eXkUOATHS9qxHEZi6KTm
PROJECT_NAME="PublicDatasets"
TASK_NAME="upload-mnist"

# create dataset
python create_ds.py \
	--project "$PROJECT_NAME" \
	--task "$TASK_NAME" \
	--s3-access "$PUBLIC_S3_ACCESS" \
	--s3-secret "$PUBLIC_S3_SECRET" \
	--s3-region "$S3_DEFAULT_REGION" \
	--s3-output "$PUBLIC_S3_OUTPUT" \
	--data-path "$DATA_PATH"

