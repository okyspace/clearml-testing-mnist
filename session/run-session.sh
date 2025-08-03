<<<<<<< HEAD:interactivesession/run.sh
IMAGE=docker.io/okydocker/ubuntu:22.04-session
QUEUE=queue-project-alpha-interactivesession-2cpu-4Gram
PROJECT="aieng"
=======
IMAGE=docker.io/okydocker/ubuntu:focal-session
QUEUE=queue-session
PROJECT="[Admin] Project-A"
>>>>>>> parent of d4c67611d (Updated):session/run-session.sh

clearml-session \
	--docker "$IMAGE" \
	--queue "$QUEUE" \
	--project "$PROJECT" \
	--jupyter-lab true \
	--vscode-server true \
	--verbose \
	#--username okaiyong \
	#--password 12345 \
	# --requirements requirements.txt
	# --public-ip true \
	# --remote-gateway "192.168.130.11" \
