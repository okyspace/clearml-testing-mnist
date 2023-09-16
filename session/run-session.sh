IMAGE=docker.io/okydocker/ubuntu:focal-session
QUEUE=queue-session
PROJECT="[Admin] Project-A"

clearml-session \
	--docker "$IMAGE" \
	--queue "$QUEUE" \
	--project "$PROJECT" \
	--jupyter-lab true \
	--vscode-server true \
	--verbose
	# --requirements requirements.txt
	# --public-ip true \
	# --remote-gateway "192.168.130.11" \
