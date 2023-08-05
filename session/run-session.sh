IMAGE=okydocker/ubuntu:focal-session
QUEUE=queue-session
PROJECT="PublicInteractiveSessions"

clearml-session \
	--docker "$IMAGE" \
	--queue "$QUEUE" \
	--project "$PROJECT" \
	--jupyter-lab true \
	--vscode-server false
	# --requirements requirements.txt
	# --public-ip true \
	# --remote-gateway "192.168.130.11" \
