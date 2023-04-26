IMAGE=default-route-openshift-image-registry.apps-crc.testing/clearml-agent/ubuntu:focal-sessions-5
QUEUE=queue-session
PROJECT="PublicInteractiveSessions"

clearml-session \
	--docker "$IMAGE" \
	--queue "$QUEUE" \
	--project "$PROJECT" \
	--jupyter-lab true \
	--vscode-server true
	# --requirements requirements.txt
	# --public-ip true \
	# --remote-gateway "192.168.130.11" \
