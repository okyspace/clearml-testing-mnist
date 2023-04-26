IMAGE=ubuntu:focal
OPENSHIFT_IMAGE=default-route-openshift-image-registry.apps-crc.testing/clearml-agent/ubuntu:focal-sessions-5
CODE_SERVER_URL=https://github.com/coder/code-server/releases/download/v4.11.0/code-server_4.11.0_amd64.deb

podman build \
	-t $OPENSHIFT_IMAGE \
	--build-arg $IMAGE \
	--build-arg CODE_SERVER_URL=$CODE_SERVER_URL \
	-f containerfile

# login
podman login -u kubeadmin -p $(oc whoami -t) --tls-verify=false default-route-openshift-image-registry.apps-crc.testing 

# push
podman push --tls-verify=false $OPENSHIFT_IMAGE