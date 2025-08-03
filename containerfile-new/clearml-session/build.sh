SOURCE_IMAGE=ubuntu:focal
IMAGE=okydocker/ubuntu:focal-session-new
CODE_SERVER_URL=https://github.com/coder/code-server/releases/download/v4.97.2/code-server_4.97.2_amd64.deb

podman build \
	-t $IMAGE \
	--build-arg $SOURCE_IMAGE \
	--build-arg CODE_SERVER_URL=$CODE_SERVER_URL \
	-f containerfile

# login
podman login -u okydocker -p Ongky6085 --tls-verify=false docker.io
# podman login -u kubeadmin -p $(oc whoami -t) --tls-verify=false default-route-openshift-image-registry.apps-crc.testing 

# push
podman push --tls-verify=false $IMAGE
