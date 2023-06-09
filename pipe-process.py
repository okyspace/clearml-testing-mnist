from __future__ import print_function

from clearml import Task, Logger
from clearml import Dataset

clearml_project = "PublicProjects"
clearml_task = "process-mnist"
queue = 'queue-cpu-only'
output = "s3://minio.apps-crc.testing:80/clearml-datasets/mnist"
dataset_id = 'c1bafd4f901045afaceb27c18bc6ab18'
docker_args = "--env AWS_ACCESS_KEY_ID=34VtVODme9ZwMQMK --env AWS_SECRET_ACCESS_KEY=m0RIai6tzDpyTttKPg7OvSQjdcbemK0h --env TRAINS_AGENT_GIT_USER='' --env TRAINS_AGENT_GIT_PASS='' --env GIT_SSL_NO_VERIFY=true"
image = "default-route-openshift-image-registry.apps-crc.testing/clearml-agent/ubuntu:bionic"

def main():
    task = Task.init(project_name=clearml_project, task_name=clearml_task, output_uri=output)
    task.set_base_docker(docker_image=image, docker_arguments=docker_args)
    task.execute_remotely(queue_name=queue, exit_process=True)

    import argparse

    def get_args():
      # process settings
      parser = argparse.ArgumentParser(description='MNIST Data Processing')
      parser.add_argument('--datasets-id', default=dataset_id)
      args = parser.parse_args()
      return args

    args = get_args()
    print('dataset {}'.format(args.datasets_id))


if __name__ == '__main__':
    main()
