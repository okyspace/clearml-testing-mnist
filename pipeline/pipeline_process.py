from __future__ import print_function

def get_args():
    import argparse
    # process settings
    parser = argparse.ArgumentParser(description='MNIST Data Processing')
    parser.add_argument('--datasets-id',    default="")
    parser.add_argument('--project',        type=str, help='project name', default='')
    parser.add_argument('--task',           type=str, help='task name', default='')
    parser.add_argument('--output',         type=str, help='output', default='')
    parser.add_argument('--image',          type=str, help='image', default='')
    parser.add_argument('--container-args', type=str, help='container-args', default='')
    args = parser.parse_args()
    return args

def process(project, task, dataset_id, output, image, container_args, queue):
    from clearml import Task, Logger
    from clearml import Dataset

    task = Task.init(project_name=project, task_name=task, output_uri=output)
    task.set_base_docker(docker_image=image, docker_arguments=container_args)
    task.execute_remotely(queue_name=queue, exit_process=True)

    # DO SOMETHING
    print("Processing codes to be added....")

    return dataset_id


if __name__ == '__main__':
    args = get_args()
    process(
        project=args.project, 
        task=args.task, 
        dataset_id=args.dataset_id,
        output=args.output,
        image=args.image,
        container_args=args.container_args,
        queue=args.queue
    )
