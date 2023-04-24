from __future__ import print_function
import os
import argparse

from clearml import PipelineController
from pipeline_create_dataset import create
from pipeline_process import process
from pipeline_train import train_model
# from pipeline_deploy import deploy

def get_args():
    parser = argparse.ArgumentParser(description='Pipeline')
    # for pipeline
    parser.add_argument('--pipeline-project',   type=str, help='project name', default='')
    parser.add_argument('--pipeline-task',      type=str, help='task name', default='')
    parser.add_argument('--queue',              type=str, help='queue name', default='')
    # for dataset
    parser.add_argument('--dataset-project',    type=str, help='dataset name', default='')
    parser.add_argument('--dataset-name',       type=str, help='dataset name', default='')
    parser.add_argument('--dataset-input',      type=str, help='dataset input', default='')
    parser.add_argument('--dataset-output',     type=str, help='dataset output', default='')
    # for model training
    parser.add_argument('--model-output',       type=str, help='model output', default='')
    # common
    parser.add_argument('--s3-access',          type=str, help='access key', default='')
    parser.add_argument('--s3-secret',          type=str, help='secret key', default='')
    parser.add_argument('--s3-region',          type=str, help='region', default='')
    # process
    parser.add_argument('--process-project',    type=str, help='project', default='')
    parser.add_argument('--process-task',       type=str, help='task', default='')
    parser.add_argument('--process-queue',      type=str, help='queue', default='')
    parser.add_argument('--image',              type=str, help='image', default='')
    parser.add_argument('--container-args',     type=str, help='container-args', default='')
    parser.add_argument('--output',             type=str, help='output', default='')
    # train
    parser.add_argument('--experiment-project', type=str, help='project', default='')
    parser.add_argument('--experiment-task',    type=str, help='task', default='')
    parser.add_argument('--experiment-output',  type=str, help='output', default='')
    parser.add_argument('--experiment-queue',   type=str, help='queue', default='')
    parser.add_argument('--experiment-image',   type=str, help='image', default='')
    parser.add_argument('--experiment-container-args',  type=str, help='container-args', default='')
    parser.add_argument('--experiment-weights',         type=str, help='weights', default='')    
    parser.add_argument('--epochs',                     type=int, help='epochs', default='')   
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    # set s3 credentials
    os.environ['AWS_ACCESS_KEY_ID'] = args.s3_access
    os.environ['AWS_SECRET ACCESS_KEY'] = args.s3_secret
    os.environ['AWS_DEFAULT_REGION'] = args.s3_region

    # create the pipeline controller
    pipe = PipelineController(
        project=args.pipeline_project,
        name=args.pipeline_task,
        version='1.0',
        add_pipeline_tags=True,
    )

    # set the default execution queue to be used (per step we can override the execution)
    pipe.set_default_execution_queue('queue-cpu-only')

    pipe.add_function_step(
        name='Create_Dataset',
        function=create,
        function_kwargs=dict(
            dataset_project=args.dataset_project, 
            dataset_name=args.dataset_name, 
            files=args.dataset_input, 
            output=args.dataset_output),
        function_return=['dataset_id'],
        cache_executed_step=True,
    )
    pipe.add_function_step(
        name='Process_Dataset',
        # parents=['Create_Dataset'],  # the pipeline will automatically detect the dependencies based on the kwargs inputs
        function=process,
        function_kwargs=dict(
            project=args.process_project,
            task=args.process_task,
            queue=args.process_queue,
            image=args.image,
            output=args.output,
            container_args=args.container_args,
            dataset_id='${Create_Dataset.dataset_id}'),
        function_return=['dataset_id'],
        cache_executed_step=True,
    )
    pipe.add_function_step(
        name='Train_Model',
        # parents=['Process_Dataset'],  # the pipeline will automatically detect the dependencies based on the kwargs inputs
        function=train_model,
        function_kwargs=dict(
            project=args.experiment_project,
            task=args.experiment_task,
            queue=args.experiment_queue,
            image=args.experiment_image,
            output=args.experiment_output,
            container_args=args.experiment_container_args,
            epochs=args.epochs,
            dataset_id='${Process_Dataset.dataset_id}',
            batch_size=32, 
            test_batch_size=32, 
            log_interval=10, 
            seed=1, 
            lr=0.01, 
            gamma=0.07, 
            save_model=True,
            weights=args.experiment_weights),
        function_return=['model_id'],
        cache_executed_step=True,
    )

    # For debugging purposes run on the pipeline on current machine
    # Use run_pipeline_steps_locally=True to further execute the pipeline component Tasks as subprocesses.
    # pipe.start_locally(run_pipeline_steps_locally=True)

    # Start the pipeline on the services queue (remote machine, default on the clearml-server)
    pipe.start()

    print('pipeline completed')    


if __name__ == '__main__':
    main()
