from __future__ import print_function
import os
import argparse

from clearml import PipelineController
from pipeline_process import process
from pipeline_train import train_model
from pipeline_eval import eval_model
from pipeline_testing import test_model
from pipeline_integration import integ_test

def get_args():
    parser = argparse.ArgumentParser(description='Pipeline')
    # for pipeline
    parser.add_argument('--pipeline-project',   type=str, help='project name', default='')
    parser.add_argument('--pipeline-task',      type=str, help='task name', default='')
    parser.add_argument('--queue',              type=str, help='queue name', default='')
    # for model training
    parser.add_argument('--model-output',       type=str, help='model output', default='')
    # process
    parser.add_argument('--process-project',    type=str, help='project', default='')
    parser.add_argument('--process-task',       type=str, help='task', default='')
    parser.add_argument('--process-queue',      type=str, help='queue', default='')
    parser.add_argument('--process-dataset-id', type=str, help='dataset-id', default='')
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

    # create the pipeline controller
    pipe = PipelineController(
        project=args.pipeline_project,
        name=args.pipeline_task,
        version='1.0',
        add_pipeline_tags=True,
    )

    # set the default execution queue to be used (per step we can override the execution)
    pipe.set_default_execution_queue(args.queue)

    pipe.add_function_step(
        name='Preprocessing',
        # parents=['Create_Dataset'],  # the pipeline will automatically detect the dependencies based on the kwargs inputs
        function=process,
        function_kwargs=dict(
            project=args.process_project,
            task=args.process_task,
            queue=args.process_queue,
            image=args.image,
            output=args.output,
            container_args=args.container_args,
            dataset_id=args.process_dataset_id),
        function_return=['dataset_id'],
        cache_executed_step=True,
    )

    pipe.add_function_step(
        name='Train_Model-1',
        parents=['Preprocessing'],  # the pipeline will automatically detect the dependencies based on the kwargs inputs
        function=train_model,
        function_kwargs=dict(
            project=args.experiment_project,
            task=args.experiment_task,
            queue=args.experiment_queue,
            image=args.experiment_image,
            output_uri=args.experiment_output,
            container_args=args.experiment_container_args,
            epochs=args.epochs,
            dataset_id='${Preprocessing.dataset_id}',
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

    pipe.add_function_step(
        name='Train_Model-2',
        parents=['Preprocessing'],  # the pipeline will automatically detect the dependencies based on the kwargs inputs
        function=train_model,
        function_kwargs=dict(
            project=args.experiment_project,
            task=args.experiment_task,
            queue=args.experiment_queue,
            image=args.experiment_image,
            output_uri=args.experiment_output,
            container_args=args.experiment_container_args,
            epochs=args.epochs,
            dataset_id='${Preprocessing.dataset_id}',
            batch_size=64, 
            test_batch_size=64, 
            log_interval=10, 
            seed=1, 
            lr=0.01, 
            gamma=0.07, 
            save_model=True,
            weights=args.experiment_weights),
        function_return=['model_id'],
        cache_executed_step=True,
    )

    pipe.add_function_step(
        name='Train_Model-3',
        parents=['Preprocessing'],  # the pipeline will automatically detect the dependencies based on the kwargs inputs
        function=train_model,
        function_kwargs=dict(
            project=args.experiment_project,
            task=args.experiment_task,
            queue=args.experiment_queue,
            image=args.experiment_image,
            output_uri=args.experiment_output,
            container_args=args.experiment_container_args,
            epochs=args.epochs,
            dataset_id='${Preprocessing.dataset_id}',
            batch_size=64, 
            test_batch_size=64, 
            log_interval=10, 
            seed=1, 
            lr=0.02, 
            gamma=0.07, 
            save_model=True,
            weights=args.experiment_weights),
        function_return=['model_id'],
        cache_executed_step=True,
    )

    pipe.add_function_step(
        name='Eval_Model',
        parents=['Train_Model-1', 'Train_Model-2', 'Train_Model-3'],  # the pipeline will automatically detect the dependencies based on the kwargs inputs
        function=eval_model,
        cache_executed_step=True,
    )

    pipe.add_function_step(
        name='Robustness_Testing',
        parents=['Eval_Model'],  # the pipeline will automatically detect the dependencies based on the kwargs inputs
        function=test_model,
        cache_executed_step=True,
    )

    pipe.add_function_step(
        name='Integration_Testing',
        parents=['Robustness_Testing'],  # the pipeline will automatically detect the dependencies based on the kwargs inputs
        function=integ_test,
        cache_executed_step=True,
    )
    # For debugging purposes run on the pipeline on current machine
    # Use run_pipeline_steps_locally=True to further execute the pipeline component Tasks as subprocesses.
    #pipe.start_locally(run_pipeline_steps_locally=True)

    # Start the pipeline on the services queue (remote machine, default on the clearml-server)
    pipe.start(queue=args.queue)

    print('pipeline completed')    


if __name__ == '__main__':
    main()
