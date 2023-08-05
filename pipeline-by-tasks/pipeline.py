from clearml import Task
from clearml.automation import PipelineController

PIPELINE_NAME = "Public Pipelines"
PIPELINE_PROJECT = "Pipeline Project"
PIPELINE_VERSION = "0.0.1"
PROCESS_TASK_ID = "179462cc8381430e87792b94c5068b20"
TRAINING_TASK_ID = "2df9b231398f4e15a95ec608d65c9fb2"
QUEUE = 'queue-cpu-only'

def main():

	pipe = PipelineController(
		name=PIPELINE_NAME,
		project=PIPELINE_PROJECT,
		version=PIPELINE_VERSION,
		add_pipeline_tags=True
	)

	pipe.set_default_execution_queue(QUEUE)

	pipe.add_step(
		name="stage_process",
		parents=[],
		base_task_id=PROCESS_TASK_ID
	)

	pipe.add_step(
		name="stage_train",
		parents=["stage_process"],
		base_task_id=TRAINING_TASK_ID
	)

	pipe.start_locally()

	# pipe.start()
	print("Pipeline done !!!!")

if __name__ == '__main__':
  main()
