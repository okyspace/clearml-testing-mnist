from clearml import Task
from clearml.automation import PipelineController

PIPELINE_NAME = "Public Pipelines"
PIPELINE_PROJECT = "Pipeline Project"
PIPELINE_VERSION = "0.0.1"
PROCESS_TASK_ID = "d19cf78a082b4da9b38088855eef50b7"
TRAINING_TASK_ID = "25525276e3174916ac6bc7446205bbc6"
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
