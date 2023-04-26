from clearml import Task


PROJECT = 'PublicHyperparams/HyperparamsTuningByScripts'
TASK = 'optimising-mnist-public'
QUEUE = 'queue-public'
TASK_ID = "c8380c78003343c7964c8d025a82c009"

image = "default-route-openshift-image-registry.apps-crc.testing/clearml-agent/ubuntu:focal-sessions"
docker_args = "--env AWS_ACCESS_KEY_ID=bZPfiMnVhRsuJfTl --env AWS_SECRET_ACCESS_KEY=qpCTdahpdAH2Skw79QB7sDwVLGbdECXg  --env GIT_SSL_NO_VERIFY=false"

def check_task(task_id):
	print('Task ID {}'.format(task_id))
	t = Task.get_task(task_id=task_id)
	print('metrics {}'.format(t.get_last_scalar_metrics()))


def main():
	task = Task.init(
		project_name=PROJECT,
		task_name=TASK,
		task_type=Task.TaskTypes.optimizer,
		reuse_last_task_id=False
	)

	task.set_base_docker(docker_image=image, docker_arguments=docker_args)
	task.execute_remotely(queue_name=QUEUE, exit_process=True)

	from clearml.automation.optuna import OptimizerOptuna
	from clearml.automation.hpbandster import OptimizerBOHB
	from clearml.automation import (DiscreteParameterRange, HyperParameterOptimizer, RandomSearch, UniformIntegerParameterRange)

	def get_search_strategy():
		'''
		optuna, bohb, random uniform sampling, full grid search, custom
		'''
		try:
			aSearchStrategy = OptimizerOptuna
		except ImportError as ex:
			try:
				aSearchStrategy = OptimizerBOHB
			except ImportError as ex:
				aSearchStrategy = RandomSearch
				print('Random Search used...')
		return aSearchStrategy

	def get_best_params(
		job_id, 
		objective_value, 
		objective_iteration, 
		job_params, 
		top_perf_job_id
	):
		print('Job completed.', job_id, objective_value, objective_iteration, job_params, top_perf_job_id)

	ss = get_search_strategy()

	# set up arguments
	args = { 'template_task_id': TASK_ID }

	# add to task
	args = task.connect(args)

	# create optimizer object
	optimizer = HyperParameterOptimizer(
		base_task_id=args['template_task_id'],
		hyper_parameters=[
			# UniformIntegerParameterRange('Args/lr', min_value=0.5, max_value=1.0, step_size=0.1),
			DiscreteParameterRange('Args/batch_size', values=[96, 128]),
		],
		objective_metric_title='test',
		objective_metric_series='accuracy',
		objective_metric_sign='max',
		max_number_of_current_tasks=2,
		optimizer_class=ss,
		execution_queue=QUEUE,
		time_limit_per_job=10.,
		pool_period_min=0.1,
		total_max_jobs=10,
		min_iteration_per_job=10,
		max_iteration_per_job=30,
	)


	# set report period and start optimisation
	optimizer.set_report_period(0.2)
	optimizer.start(job_complete_callback=get_best_params)
	optimizer.set_time_limit(in_minutes=90.0)
	optimizer.wait()
	top_exp = optimizer.get_top_experiments(top_k=3)
	print([t.id for t in top_exp])
	optimizer.stop()

	print('Done')


if __name__ == '__main__':
	check_task(TASK_ID)
	main()
