from clearml import Dataset
import argparse

def create_datasets(dataset_project, dataset_name, files, output):
  ds = Dataset.create(dataset_project=dataset_project, dataset_name=dataset_name, output_uri=output)
  ds.add_files(path=files)
  ds.upload()
  ds.finalize()

def get_args():
  parser = argparse.ArgumentParser(description='Dataset')
  parser.add_argument('--project',   type=str, help='project name', default='')
  parser.add_argument('--task',      type=str, help='task name', default='')
  parser.add_argument('--s3-output', type=str, help='output_uri', default='')
  parser.add_argument('--data-path', type=str, help='data path', default='')
  args = parser.parse_args()
  return args

def main():
  args = get_args()
  id = create_datasets(args.project, args.task, args.data_path, args.s3_output)
  print('dataset created {}'.format(id))


if __name__ == '__main__':
  main()
