from __future__ import print_function

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Dataset')
    parser.add_argument('--dataset-project',    type=str, help='dataset name', default='')
    parser.add_argument('--dataset-name',       type=str, help='dataset name', default='')
    parser.add_argument('--dataset-path',       type=str, help='dataset path', default='')
    parser.add_argument('--dataset-output',     type=str, help='dataset output', default='')
    parser.add_argument('--s3-access',          type=str, help='access key', default='')
    parser.add_argument('--s3-secret',          type=str, help='secret key', default='')
    parser.add_argument('--s3-region',          type=str, help='region', default='')
    args = parser.parse_args()
    return args

def create(dataset_project, dataset_name, files, output):
  from clearml import Dataset
  print('dataset_project {}, dataset_name {}, files {}, output {}'.format(dataset_project, dataset_name, files, output))
  ds = Dataset.create(dataset_project=dataset_project, dataset_name=dataset_name, output_uri=output)
  ds.add_files(path=files)
  ds.upload()
  ds.finalize()
  dataset_id = ds.id
  print('dataset created {}'.format(dataset_id))
  return dataset_id


if __name__ == '__main__':
  args = get_args()

  # set s3 credentials
  os.environ['AWS_ACCESS_KEY_ID'] = args.s3_access
  os.environ['AWS_SECRET ACCESS_KEY'] = args.s3_secret
  os.environ['AWS_DEFAULT_REGION'] = args.s3_region
  create(
    dataset_project=args.dataset_project, 
    dataset_name=args.dataset_name, 
    files=args.dataset_path, 
    output=args.dataset_output
  )
