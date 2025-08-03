from __future__ import print_function
import argparse
import os
from clearml import Task, Logger, Dataset, OutputModel
import torch
import torchvision
from torchvision import datasets, transforms

def get_args():
    """
    Parse and return command-line arguments.
    """
    parser = argparse.ArgumentParser(description='PyTorch MNIST Trainer with ClearML')
    # ClearML settings
    parser.add_argument('--clearml-project', required=True,
                        help='ClearML project name')
    parser.add_argument('--clearml-task', required=True,
                        help='ClearML task name')
    parser.add_argument('--clearml-queue', required=True,
                        help='ClearML execution queue')
    parser.add_argument('--clearml-image', required=True,
                        help='Docker image for training')
    parser.add_argument('--clearml-output', 
                        help='S3 bucket/URI for experiment outputs')
    parser.add_argument('--clearml-dataset-id',
                        help='ClearML dataset ID for training data')
    # Experiment settings
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Training batch size (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=16,
                        help='Test batch size (default: 16)')
    parser.add_argument('--epochs', type=int, default=14,
                        help='Number of epochs (default: 14)')
    parser.add_argument('--learning-rate', type=float, default=1.0,
                        help='Learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7,
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use CUDA for training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='Quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=200,
                        help='Batches to wait before logging status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='Save model checkpoints')
    parser.add_argument('--model-filename', default="mnist.pt",
                        help='Filename for saving the model')
    parser.add_argument('--checkpoint-interval', type=int, default=2,
                        help='Epochs between checkpoints')
    return parser.parse_args()


def init_clearml_task(project_name, task_name, output_uri, docker_image, queue_name):
    """
    Initialize and launch a ClearML task.
    """
    task = Task.init(project_name=project_name, task_name=task_name, output_uri=output_uri)
    task.set_base_docker(docker_image=docker_image)
    task.execute_remotely(queue_name=queue_name, exit_process=True)
    return task


def pull_datasets(clearml_dataset_id, batch_size, test_batch_size):
    """
    Pull datasets from ClearML or download MNIST from the internet.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Grayscale(num_output_channels=1)
    ])

    if clearml_dataset_id:
        dataset = Dataset.get(dataset_id=clearml_dataset_id)
        local_path = dataset.get_local_copy()
        train_ds = datasets.ImageFolder(root=os.path.join(local_path, 'train'), transform=transform)
        test_ds = datasets.ImageFolder(root=os.path.join(local_path, 'test'), transform=transform)
    else:
        train_ds = datasets.MNIST('../data', train=True, download=True, transform=transform)
        test_ds = datasets.MNIST('../data', train=False, download=True, transform=transform)
        
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader


def init_training(use_cuda, learning_rate, gamma):
    """
    Initialize device, model, optimizer, and scheduler.
    """
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.optim.lr_scheduler import StepLR

    class Net(nn.Module):
        """
        Simple CNN for MNIST.
        """
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    return device, model, optimizer, scheduler


def train_model(model, device, train_loader, optimizer, epoch, log_interval, dry_run):
    """
    Training loop for one epoch.
    """
    print(f"Training epoch {epoch} ...")
    # ... Insert your training code here ...


def test_model(model, device, test_loader):
    """
    Evaluate the model on the test set.
    """
    print("Testing model ...")
    # ... Insert your testing code here ...


def save_model(model, filename):
    """
    Save the model checkpoint.
    """
    print(f"Saving model to {filename} ...")
    torch.save(model.state_dict(), filename)


def load_model(model, model_path):
    """
    Load a model checkpoint.
    """
    print(f"Loading model from {model_path} ...")
    model.load_state_dict(torch.load(model_path))
    return model


if __name__ == '__main__':
    args = get_args()
    # Set random seed
    torch.manual_seed(args.seed)

    # Initialize ClearML Task
    task = init_clearml_task(
        args.clearml_project,
        args.clearml_task,
        args.clearml_output,
        args.clearml_image,
        args.clearml_queue
    )

    # Init training components
    device, model, optimizer, scheduler = init_training(
        args.use_cuda,
        args.learning_rate,
        args.gamma
    )

    # Pull datasets
    train_loader, test_loader = pull_datasets(
        args.clearml_dataset_id,
        args.batch_size,
        args.test_batch_size
    )

    # Train/test loop
    for epoch in range(1, args.epochs + 1):
        train_model(
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            args.log_interval,
            args.dry_run
        )
        test_model(
            model,
            device,
            test_loader
        )
        scheduler.step()

        # Save model at checkpoint interval
        if epoch % args.checkpoint_interval == 0 and args.save_model:
            save_model(model, args.model_filename)
