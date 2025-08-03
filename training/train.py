from __future__ import print_function

from clearml import Task, Logger
from clearml import Dataset
from clearml import OutputModel
import os

def main(
    clearml_project,
    clearml_task,
    clearml_output,
    clearml_image,
    clearml_queue,
    clearml_dataset_id,
    batch_size,
    test_batch_size,
    lr,
    gamma,
    save_model,
    model_filename,
    dry_run,
    log_interval,
    no_cuda,
    seed,
    epochs
):
	###################################################################################################################
	######## These are the three lines of codes added to the ML training codes for clearml to execute your task. ######
	###################################################################################################################
    # create clearml task in defined project and specify where the codes will output to
    task = Task.init(project_name=clearml_project, task_name=clearml_task, output_uri=clearml_output)
    # set the container to be used; the ml training will be done in the container
    task.set_base_docker(docker_image=clearml_image)
    # set the clearml queue. the queue will have a defined specs (cpu, ram, gpu) and 
    # clearml will spin up a pod based on the defined specs to run the container set earlier.
    task.execute_remotely(queue_name=clearml_queue, exit_process=True)
    ###################################################################################################################

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.optim.lr_scheduler import StepLR
    #from torch.utils.tensorboard import SummaryWriter

    class Net(nn.Module):
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


    def train(model, device, train_loader, optimizer, epoch, log_interval, dry_run):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                Logger.current_logger().report_scalar(
                    'train', 'loss', iteration=(epoch * len(train_loader) + batch_idx), value=loss.item())                    
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                if dry_run:
                    break


    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        Logger.current_logger().report_scalar(
            'test', 'loss', iteration=epoch, value=test_loss)
        Logger.current_logger().report_scalar(
            'test', 'accuracy', iteration=epoch, value=(correct / len(test_loader.dataset)))

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    use_cuda = not no_cuda
    torch.manual_seed(seed)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print('device {}'.format(device))

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Grayscale(num_output_channels=1)
    ])

    if clearml_dataset_id is not None:
        # get data from clearml datasets
        dataset_path = Dataset.get(dataset_id=clearml_dataset_id)
        dataset_path = dataset_path.get_local_copy()
        train_ds = datasets.ImageFolder(root=os.path.join(dataset_path, 'train'), transform=transform)
        test_ds = datasets.ImageFolder(root=os.path.join(dataset_path, 'test'), transform=transform)
    else: 
        # get from online MNIST
        train_ds = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)
        test_ds = datasets.MNIST('../data', train=False,
                        transform=transform)

    # get data loader
    train_loader = torch.utils.data.DataLoader(train_ds,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_ds, **test_kwargs)

    # get model and optimizer
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    # train    
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, log_interval, dry_run)
        test(model, device, test_loader)
        scheduler.step()

    # save model weights
    if save_model:
        torch.jit.script(model).save(model_filename)
        # OutputModel().update_weights(weights_file)
        # save in torchscipt instead. 
        # torch.save(model.state_dict(), args.model_filename)

def get_args():
    import argparse
    # clearml settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST')
    parser.add_argument('--clearml-project', 
                    help='enter the project in clearml; must be a project that you have access rights')
    parser.add_argument('--clearml-task', 
                    help='enter the task name')
    parser.add_argument('--clearml-queue', 
                    help='enter the queue to orchestrate the clearml task')
    parser.add_argument('--clearml-image', 
                    help='enter the image to be used for model training')
    parser.add_argument('--clearml-output', 
                    help='enter the s3 bucket to store experiment outputs')
    # experiment settings        
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                    help='input batch size for testing (default: 16)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                    help='number of epochs to train (default: 14)')
    parser.add_argument('--learning-rate', type=float, default=1.0, metavar='LR',
                    help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                    help='For Saving the current Model')
    parser.add_argument('--model-filename', default="mnist.pt",
                    help='file name for model')
    parser.add_argument('--clearml-dataset-id', default=None,
                    help='clearml dataset id')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(
        args.clearml_project,
        args.clearml_task,
        args.clearml_output,
        args.clearml_image,
        args.clearml_queue,
        args.clearml_dataset_id,
        args.batch_size,
        args.test_batch_size,
        args.learning_rate,
        args.gamma,
        args.save_model,
        args.model_filename,
        args.dry_run,
        args.log_interval,
        args.no_cuda,
        args.seed,
        args.epochs
    )
