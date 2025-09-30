from __future__ import print_function

from clearml import Task, Logger
from clearml import Dataset
from clearml import OutputModel
import os


clearml_project = "[Admin] Project-A"
clearml_task = "train-mnist"
queue = 'queue-2cpu-4GRAM'
output = ""
clearml_dataset_id = None

image = "docker.io/okydocker/pytorch:1.13.1-cuda11.6-cudnn8-runtime"
weights_file = "mnist.pt"

def main():
    task = Task.init(project_name=clearml_project, task_name=clearml_task, output_uri=output)
    task.set_base_docker(docker_image=image)
    task.execute_remotely(queue_name=queue, exit_process=True)

    import argparse
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


    def train(args, model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                Logger.current_logger().report_scalar(
                    'train', 'loss', iteration=(epoch * len(train_loader) + batch_idx), value=loss.item())                    
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                if args.dry_run:
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

    def get_args():
        # Training settings
        parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
        parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
        parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
        parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 14)')
        parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
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
        parser.add_argument('--datasets-id', default=clearml_dataset_id)
        args = parser.parse_args()
        return args

    args = get_args()
    use_cuda = not args.no_cuda
    torch.manual_seed(args.seed)

    print('datasets_id {}'.format(args.datasets_id))

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print('device {}'.format(device))

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
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

    train_loader = torch.utils.data.DataLoader(train_ds,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_ds, **test_kwargs)

    # get model
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # train
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    # save weights
    if args.save_model:
        torch.jit.script(model).save(weights_file)
        OutputModel().update_weights(weights_file)
        # save in torchscipt instead. 


if __name__ == '__main__':
    main()
