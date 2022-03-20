import torch

from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.optim.sgd import SGD

from scheduler import Scheduler

def lr_criterion(epoch, last_lr, last_loss, current_lr, current_loss):
    if epoch > 2:
        if last_loss < current_loss:
            return last_lr
        else:
            return None
    else:
        return None

if __name__ == '__main__':
    model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
    optim = SGD(model, 0.1)

    scheduler_steplr = StepLR(optim, step_size=10, gamma=0.1)
    scheduler_warmup = Scheduler(optim, start_lr=0.0001, end_lr=0.01, num_steps=100, criterion=lr_criterion, underlying_scheduler=scheduler_steplr)

    loss = torch.Tensor([0])

    for epoch in range(0, 100):
        print("Stepped")
        print(epoch, optim.param_groups[0]['lr'])

        if epoch > 65:
            loss = torch.Tensor([1])

        scheduler_warmup.step(loss)
        optim.step()
    print(epoch, optim.param_groups[0]['lr'])
