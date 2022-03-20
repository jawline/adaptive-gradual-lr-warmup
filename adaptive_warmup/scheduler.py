from torch.optim.lr_scheduler import _LRScheduler

class Scheduler(_LRScheduler):

    """ 
    This scheduler will increase the learning rate toward end LR from start LR until
    a criterion (lambda prev_loss, current_loss) returns a desired learning rate
    or num_steps has expired.

    criterion :: epoch -> previous_learning_rate -> previous_loss -> current_learning_rate -> current_loss -> [ final_learning_rate | None ]

    After that it will pass through steps to underlying_scheduler.
    """
    def __init__(self, optimizer, start_lr=0.00001, end_lr=0.0001, num_steps=10, criterion=None, underlying_scheduler=None, pass_through_loss_to_underlying=False):

        self.optimizer = optimizer
        self.pass_through_loss_to_underlying = pass_through_loss_to_underlying

        # Initialize this to -1 because the torch
        # super constructor seems to call step
        self.current_step = -1
        self.num_steps = num_steps

        self.start_lr = start_lr
        self.end_lr = end_lr

        self.criterion = criterion
        self.underlying_scheduler = underlying_scheduler

        self.last_lr = start_lr
        self.current_lr = start_lr

        self.current_loss = None
        self.last_loss = None

        # We also maintain finished to track if the criterion has returned true
        self.finished = False

        super(Scheduler, self).__init__(optimizer)

    def update_lr(self, to):
        self.last_lr = self.current_lr
        self.current_lr = to
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = to

    def calculate_next_lr(self):
        diff_lrs = self.end_lr - self.start_lr
        step_diff = diff_lrs / (self.num_steps - 1)
        return self.start_lr + (self.current_step * step_diff)

    def step(self, loss=None):
        print("Step")
        self.current_step += 1

        if self.current_step >= self.num_steps:
            self.finished = True

        if self.criterion != None:
            crit = self.criterion(self.current_step, self.last_lr, self.last_loss, self.current_lr, self.current_loss)
        else:
            crit = None

        if crit != None:
            self.update_lr(crit)
            self.finished = True

        if self.finished:
            if self.pass_through_loss_to_underlying:
                self.underlying_scheduler.step(loss)
            else:
                self.underlying_scheduler.step()
        else:
            self.update_lr(self.calculate_next_lr())

        self.last_loss = self.current_loss
        self.current_loss = loss
