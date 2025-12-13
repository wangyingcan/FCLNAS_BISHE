from tensorboardX import SummaryWriter

writerTf = SummaryWriter(comment='lr')
print('tensorboardX logdir',writerTf.logdir)

def set_lr_shusen(t=0, a=1e-2):
    return (1 / 5) / (5 + (t * a))


def set_lr_google(lr_google=0.25):
    return 0.99 * lr_google


lr_google = 0.025
for round in range(250):
    lr_google = set_lr_google(lr_google=lr_google)
    writerTf.add_scalar('lr_google-round', lr_google, round)

for i in range(5 * 250):
    lr = set_lr_shusen(t=i, a=0.01)
    writerTf.add_scalar('lr-1e-2', lr, i)

    lr = set_lr_shusen(t=i, a=0.0001)
    writerTf.add_scalar('lr-1e-4', lr, i)

    lr = set_lr_shusen(t=i, a=0.000001)
    writerTf.add_scalar('lr-1e-6', lr, i)
