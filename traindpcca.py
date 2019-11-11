"""=============================================================================
Train a joint variational autoencoder with pCCA.
============================================================================="""

import argparse
import time

import torch
import torch.utils.data
from   torch.nn.utils import clip_grad_norm_
from   torch import optim
from   torch.nn import functional as F

import cuda
from   data import loader
from   models import DPCCA
import pprint

# ------------------------------------------------------------------------------

LOG_EVERY        = 10
SAVE_MODEL_EVERY = 100

mm     = torch.mm
device = cuda.device()

# ------------------------------------------------------------------------------

def main(args):
    """Main program: train -> test once per epoch while saving samples as
    needed.
    """
    start_time = time.time()
    pprint.set_logfiles(args.directory)

    pprint.log_section('Loading config.')
    cfg = loader.get_config(args.dataset)
    pprint.log_config(cfg)

    if args.full_batch != 0:
        args.batch_size = -1

    pprint.log_section('Loading script arguments.')
    pprint.log_args(args)

    pprint.log_section('Loading dataset.')
    train_loader, test_loader = loader.get_data_loaders(cfg,
                                                        args.batch_size,
                                                        args.n_workers,
                                                        args.pin_memory,
                                                        args.cv_pct)
    pprint.save_test_indices(test_loader.sampler.indices)

    model = DPCCA(cfg, args.latent_dim, args.em_iters)
    model = model.to(device)

    pprint.log_section('Model specs.')
    pprint.log_model(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    pprint.log_section('Training model.')
    for epoch in range(1, args.n_epochs + 1):

        train_msgs, grad_msgs = train(args, train_loader, model, optimizer)
        with torch.no_grad():
            test_msgs = test(cfg, args, epoch, test_loader, model)

        pprint.log_line(epoch, train_msgs, test_msgs)
        pprint.log_gradients(epoch, grad_msgs)

        if epoch % LOG_EVERY == 0:
            save_samples(args.directory, model, test_loader, cfg, epoch)
        if epoch % SAVE_MODEL_EVERY == 0:
            save_model(args.directory, model)

    hours = round((time.time() - start_time) / 3600, 1)
    pprint.log_section('Job complete in %s hrs.' % hours)

    save_model(args.directory, model)
    pprint.log_section('Model saved.')

# ------------------------------------------------------------------------------

def train(args, train_loader, model, optimizer):
    """Train PCCA model and update parameters in batches of the whole train set.
    """
    model.train()

    ae_loss1_sum  = 0
    ae_loss2_sum  = 0
    l1_loss_sum   = 0
    grad_norm_sum = 0

    for i, (x1, x2) in enumerate(train_loader):
        optimizer.zero_grad()

        x1 = x1.to(device)
        x2 = x2.to(device)
        x1r, x2r = model.forward([x1, x2])

        ae_loss1 = F.mse_loss(x1r, x1)
        ae_loss2 = F.mse_loss(x2r, x2)
        l1_loss  = decaying_l1_penalty(model, args.l1_coef)
        loss     = ae_loss1 + ae_loss2 + l1_loss

        loss.backward()
        # Perform gradient clipping *before* calling `optimizer.step()`.
        grad_norm_sum += clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        ae_loss1_sum += ae_loss1.item()
        ae_loss2_sum += ae_loss2.item()
        l1_loss_sum  += l1_loss.item()

    #negloglike     = model.neg_log_likelihood([x1, x2]) / (i+1)
    ae_loss1_sum  /= (i+1)
    ae_loss2_sum  /= (i+1)
    l1_loss_sum   /= (i+1)
    grad_norm_sum /= (i+1)

    train_msgs = [ae_loss1_sum, ae_loss2_sum, l1_loss_sum, 
                  model.pcca.iters_req]
    grad_msgs  = [grad_norm_sum]
    #nlls       = model.pcca.nlls

    return train_msgs, grad_msgs#, nlls

# ------------------------------------------------------------------------------

def test(cfg, args, epoch, test_loader, model):
    """Test model by computing the average loss on a held-out dataset. No
    parameter updates.
    """
    model.eval()

    ae_loss1_sum = 0
    ae_loss2_sum = 0
    l1_loss_sum  = 0

    for i, (x1, x2) in enumerate(test_loader):

        x1 = x1.to(device)
        x2 = x2.to(device)
        x1r, x2r = model.forward([x1, x2])

        ae_loss1 = F.mse_loss(x1r, x1)
        ae_loss2 = F.mse_loss(x2r, x2)
        l1_loss  = decaying_l1_penalty(model, args.l1_coef)

        ae_loss1_sum += ae_loss1.item()
        ae_loss2_sum += ae_loss2.item()
        l1_loss_sum  += l1_loss.item()

        if i == 0 and epoch % LOG_EVERY == 0:
            cfg.save_comparison(args.directory, x1, x1r, epoch, is_x1=True)
            cfg.save_comparison(args.directory, x2, x2r, epoch, is_x1=False)

    #negloglike    = model.neg_log_likelihood([x1, x2]) / (i+1)
    ae_loss1_sum /= (i+1)
    ae_loss2_sum /= (i+1)
    l1_loss_sum  /= (i+1)

    test_msgs = [ae_loss1_sum, ae_loss2_sum, l1_loss_sum]#, negloglike]

    return test_msgs

# ------------------------------------------------------------------------------

def decaying_l1_penalty(model, l1_coef):
    """Compute L1 penalty with decaying coefficient applied to columns. For
    implementation details, see:

    https://discuss.pytorch.org/t/simple-l2-regularization/139
    """
    if l1_coef == 0:
        return torch.tensor(0, device=device).float()

    reg_loss = 0
    for param in model.pcca.parameters_('y2'):
        for i, col in enumerate(param.t()):
            neg_i       = torch.tensor(-i, device=device).float()
            col_l1_coef = l1_coef * torch.exp(neg_i)
            reg_loss   += col_l1_coef * torch.norm(col, 1)
    return reg_loss

# ------------------------------------------------------------------------------

def l1_penalty(model, l1_coef):
    """Compute L1 penalty. For implementation details, see:

    https://discuss.pytorch.org/t/simple-l2-regularization/139
    """
    reg_loss = 0
    for param in model.pcca.parameters_('y2'):
        reg_loss += torch.norm(param, 1)
    return l1_coef * reg_loss

# ------------------------------------------------------------------------------

def save_samples(directory, model, test_loader, cfg, epoch):
    """Save samples from test set.
    """
    with torch.no_grad():
        n  = len(test_loader.sampler.indices)
        x1_batch = torch.Tensor(n, cfg.N_CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE)
        x2_batch = torch.Tensor(n, cfg.N_GENES)
        labels   = []

        for i in range(n):

            j = test_loader.sampler.indices[i]

            x1, x2 = test_loader.dataset[j]
            lab    = test_loader.dataset.labels[j]
            x1_batch[i] = x1
            x2_batch[i] = x2
            labels.append(lab)

        x1_batch = x1_batch.to(device)
        x2_batch = x2_batch.to(device)

        cfg.save_samples(directory, model, epoch, x1_batch, x2_batch, labels)

# ------------------------------------------------------------------------------

def save_model(directory, model):
    """Save PyTorch model's state dictionary for provenance.
    """
    fpath = '%s/model.pt' % directory
    state = model.state_dict()
    torch.save(state, fpath)

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser()

    p.add_argument('--directory',  type=str,   default='experiments/test')
    p.add_argument('--n_gpus',     type=int,   default=1)
    p.add_argument('--wall_time',  type=int,   default=24)
    p.add_argument('--seed',       type=int,   default=0)

    p.add_argument('--dataset',    type=str,   default='gtexv8')
    p.add_argument('--batch_size', type=int,   default=128)
    p.add_argument('--n_epochs',   type=int,   default=100)
    p.add_argument('--cv_pct',     type=float, default=0.1)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--latent_dim', type=int,   default=10)
    p.add_argument('--l1_coef',    type=float, default=0.0)
    p.add_argument('--em_iters',   type=int,   default=1)

    # If 1, train using entire dataset in one go.
    p.add_argument('--full_batch', type=int,   default=0)

    # For now, I am fixing this hyperparameter. My primary goal is to prevent
    # the gradients from exploding to extremely large numbers. The authors of
    # the original paper ("On the difficulty of training Recurrent Neural
    # Networks") might agree, given that they write:
    #
    #     "In our experiments we have noticed that for a given task and model
    #     size, training is not very sensitive to this hyperparameter."
    #
    p.add_argument('--clip',       type=float, default=1)

    args, _ = p.parse_known_args()

    args.n_workers       = 4 * args.n_gpus
    args.pin_memory      = torch.cuda.is_available()

    # For easy debugging locally.
    if args.directory == 'experiments/test':
        LOG_EVERY = 1
        SAVE_MODEL_EVERY = 5

    torch.manual_seed(args.seed)
    main(args)
