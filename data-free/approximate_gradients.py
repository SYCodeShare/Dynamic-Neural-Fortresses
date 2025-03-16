import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models


def estimate_gradient_objective(args, victim_model, clone_model, x, epsilon = 1e-7, m = 5, verb=False, num_classes=10, device = "cpu", pre_x=False):

    if pre_x and args.G_activation is None:
        raise ValueError(args.G_activation)

    clone_model.eval()
    victim_model.eval()
    with torch.no_grad():
        # Sample unit noise vector
        N = x.size(0)
        C = x.size(1)
        S = x.size(2)
        dim = S**2 * C

        u = np.random.randn(N * m * dim).reshape(-1, m, dim) # generate random points from normal distribution

        d = np.sqrt(np.sum(u ** 2, axis = 2)).reshape(-1, m, 1)  # map to a uniform distribution on a unit sphere
        u = torch.Tensor(u / d).view(-1, m, C, S, S)
        u = torch.cat((u, torch.zeros(N, 1, C, S, S)), dim = 1) # Shape N, m + 1, S^2

            

        u = u.view(-1, m + 1, C, S, S)

        evaluation_points = (x.view(-1, 1, C, S, S).cpu() + epsilon * u).view(-1, C, S, S)
        if pre_x: 
            evaluation_points = args.G_activation(evaluation_points) # Apply args.G_activation function

        # Compute the approximation sequentially to allow large values of m
        pred_victim = []
        pred_clone = []
        max_number_points = 32*156  # Hardcoded value to split the large evaluation_points tensor to fit in GPU
        
        for i in (range(N * m // max_number_points + 1)): 
            pts = evaluation_points[i * max_number_points: (i+1) * max_number_points]
            pts = pts.to(device)
            pred_victim_pts, j = victim_model(pts,return_all_outputs = True) 
            num_exits = len(pred_victim_pts)
            batch_size = len(pred_victim_pts[0])
            sample_indices = torch.arange(batch_size)
            stacked_output = torch.stack(pred_victim_pts, dim=0)
            selected_outputs = stacked_output[j, sample_indices]
            pred_victim_pts=selected_outputs
            pred_victim_pts.detach() 
            pred_clone_pts = clone_model(pts)
            pred_victim.append(pred_victim_pts)
            pred_clone.append(pred_clone_pts)
        pred_victim = torch.cat(pred_victim, dim=0).to(device)
        pred_clone = torch.cat(pred_clone, dim=0).to(device)
        u = u.to(device)
        if args.loss == "l1":
            loss_fn = F.l1_loss
            if args.no_logits:
                pred_victim = F.log_softmax(pred_victim, dim=1).detach()
                if args.logit_correction == 'min':
                    pred_victim -= pred_victim.min(dim=1).values.view(-1, 1).detach()
                elif args.logit_correction == 'mean':
                    pred_victim -= pred_victim.mean(dim=1).view(-1, 1).detach()


        elif args.loss == "kl":
            loss_fn = F.kl_div
            pred_clone = F.log_softmax(pred_clone, dim=1)
            pred_victim = F.softmax(pred_victim.detach(), dim=1)

        else:
            raise ValueError(args.loss)

        # Compute loss
        if args.loss == "kl":
            loss_values = - loss_fn(pred_clone, pred_victim, reduction='none').sum(dim = 1).view(-1, m + 1) 
        else:
            loss_values = - loss_fn(pred_clone, pred_victim, reduction='none').mean(dim = 1).view(-1, m + 1) 

        # Compute difference following each direction
        differences = loss_values[:, :-1] - loss_values[:, -1].view(-1, 1)
        differences = differences.view(-1, m, 1, 1, 1)

        # Formula for Forward Finite Differences
        gradient_estimates = 1 / epsilon * differences * u[:, :-1]
        if args.forward_differences:
            gradient_estimates *= dim            

        if args.loss == "kl":
            gradient_estimates = gradient_estimates.mean(dim = 1).view(-1, C, S, S) 
        else:
            gradient_estimates = gradient_estimates.mean(dim = 1).view(-1, C, S, S) / (num_classes * N) 

        clone_model.train()
        loss_G = loss_values[:, -1].mean()
        return gradient_estimates.detach(), loss_G


def compute_gradient(args, victim_model, clone_model, x, pre_x=False, device="cpu"):
    if pre_x and args.G_activation is None:
        raise ValueError(args.G_activation)

    clone_model.eval()
    N = x.size(0)
    x_copy = x.clone().detach().requires_grad_(True)
    x_ = x_copy.to(device)
    if pre_x:
        x_ = args.G_activation(x_)

    t_logit,j = victim_model(x_,return_all_outputs = True)
    batch_size = len(t_logit[0])
    sample_indices = torch.arange(batch_size)
    stacked_output = torch.stack(t_logit, dim=0)
    selected_outputs = stacked_output[j, sample_indices]
    pred_victim=selected_outputs
    pred_clone = clone_model(x_)
    if args.loss == "l1":
        loss_fn = F.l1_loss
        if args.no_logits:
            pred_victim_no_logits = F.log_softmax(pred_victim, dim=1)
            if args.logit_correction == 'min':
                pred_victim = pred_victim_no_logits - pred_victim_no_logits.min(dim=1).values.view(-1, 1)
            elif args.logit_correction == 'mean':
                pred_victim = pred_victim_no_logits - pred_victim_no_logits.mean(dim=1).view(-1, 1)
            else:
                pred_victim = pred_victim_no_logits

    elif args.loss == "kl":
        loss_fn = F.kl_div
        pred_clone = F.log_softmax(pred_clone, dim=1)
        pred_victim = F.softmax(pred_victim, dim=1)

    else:
        raise ValueError(args.loss)

    loss_values = -loss_fn(pred_clone, pred_victim, reduction='mean')
    loss_values.backward()
    clone_model.train()
    return x_copy.grad, loss_values


class Args(dict):
    def __init__(self, **args):
        for k,v in args.items():
            self[k] = v
