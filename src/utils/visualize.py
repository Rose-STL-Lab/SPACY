import os
from datetime import datetime
import torch
import matplotlib.pyplot as plt

def get_savedir(data_string, seed):
    """ Generate a directory name for saving logs and results.
    
    Args:
        data_string (str): A string representing the data configuration.
        seed (int): The random seed used for training.
    Returns:
        str: The generated directory name.
    """
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    string = f'seed_{seed}_{dt_string}'
    return os.path.join('logs', data_string, string)


def visualize_validation_step(save_dir, F_mask, F_gt, F_evolve, rho_logvar, model):
    """
    Visualizes the output of a validation step.
    """
    if F_mask is not None:
        for i in range(F_mask.shape[0]):
            for j in range(F_mask.shape[1]):
                plt.imsave(os.path.join(save_dir, f'F_pred_{i}_{j}.png'), F_mask[i][j].detach().cpu().numpy())
        plt.close()

    if F_gt is not None and F_gt != []:
        for i in range(F_gt.shape[0]):
            for j in range(F_gt.shape[1]):
                plt.imsave(os.path.join(save_dir, f'F_gt_{i}_{j}.png'), F_gt[i][j].detach().cpu().numpy())
        plt.close()

    if F_mask is not None:
        combined_factor = torch.sum(F_mask.detach().cpu(), axis=1).reshape(model.num_variates, model.ny, model.nx)
        F_evolve.append(combined_factor)
        rho_logvar.append(model.spatial_factors.rho_logvar.detach().cpu())


def visualize_test_step(save_dir, model , F_mask, G_recon, F_evolve = None, rho_logvar = None):
    """
    Visualizes the output of a test step.
    """
    sp = model.spatial_factors
    center, scale = sp.get_centers_and_scale()

    torch.save(center.detach().cpu(), os.path.join(save_dir, f'center.pt'))
    torch.save(scale.detach().cpu(), os.path.join(save_dir, f'scale.pt'))
    
    if F_evolve:
        F_evolve_tensor = torch.stack(F_evolve, axis = 0)
        torch.save(F_evolve_tensor.detach().cpu(), os.path.join(save_dir, f'F_evolve.pt'))
    
    if rho_logvar:
        rho_logvar_tensor = torch.stack(rho_logvar, axis = 0)
        torch.save(rho_logvar_tensor.detach().cpu(), os.path.join(save_dir, f'rho_logvar.pt'))

    if F_mask is not None:
        torch.save(F_mask.detach().cpu(), os.path.join(save_dir, f'F_pred.pt'))
    
    if G_recon is not None:
        torch.save(G_recon.detach().cpu(), os.path.join(save_dir, f'G_pred.pt'))

    if model.num_variates > 1:
        alpha = model.alpha.sample_alpha()
        torch.save(alpha.detach().cpu(), os.path.join(save_dir, f'alpha.pt'))
