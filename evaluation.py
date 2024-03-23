"""
The evaluation methods we use here are mostly based on the paper about "Validation Diagnostics for SBI algorithms based on Normalizing Flows" [https://arxiv.org/abs/2211.09602]
We use the the following analysis:
- Local Consisteny Analysis with 
    - Pair Plots: Visualize the network posterior along the ground truth 
    - PP-plots:
    - Local Test statistics
- Global Consistency Analysus with global Pit
"""


import argparse 
import torch
from utils import BoidImagesDataset,  valid_transform
from autoencoders import LightningVAE, LightningSummaryFC, LightningSummaryConv
from flows import LightningRealNVP
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from scipy.stats import norm
from simulation import BoidSimulation
import seaborn as sns

parameter_names = [
        'N_boids',
        'N_obstacles',
        'L_separation',
        'L_coherence',
        'L_alignment',
        'L_avoidance',
        'visual_range',
        'avoid_range'
    ]

device = 'cpu'
torch.set_float32_matmul_precision('medium')

space_length = 64

parameter_ranges = [
    [50, 500], # n_boids
    [0,5], # n_obstacles
    [1,4], # l_seperation
    [0,4], # l_coherence
    [0,8], # l_alignment
    [0,25], # l_avoidance
    [0,64], # visual_range
    [0,8] # avoid_range
]

parameter_variables = [
    [60, 200, 400], # n_boids
    [1, 2, 4], # n_obstacles
    [1.5, 2, 3.5], # l_seperation
    [0.5, 2, 3.5], # l_coherence
    [1, 4, 7], # l_alignment
    [2, 10, 20], # l_avoidance
    [5, 10, 20], # visual_range
    [2, 5, 7] # avoid_range
]

def create_ground_truth_one_not_fixed(idx):
    """
    Create three ground truth instances were all parameters are fixed exept the one with index idx, were we use the values from variables
    """
    # parameters which are fixed (expect the one which we change later): N_boids,N_obstacles,L_separation,L_coherence,L_aligment,L_avoidance,visual_range,avoid_range. 
    p = [100, 3, 3, 2, 7, 20, 10, 5]

    images = torch.zeros((3,3,64,64))
    parameters = torch.zeros((3,8))

    boid_simulation = BoidSimulation(space_length=64)
    boid_simulation.placeBoids(p[0], 5)  
    boid_simulation.placeObstacles(p[1], 5, 10)

    for i in range(3):
        p[idx] = parameter_variables[idx][i]  # change the parameter which is not fixed

        parameters[i] = torch.tensor(p,dtype=float)

        print("simulation...")

        boid_simulation.simulate(separation=p[2],
                                    coherence=p[3],
                                    alignment= p[4],
                                    avoidance= p[5],
                                    visual_range=p[6],
                                    avoid_range= p[7],
                                    animate=False,
                                    num_time_steps=500,
                                    dt=0.1)
        
        images[i] = valid_transform(boid_simulation.finalStateImage())

    return parameters, images
    """
    Create three ground truth instances were all parameters are fixed exept the one with index idx, were we use the values from variables
    """
    # params: N_boids,N_obstacles,L_separation,L_coherence,L_aligment,L_avoidance,visual_range,avoid_range
    parameter_fixed = [100, 3, 3, 2, 7, 20, 10, 5]


    boid_simulation = BoidSimulation(space_length=space_length)
    boid_simulation.placeBoids(parameter_fixed[0], 5)
    boid_simulation.placeObstacles(parameter_fixed[1], 5, 10)

    images = torch.zeros((3,3,64,64))
    parameters = torch.zeros((3,8))

    for i in range(3):
        parameter_fixed[idx] = parameter_variables[idx][i] 
        parameters[i] = torch.tensor(parameter_fixed,dtype=float)

        print("simulation...")
        boid_simulation.simulate(separation=parameter_fixed[2],
                                    coherence=parameter_fixed[3],
                                    alignment= parameter_fixed[4],
                                    avoidance= parameter_fixed[5],
                                    visual_range = parameter_fixed[6],
                                    avoid_range= parameter_fixed[7],
                                    animate=False,
                                    num_time_steps=500,
                                    dt=0.1)

        images[i] = valid_transform(boid_simulation.finalStateImage())

    return parameters, images
    

def create_local_pair_plot(num_parameters: int, dataset:BoidImagesDataset, model_encoder, model_flow, path_to_evaluation_folder):
    """
    Local consistency analysis by creating a pair plot and saving it to models/<model>/evaluation folder.
    """
    colors = ["cornflowerblue", "royalblue", "midnightblue"] # colors for the legend
    num_ground_truth_instances = 3 # number of grund truth instances per parameter

    model_encoder.to(device)
    model_flow.to(device)

    param_indices = None
    if num_parameters == 4:
        param_indices = [2,3,4,5]
    else:
        param_indices = [0,1,2,3,4,5,6,7]
        
    for iter, param_idx in enumerate(param_indices):
        print(f"Create {path_to_evaluation_folder}/local_pair_plot_{param_idx}_num_Param_{num_parameters}.png")
        params, img = create_ground_truth_one_not_fixed(param_idx)
        sampledParams = np.zeros((num_ground_truth_instances,10000,num_parameters))

        for k in range(num_ground_truth_instances):
            image = img[k]
            condition = model_encoder.latent(image.reshape(1,3,64,64)) # transform to latent space
            condition = condition.reshape(-1).to(device) 
            num_samples = 10000
            noise = torch.randn(num_samples, num_parameters).to(device)
            sampledParam= model_flow.inverse(noise, condition).detach().cpu()

            sampledParams[k] = sampledParam
            
        fig, ax = plt.subplots(num_parameters,num_parameters,  figsize=(25,30))
        for i, p_i in enumerate(param_indices):
            for j, p_j in enumerate(param_indices):
                parameter_distance_i = (parameter_ranges[p_i][1] - parameter_ranges[p_i][0]) / 4.0
                parameter_distance_j = (parameter_ranges[p_j][1] - parameter_ranges[p_j][0]) / 4.0

                ax[i][j].set_xlim([parameter_ranges[p_j][0] -parameter_distance_j, parameter_ranges[p_j][1] + parameter_distance_j])

                # axis configuration similar to the paper
                if i==j:
                    if i==0:
                        ax[i][j].set_ylabel(parameter_names[p_i])

                    if i==num_parameters-1:
                        ax[i][j].set_xlabel(parameter_names[p_j])
                    else:
                        ax[i][j].set_xticks([])

                    ax[i][j].set_yticks([])
                
                else:
                    ax[i][j].set_ylim([parameter_ranges[p_i][0] -parameter_distance_i, parameter_ranges[p_i][1] + parameter_distance_i])

                    if i == num_parameters-1 and j==0:
                        ax[i][j].set_xlabel(parameter_names[p_j])
                        ax[i][j].set_ylabel(parameter_names[p_i])
                    elif j==0:
                        ax[i][j].set_xticks([])
                        ax[i][j].set_ylim([parameter_ranges[p_i][0] -parameter_distance_i, parameter_ranges[p_i][1] + parameter_distance_i])
                        ax[i][j].set_ylabel(parameter_names[p_i])
                    elif i ==  num_parameters-1:
                        ax[i][j].set_yticks([])
                        ax[i][j].set_xlabel(parameter_names[p_j])
                    else:
                        ax[i][j].set_xticks([])
                        ax[i][j].set_yticks([])
                
                for k in range(num_ground_truth_instances):
                    if i ==j:
                        ax[i][j].hist(sampledParams[k][:,j], weights = np.ones(len(sampledParams[k]))/len(sampledParams[k]) , alpha = 0.5, color = colors[k], label = f"{parameter_names[param_idx]}= {params[k][param_idx]}", bins=100)
                        ax[i][j].vlines(params[k][p_j], 0, 0.2, linestyles= "dashed", color = "red", linewidth = 2.0)

                        handles, labels = ax[i][j].get_legend_handles_labels()
                    elif i>j:
                        counts, xbins, ybins = np.histogram2d(sampledParams[k][:,j], sampledParams[k][:,i], bins=20)
                        extent = [xbins.min(), xbins.max(), ybins.min(), ybins.max()]

                        ax[i][j].contour(counts.T, extent=extent, levels= [ counts.max() * 0.7, counts.max()*0.9], colors=colors[k])
                        ax[i][j].scatter(params[k][p_j],params[k][p_i] ,color ="red", s= 30.0)
                    else:
                        ax[i][j].set_axis_off()

        fig.legend(handles,labels, loc='center right', fontsize = 20.0,  title_fontsize=15, bbox_to_anchor= (0.9,0.6)).set_title("Estimated posterior for")
    
        # for title
        fixed_params = params[0][:param_idx].detach().numpy()
        fixed_params = np.append(fixed_params,  params[0][param_idx+1 :])

        fixed_params_names = parameter_names[:param_idx]
        fixed_params_names = np.append(fixed_params_names, parameter_names[param_idx+1 :])

        fig.suptitle(f"Local pair-plots \n  Fixed: {fixed_params_names} = \n {fixed_params}", fontsize= 20)
        plt.savefig(f"{path_to_evaluation_folder}/local_pair_plot_{param_idx}_num_Param_{num_parameters}.png")

def compute_pi_and_ri_alpha(num_parameters: int, dataset, model_encoder, model_flow):
    """
    Compute and return PIT and ri_alpha values
    """
    len_evaluation_dataset = dataset.__len__()
    P_i_values = np.zeros((len_evaluation_dataset, num_parameters))

    param_indices = None
    if num_parameters == 4:
        param_indices = [2,3,4,5]
    else:
        param_indices = [0,1,2,3,4,5,6,7]

    print("compute pit values")
    for i, data in enumerate(tqdm(evaluation_dataset)):
        if i == len_evaluation_dataset-1: # handle this case
            break
        params, img = data
        params = params[param_indices]

        # encode
        params = torch.from_numpy(params).reshape(1, -1).to(device)

        img = img.to(device)
        encoded = model_encoder.latent(img.reshape(1,3,64,64))

        # pass in model flow -> get norm distr.
        code = model_flow.forward(params, encoded)[0]

        # apply CDF -> get probs
        P_i_values[i] = norm.cdf(code.detach().cpu())

    grid_resolution = 200
    # create grid of alpha values
    alphas = np.linspace(0, 1, grid_resolution, endpoint=True)

    print("compute ri alphas values")
    # calc r_i_alpha values (mean of P_i <= alpha)
    r_i_alphas = np.zeros((num_parameters, grid_resolution))

    for i, column in enumerate(tqdm(P_i_values.T)):
        r_i_alphas[i] = np.mean(column[:, None] <= alphas, axis=0)

    return P_i_values, r_i_alphas, alphas
    
def create_global_pit(num_parameters, P_i_values, r_i_alphas, alphas, path_to_evaluation_folder):
    """
    Global consistency analysis by creating a global PIT plot
    """
    param_indices = None
    if num_parameters == 4:
        param_indices = [2,3,4,5]
    else:
        param_indices = [0,1,2,3,4,5,6,7]

    # plot r_i_alphas
    fig, ax = plt.subplots(1, num_parameters, figsize=(25, 5))
    for i, values in enumerate(r_i_alphas):
        ax[i].plot(alphas, values)
        ax[i].plot(alphas, alphas)
        ax[i].set_title(parameter_names[param_indices[i]])
        ax[i].set_aspect(1)
        ax[i].fill_between(alphas, (alphas-0.05), (alphas+0.05), color='b', alpha=.1)
        ax[i].set_xlabel(r"$\alpha")
        ax[i].set_ylabel(r"empirical $r_{i,\alpha}")

        inset_axes = ax[i].inset_axes(bounds=[0.5,0,0.5,0.2])
        inset_axes.hist(r_i_alphas[i], bins=10)
        inset_axes.axis(False)
        inset_axes.hlines(len(values) / 10, 0, 1, colors='black')
        inset_axes.fill_between(alphas, (len(values) / 10 - 0.05*len(alphas)), (len(values) / 10 + 0.05*len(alphas)), color='b', alpha=.3)

    plt.savefig(f"{path_to_evaluation_folder}/global_pit_num_Param_{num_parameters}.png")

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for i, values in enumerate(r_i_alphas):
        ax.plot(alphas, values, label=parameter_names[param_indices[i]])
        ax.set_aspect(1)


    ax.plot(alphas, alphas, color='black', linestyle='--')
    ax.set_xlabel(r"$\alpha")
    ax.set_ylabel(r"empirical $r_{i,\alpha}")
    ax.fill_between(alphas, (alphas-0.05), (alphas+0.05), color='g', alpha=.2)
    ax.legend()

    plt.title('Global PIT of boid parameters')

    plt.savefig(f"{path_to_evaluation_folder}/global_pit_all_in_one_num_Param_{num_parameters}.png")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Train a specified summary network to use as a summary network'
    )

    parser.add_argument('--model_folder', required=True, help='Name of model folder in which encoder, flow and evaluation are stored')
    parser.add_argument('--dataset', required=True, help="Path to data set folder on which the model is evaluated")
    parser.add_argument('--num_parameters', required=True, help="Number of parameters to analyse. Can be 4 or 8")

    args = parser.parse_args()

    # load evaluation dataset

    if not os.path.exists(args.dataset):
        raise Exception(f"dataset folder {args.dataset} not found")

    evaluation_dataset = BoidImagesDataset(args.dataset, transform=valid_transform)
    
    # load encoder 
    # encoder checkpoint and hyperparameter are located at models/<model_name>/encoder 

    path_to_encoder_folder = f"models/{args.model_folder}/encoder"

    encoder_checkpoint_files = [filename for filename in os.listdir(path_to_encoder_folder) if filename.endswith("ckpt")] # search for checkpoint file
    encoder_hyperparameter_files = [filename for filename in os.listdir(path_to_encoder_folder) if filename.endswith("yaml")] # search for hyperparameter file
    encoder_checkpoint = None
    encoder_hyperparameter = None

    if encoder_checkpoint_files.__len__() != 1 or encoder_hyperparameter_files.__len__() !=1: # only one file for checkpoint and hyperparameter should be contained
        raise Exception(f"There is no checkpoint file in the encoder folder {path_to_encoder_folder} or there are multiple ones")
    
    encoder_checkpoint = f"{path_to_encoder_folder}/{encoder_checkpoint_files[0]}"
    encoder_hyperparameter = f"{path_to_encoder_folder}/{encoder_hyperparameter_files[0]}"

    model_encoder = None
    if args.model_folder.startswith("FC4") or args.model_folder.startswith("FC8"): # fully connected summary network
        model_encoder = LightningSummaryFC.load_from_checkpoint(encoder_checkpoint, hparams_file=encoder_hyperparameter)
    elif args.model_folder.startswith("CONV4") or args.model_folder.startswith("CONV8"): # convolution summary network
        model_encoder = LightningSummaryConv.load_from_checkpoint(encoder_checkpoint, hparams_file=encoder_hyperparameter)
    elif args.model_folder.startswith("AE"): # vanilla autoencoder
        model_encoder = LightningVAE.load_from_checkpoint(encoder_checkpoint, hparams_file=encoder_hyperparameter)
    else:
        raise Exception(f"No model folder with name {args.model_folder} exists")

    model_encoder.to(device)
    # load flow 
    # flow checkpoint and hyperparameter are located at models/<model_name>/flow ... 

    num_parameters = int(args.num_parameters) # get_number of parameters
    path_to_flow_folder = None
    if num_parameters == 4:
        path_to_flow_folder = f"models/{args.model_folder}/flow00111100"
    elif num_parameters == 8:
        path_to_flow_folder = f"models/{args.model_folder}/flow11111111"
    else:
        raise Exception("Number of parameters must be either 4 or 8.")
    
    flow_checkpoint_files = [filename for filename in os.listdir(path_to_flow_folder) if filename.endswith("ckpt")]  # search for checkpoint file
    flow_hyperparameter_files = [filename for filename in os.listdir(path_to_flow_folder) if filename.endswith("yaml")] # search for hyperparameter file
    flow_checkpoint = None
    flow_hyperparameter = None

    model_flow = None 

    if flow_checkpoint_files.__len__() != 1 or flow_hyperparameter_files.__len__() !=1:  # only one file for checkpoint and hyperparameter should be contained
        raise Exception(f"There is no checkpoint file in the flow folder {path_to_flow_folder} or there are multiple ones")

    flow_checkpoint = f"{path_to_flow_folder}/{flow_checkpoint_files[0]}"
    flow_hyperparameter = f"{path_to_flow_folder}/{flow_hyperparameter_files[0]}"
    
    model_flow = LightningRealNVP.load_from_checkpoint(flow_checkpoint, hparams_file=flow_hyperparameter, encoder = model_encoder)
    model_flow.to(device)

    # create evaluation folder if not alreay created 
    if not os.path.exists(f"models/{args.model_folder}/evaluation"):
        os.mkdir(f"models/{args.model_folder}/evaluation")

    path_to_evaluation_folder = f"models/{args.model_folder}/evaluation"

    # Evaluation method 1 - Local pair plot
        
    print("create local pair plot")
    create_local_pair_plot(num_parameters=num_parameters, dataset=evaluation_dataset, model_encoder=model_encoder, model_flow=model_flow, path_to_evaluation_folder= path_to_evaluation_folder)

    # Evaluation method 2 - PIT

    # Compute pi values and r_i_alphas
    print("create PIT plots")
    P_i_values, r_i_alphas, alphas = compute_pi_and_ri_alpha(num_parameters=num_parameters, dataset=evaluation_dataset, model_encoder=model_encoder, model_flow=model_flow)
    
    print("create global pit plot")
    create_global_pit(num_parameters,P_i_values=P_i_values, r_i_alphas=r_i_alphas, alphas=alphas, path_to_evaluation_folder=path_to_evaluation_folder)