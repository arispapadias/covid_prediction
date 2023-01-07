
import torch
import numpy as np


import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
# FONTSIZE = 18
# font = {'size': FONTSIZE, 'family': 'Times New Roman'}
# matplotlib.rc('xtick', labelsize=FONTSIZE)
# matplotlib.rc('ytick', labelsize=FONTSIZE)
# matplotlib.rc('font', **font)
# matplotlib.rcParams['mathtext.fontset'] = 'custom'
# matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
# matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'
# matplotlib.rcParams['mathtext.bf'] = 'Times New Roman:bold'
# # Plotting parameters
# rc('text', usetex=True)
# plt.rcParams["text.usetex"] = True
# plt.rcParams['xtick.major.pad'] = '10'
# plt.rcParams['ytick.major.pad'] = '10'




def plotSamples(network, samples_input, samples_target, saving_fig_path, scaler=None):
    samples_input = torch.Tensor(samples_input).float()
    samples_target = torch.Tensor(samples_target).float()


    samples_output = network.forward(samples_input)
    samples_output = samples_output.detach().cpu().numpy()
    samples_target = samples_target.detach().cpu().numpy()


    # Transform back to unscaled values if scaler is provided
    if scaler is not None:
        print("Scaler provided...")
        samples_output = scaler.descale(samples_output)
        samples_target = scaler.descale(samples_target)

    po = samples_output[:,0]
    pt = samples_target[:,0]

    error = np.sqrt(np.mean((po-pt)**2))


    plt.plot(pt, po, "bx")
    plt.plot(pt, pt, "r-")
    plt.xlabel("Target values")
    plt.ylabel("Output values")
    plt.title(f"RMSE error={error:.3f}")
    # plt.show()
    fig_path = saving_fig_path + "samples.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    time = np.arange(len(po))
    plt.plot(time, po, "bx", label="Target values")
    plt.plot(time, pt, "r-", label="Output values")
    plt.xlabel("Day")
    plt.ylabel("Value")
    plt.title(f"RMSE error={error:.3f}")
    plt.legend(loc="upper left",
                bbox_to_anchor=(1.05, 1),
                borderaxespad=0.,
                frameon=False)
    # plt.show()
    fig_path = saving_fig_path + "samples_in_time.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    return 0



def plotSamplesIterative(network, data, saving_fig_path, scaler=None):
    data = torch.Tensor(data).float()
    # Iterative forecasting with the network
    print(data.size())

    prediction_horizon = 100
    """ Timesteps need to be ordered ! """
    # Sample a random initial condition from the data, for example day 17
    # We assume that we know all data until day 17, day 17 is the PRESENT
    # we want to predict the future. 17-17+prediction_horizon
    idx_ = 10
    input_  = data[idx_ : idx_+network.dim_in]
    input_ = torch.reshape(input_, (1, network.dim_in))
    # print(input_.size())

    target_ic = []
    prediction_ic = []
    print("Iterative...")
    for current_timesteps in range(prediction_horizon):
        # input_ is of dimension [K, T] where K=samples processed in parallel, T is the timestep.
        # In this case: K=1, input_ = [1, T]
        prediction_ = network.forward(input_)
        prediction_ic.append(prediction_[0].detach().cpu().numpy())

        target_ = data[idx_ + network.dim_in + network.dim_out + current_timesteps]
        target_ = torch.reshape(target_, (1, network.dim_out))
        target_ic.append(target_[0].detach().cpu().numpy())

        # input_        is [K, dim_in] = [K, 5]
        # prediction_   is [K, dim_out] = [K, 1]
        input_ = torch.cat((input_, prediction_), axis=1)
        # input_        is [K, dim_in + dim_out] = [K, 6]
        # print(input_.size())
        # input_ = input_[:, 1:] # Discard the first time-step
        input_ = input_[:, network.dim_out:] # Discard the first time-step
        # input_        is [K, dim_in] = [K, 5]


    target_ic = np.array(target_ic)
    prediction_ic = np.array(prediction_ic)

    # Transform back to unscaled values if scaler is provided
    if scaler is not None:
        print("Scaler provided...")
        target_ic = scaler.descale(target_ic)
        prediction_ic = scaler.descale(prediction_ic)

    error = np.sqrt(np.mean((prediction_ic-target_ic)**2))



    time = np.arange(prediction_horizon)
    plt.plot(time, target_ic, "bx", label="Target")
    plt.plot(time, prediction_ic, "r-", label="Prediction")
    plt.title(f"RMSE error={error:.3f}")
    plt.xlabel("Time (day)")
    plt.ylabel("Value (deaths per day)")
    plt.legend(loc="upper left",
                       bbox_to_anchor=(1.05, 1),
                       borderaxespad=0.,
                       frameon=False)
    fig_path = saving_fig_path + "samples_iterative.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Deaths at the last day {prediction_ic[-1]}")
    return 0


