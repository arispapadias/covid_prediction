import pickle
import numpy as np
import os

import activations
import torch
import torch.nn as nn

from tqdm import tqdm

from RNN_network import *
from scaler import *
from data import *
from plotting import *

# is_covid=False
is_covid=True
data = getData(is_covid=is_covid)

print(np.shape(data))


results_path = "./Results"
os.makedirs(results_path, exist_ok=True)


# data = deaths_per_day_smooth
# data = np.reshape(data, (-1, 1))
# scaler_type = "standard"
# scaler = scalerClass(scaler_type, data)
# print("Before scaling:")
# print(np.mean(data))
# print(np.std(data))
# data = scaler.scale(data)
# print("After scaling:")
# print(np.mean(data))
# print(np.std(data))
# print(np.min(data))
# print(np.max(data))

scaler_type = "minMaxZeroOne"
scaler = scalerClass(scaler_type)
scaler.fit(data)

""" Save the scaler """
scaler_path = results_path + "/scaler.pickle"
scaler.save_to_file(scaler_path)

print("Before scaling:")
print(np.max(data))
print(np.min(data))
data = scaler.scale(data)
print("After scaling:")
print(np.max(data))
print(np.min(data))



# 2. Definition of input and output
# Many to one mapping

timesteps_input = 6
timesteps_output = 1
# Form the data batches
samples_input, samples_target = createBatches(data, timesteps_input, timesteps_output)
num_samples = len(samples_input)

# 3. selection of the RNN architecture
# ---- DETERMINATION OF OUTPUT ACTIVATION act_out ----
# a) the range (y values of f(x)=y) of the activation function at the output NEEDS TO MATCH the output DATA range.
# E.G. If the output data are scaled to [0,1] then the activation function should map (real numbers) R->[0,1].
# If the output data are scaled to N(0,1) (normal zero mean unit variance), the activation should map to the real numbers !

# ---- DETERMINATION OF INTERNAL ACTIVATION act (internal) ----
# better to be continuous, derivative defined anywhere, BOUNDED in (-1,1), (0,1), with "good" properties.

params = {
"dim_in": 6,
"input_size": 1,
"hidden_size":12,
"num_layers":1,
"act_out":"tanhplus",
"dim_out":timesteps_output,
}

network = RNN(params)


K = 32 # BATCH SIZE
# input_data = samples_input[0:K, :]
# input_data = torch.tensor(input_data).float()
# output = network.forward(input_data)
# print(output.size())
# print(output)

# 4. shuffle the data

# idx_shuffle = np.random.permutation(len(samples_input))
# samples_input   = samples_input[idx_shuffle]
# samples_target  = samples_target[idx_shuffle]

# 5. divide between training and validation
# N_train = int(num_samples*0.8)
N_train = int(num_samples*0.8)
print("Training samples {:}/{:}.".format(N_train, num_samples))
samples_input_train   = samples_input[:N_train]
samples_target_train  = samples_target[:N_train]
samples_input_val   = samples_input[N_train:]
samples_target_val  = samples_target[N_train:]
print("Training data shapes:")
print(np.shape(samples_input_train))
print(np.shape(samples_target_train))
print(np.shape(samples_input_val))
print(np.shape(samples_target_val))


batch_size=32

from torch.utils.data import TensorDataset, DataLoader

samples_input_train = torch.Tensor(samples_input_train)
samples_target_train = torch.Tensor(samples_target_train)

samples_input_val = torch.Tensor(samples_input_val)
samples_target_val = torch.Tensor(samples_target_val)

dataset_train = TensorDataset(samples_input_train, samples_target_train)
dataset_val = TensorDataset(samples_input_val, samples_target_val)

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

print("Dataset sizes:")
print(len(dataloader_train))
print(len(dataloader_val))

""" Let's try to plot some samples """
samples_input_val = torch.Tensor(samples_input_val)
samples_target_val = torch.Tensor(samples_target_val)





# print(network.layer_module_list.parameters())
# print(network.layer_module_list.named_parameters())

params_trainable = network.layer_module_list.parameters()
learning_rate=0.001
weight_decay = 0.0
optimizer = torch.optim.Adam(
                params_trainable,
                lr=learning_rate,
                weight_decay=weight_decay,
            )


""" By definition, an epoch is when the whole dataset (train or validation) has been "passed" through the network one time, in other words, the networks has "seen" the whole dataset one time.
Because we are learning very slowly the weights (the optimizer is slowly chaning the weights, with a small learning rate), we need to pass the whole dataset (in batches) more than once, which means to train for many EPOCHS """

def forwardSingleEpoch(dataloader, optimizer=None):
    """ Tracking the mean loss """
    loss_ = 0.0
    batch_iter = 0
    for sample in dataloader:

        if optimizer is not None:
            optimizer.zero_grad()

        sample_input, sample_target = sample
        sample_output, loss = network.forwardAndComputeLoss(sample_input, sample_target)

        if optimizer is not None:
            """ With .backward the gradient is computed """
            loss.backward()
            """ With step the weights are updated """
            optimizer.step()

        """ Here I need the loss just as VALUE not as a tensor, so I DETACH it from the graph ! """
        loss_ += loss.detach().cpu().numpy()
        batch_iter += 1
    loss_ = loss_ / batch_iter
    return loss_

loss_train = forwardSingleEpoch(dataloader_train)
loss_val = forwardSingleEpoch(dataloader_val)

print("Initial training loss = {:1.2E}".format(loss_train))
print("Initial validation loss = {:1.2E}".format(loss_val))

import os
saving_model_path = results_path + "/Trained_Models/"
os.makedirs(saving_model_path, exist_ok=True)
saving_model_path += "model_rnn"

saving_fig_path = results_path + "/Figures/"
os.makedirs(saving_fig_path, exist_ok=True)

print("Saving initial model...")
torch.save(network.layer_module_list.state_dict(), saving_model_path)

num_epochs = 1000
# Dummy
epoch_num=0
""" Dummy initial valdiation error """
val_error_min = loss_val 
""" Track the training and validation losses """
losses_train = []
losses_val = []

progress_bar = tqdm(total=num_epochs)
for epoch_iter in range(num_epochs):

    loss_train = forwardSingleEpoch(dataloader_train, optimizer=optimizer)
    loss_val = forwardSingleEpoch(dataloader_val)

    print("Epoch {:}, Train loss = {:1.2E}, Val loss = {:1.2E}".format(epoch_iter, loss_train, loss_val))

    # if epoch_iter % 100 == 0:
    #     plotSamples(network, samples_input_val, samples_target_val)

    if loss_val < val_error_min:
        print("Saving model...")
        torch.save(network.layer_module_list.state_dict(), saving_model_path)
        val_error_min = loss_val
        epoch_num = epoch_iter

    losses_train.append(loss_train)
    losses_val.append(loss_val)

    progress_bar.update(1)
    
progress_bar.close()

plotSamples(network, samples_input_val, samples_target_val, saving_fig_path)





fig_path = saving_fig_path + "losses.png"
epochs_ = np.arange(len(losses_train))
plt.plot(epochs_, np.log(losses_train), "g-", label="Train")
plt.plot(epochs_, np.log(losses_val), "b-", label="Val")
plt.plot(epoch_num, np.log(val_error_min), "rx", label="Best")
plt.xlabel("Epoch number")
plt.legend(loc="upper left",
                   bbox_to_anchor=(1.05, 1),
                   borderaxespad=0.,
                   frameon=False)
plt.ylabel("$Log_{10}(Loss)$")
# plt.show()
plt.tight_layout()
plt.savefig(fig_path, dpi=300)
plt.close()






















