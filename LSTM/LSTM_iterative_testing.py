import numpy as np

import activations
import torch
import torch.nn as nn

from tqdm import tqdm

from LSTM_network import *
from scaler import *
from data import *
from plotting import *



from data import *

is_covid=True
data = getData(is_covid=is_covid)


""" Reload scaler """
results_path = "./Results"
scaler_path = results_path + "/scaler.pickle"
scaler_type = "minMaxZeroOne"
scaler = scalerClass(scaler_type)
scaler = scaler.load_from_file(scaler_path)


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
samples_input, samples_target = createBatches(data, timesteps_input, timesteps_output)

params = {
"dim_in": 6,
"input_size": 1,
"hidden_size": 32,
"num_layers":1,
"act_out":"tanhplus",
"dim_out":timesteps_output,
}


network = LSTM(params)

saving_model_path = "./Results/Trained_Models/"
saving_model_path += "model_rnn"
saving_fig_path = "./Results/Figures/"

""" Load trained model """

network.layer_module_list.load_state_dict(torch.load(saving_model_path))


plotSamples(network, samples_input, samples_target, saving_fig_path, scaler=scaler)



plotSamplesIterative(network, data, saving_fig_path, scaler=scaler)


















