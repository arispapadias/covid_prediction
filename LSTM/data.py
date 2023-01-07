
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

import json
import numpy as np

def getData(is_covid=False):
    if is_covid:
        data_file = "../data/data.json"
        f = open(data_file,)
        data_raw = json.load(f)
        data_raw = data_raw['Greece']
        f.close()

        # {'date': '2021-08-29', 'confirmed': 581315, 'recovered': 0, 'deaths': 13581}
        temp = []
        for dict_ in data_raw:
            temp.append([dict_["date"], dict_["confirmed"], dict_["recovered"], dict_["deaths"]])

        temp = np.array(temp)
        # print(np.shape(temp))
        idx_sort = np.argsort(temp[:,0])
        temp = temp[idx_sort]
        data_all = temp[:, 1:]
        data_all = np.array(data_all, dtype=int)

        """ Only the deaths """
        deaths = data_all[:, 2]
        # plt.plot(deaths)
        # plt.show()


        """ Computing daily deaths """
        deaths_per_day = deaths[1:] - deaths[:-1]
        # plt.plot(deaths_per_day)
        # plt.show()

        """ Signal is very noisy, perform smoothing """
        from scipy.signal import savgol_filter
        window_size = 21
        # window size, polynomial order 3
        deaths_per_day_smooth = savgol_filter(deaths_per_day, window_size, 3) 

        days = np.arange(len(deaths_per_day_smooth))

        data = deaths_per_day_smooth
        # data = np.reshape(data, (-1, 1))
        data = np.reshape(data, (-1))

        data = data[300:]
    else:   

        start_time = 0
        end_time = 1
        sample_rate = 1000
        time = np.arange(start_time, end_time, 1/sample_rate)
        theta = 0
        frequency = 40
        amplitude = 1
        sinewave = amplitude * np.sin(2 * np.pi * frequency * time + theta)
        # print(np.shape(sinewave))

        # plt.plot(sinewave)
        # plt.show()
        # 
        data = sinewave
    return data



def createBatches(data, timesteps_input, timesteps_output):
    num_timesteps = len(data)
    print("Number of timesteps = {:}".format(num_timesteps))

    max_samples = num_timesteps - timesteps_input # - timesteps_output
    print("Maximum number of samples = {:}".format(max_samples))

    # 1, 2, 3, 4, 5 -> 6
    # 2, 3, 4, 5, 6 -> 7
    # 3, 4, 5, 6, 7 -> 8
    samples_input = []
    samples_target = []
    for sample_in in range(max_samples):
        sample_input = data[sample_in:sample_in+timesteps_input]
        sample_target = data[sample_in+timesteps_input:sample_in+timesteps_input+timesteps_output]
        # print(sample_in)
        # print(np.shape(sample_input))
        # print(np.shape(sample_target))
        samples_input.append(sample_input)
        samples_target.append(sample_target)

    samples_input = np.array(samples_input)
    samples_target = np.array(samples_target)
    print("Samples:")
    num_samples = len(samples_input)
    print(np.shape(samples_input))
    print(np.shape(samples_target))



    return samples_input, samples_target






