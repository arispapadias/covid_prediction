import numpy as np
import pickle

# 1. step: normalize data, either in [0,1] or standard scaling
# TODO: read about normalization, read about stadard & minMax scaling, etc.
class scalerClass(object):
    def __init__(self, scaler_type):
        self.scaler_type = scaler_type

    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)

    def scale(self, data):
        # Input data shape [K, D]
        # assert len(np.shape(data))==2
        if self.scaler_type == "standard":
            # Adding the dimension for the batch
            # mean = self.mean[np.newaxis]
            # std  = self.std[np.newaxis]
            mean = self.mean
            std = self.std
            data = (data-mean) / std
        elif self.scaler_type == "minMaxZeroOne":
            # Adding the dimension for the batch
            # min_ = self.min[np.newaxis]
            # max_  = self.max[np.newaxis]
            min_ = self.min
            max_ = self.max
            data = (data - min_) / (max_ - min_)
        return data

    def descale(self, data):
        print(np.shape(data))
        N, D = np.shape(data)
        if self.scaler_type == "minMaxZeroOne":
            min_ = self.min
            max_ = self.max
            data = data * (max_ - min_) + min_ 
        return data

    def save_to_file(self, filename):
        data_scaler = {"scaler": self} # dictrionary of values
        print(f"Saving scaler in file {filename}")

        with open(filename, 'wb') as file:  # wb means write binary
            pickle.dump(data_scaler, file, protocol=pickle.HIGHEST_PROTOCOL)

        return 0

    def load_from_file(self, filename):
        with open(filename, 'rb') as file:  # rb means read binary
            data = pickle.load(file)
        return data["scaler"]
