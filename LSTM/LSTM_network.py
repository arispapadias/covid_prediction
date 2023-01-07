import activations
import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, params):
        super(LSTM, self).__init__()
        self.input_size = params["input_size"]
        self.hidden_size  = params["hidden_size"]
        self.num_layers = params["num_layers"]
        self.act_out    = params["act_out"]
        self.dim_out    = params["dim_out"]
        self.dim_in    = params["dim_in"] 
        self.act_out_ = activations.getActivation(self.act_out)
        # print(self.layer_size)
        print("[LSTM] Building...")
        self.build()

    def build(self):
        self.rnn = torch.nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.mlp = torch.nn.Linear(self.hidden_size, self.dim_out, bias=True)
        self.layer_module_list = nn.ModuleList([
            self.rnn,
            self.mlp,
        ])
        self.initializeWeights()

    def initializeWeights(self):
        # print("[LSTM] Initializing parameters...")
        for layer_module in self.layer_module_list:
            # print(layer_module)
            for name, param in layer_module.named_parameters():
                # print(name)
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param.data)   
                elif 'bias' in name:
                    # print(len(list(param.data.size())))
                    param.data.fill_(0.001)
                else:
                    raise ValueError("Do not know how to initialize parameter {:}.".format(name))
        return 0

    def forward(self, x):
        x = torch.unsqueeze(x, dim=-1)
        o, h = self.rnn(x)
        # get last output
        o = o[:, -1]
        y = self.mlp(o)
        y = self.act_out_(y)
        return y

    def forwardAndComputeLoss(self, x, target_):
        output_ = self.forward(x)
        loss = ((target_-output_)**2).mean()
        return output_, loss
