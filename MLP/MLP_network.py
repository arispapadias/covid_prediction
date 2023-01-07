import activations
import torch
import torch.nn as nn


# eisodos x
# Ax + b (linear layer, GRAMMIKO layer) -> linear(x)
# activation(Ax+b) opou activation einai RELU/CELU/TANH
# ena layer !
# y = layer(x) != y=activation(Ax + b) != y=activation(linear(x))
# ONE LAYER
# MORE LAYERS:
# y = layer( layer( layer ( ... layer(x) ) ) )
# y = act5(A5 act4(A4  act3(A3 act2(A2 act1(A1 x + b1) + b2) + b3) + b4) + b5)
# act5 is different from the others because y needs to match the SCALED targets y_tilde [-1, 1]/[0,1]
class MLP(nn.Module):
    def __init__(self, params):
        super(MLP, self).__init__()
        self.layer_size = params["layer_size"]
        self.layer_num  = params["layer_num"]
        self.act        = params["act"]
        self.act_out    = params["act_out"]
        self.dim_in     = params["dim_in"]
        self.dim_out    = params["dim_out"]

        self.act_ = activations.getActivation(self.act)
        self.act_out_ = activations.getActivation(self.act_out)

        self.layer_size = [params["dim_in"]] + self.layer_num * [self.layer_size] + [params["dim_out"]]
        # print(self.layer_size)
        print("[MLP] Building...")
        self.build()

    def build(self):
        self.layer_module_list = []
        for i in range(len(self.layer_size)-1):
            in_features     = self.layer_size[i]
            out_features    = self.layer_size[i+1]
            self.layer_module_list.append(torch.nn.Linear(in_features, out_features, bias=True))
            # Check if output activation
            act_ = self.act_ if i < len(self.layer_size)-2 else self.act_out_
            self.layer_module_list.append(act_)

        self.layer_module_list = nn.ModuleList(self.layer_module_list)
        print(self.layer_module_list)
        self.initializeWeights()

    def initializeWeights(self):
        # print("[MLP] Initializing parameters...")
        for layer_module in self.layer_module_list:
            # print(layer_module)
            for name, param in layer_module.named_parameters():
                # print(name)
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param.data)   
                    # ndim = len(list(param.data.size()))
                    # print(ndim)
                    # if ndim > 1:
                    #     torch.nn.init.xavier_uniform_(param.data)
                elif 'bias' in name:
                    # print(len(list(param.data.size())))
                    param.data.fill_(0.001)
                else:
                    raise ValueError("Do not know how to initialize parameter {:}.".format(name))
        return 0

    def forward(self, input):
        # print(input.size())
        # input_last = input[:, -1:].clone()
        # print(input_last.size())
        # print(ark)
        # print("#"*20 + ' MLP: forward() ' + "#"*20)
        # print(input.size())
        for layer in self.layer_module_list:
            # print("-------")
            # print(layer)
            input = layer(input)
            # print(input.size())
            # print(input.min())
            # print(input.max())
        # print(input.size())
        # print(ark)
        # input = 0.0000000001*input + input_last
        return input

    def forwardAndComputeLoss(self, input_, target_):
        output_ = self.forward(input_)
        loss = ((target_-output_)**2).mean()
        return output_, loss
