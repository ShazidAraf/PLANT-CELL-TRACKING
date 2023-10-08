import torch
import torch.nn as nn

def conv_block_2d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=4, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        activation,
        nn.MaxPool2d(2),
        nn.Conv2d(out_dim, out_dim, kernel_size=4, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        activation,
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32*11*11, 64),
        nn.Linear(64, 1),
        nn.Sigmoid()
        # nn.Softmax(dim=1)
        )

class deepseed(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(deepseed, self).__init__()
        
        self.in_dim = in_dim
        activation = nn.ReLU(inplace=True)
        self.cnn = conv_block_2d(in_dim,out_dim,activation)


    
    def forward(self, x):

        out = self.cnn(x)
        return out



if __name__ == "__main__":
    
    x = torch.rand((10,1,50,50))
    model = deepseed(in_dim=1, out_dim=32)
    y = model(x)
    print(y.shape)                                  
    parameters = count_parameters(model)
    print(parameters)