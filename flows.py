import torch
import lightning as L

from typing import Any


class AffineCouplingLayer(torch.nn.Module):
    def __init__(self, input_size, hidden_size, condition_size, rotate=True):
        super(AffineCouplingLayer, self).__init__()

        self.input_size = input_size

        self.upper = self.input_size // 2
        self.lower = self.input_size - self.upper

        self.hidden_size = hidden_size
        self.condition_size = condition_size

        self.translation = torch.nn.Sequential(
            torch.nn.Linear(self.upper +self.condition_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, self.lower),
        )

        self.scale = torch.nn.Sequential(
            torch.nn.Linear(self.upper+self.condition_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, self.lower),
            torch.nn.Tanh()
        )

        rotation = torch.eye(input_size)

        if rotate:
            rotation = torch.linalg.qr(torch.randn(input_size, input_size))[0]

        self.register_buffer('rotation', rotation)


    def get_transforms(self, X, cond):

        if self.condition_size == 0:
            b = self.translation(X)
            a = self.scale(X)

        # one condition for multiple X
        elif len(cond.shape) == 1:
            full_input = torch.hstack([
                X,
                torch.ones((X.shape[0], self.condition_size)) *cond
                ])

            b = self.translation(full_input)
            a = self.scale(full_input)

        # each X has condition
        else:
            b = self.translation(torch.hstack([X, cond]))
            a = self.scale(torch.hstack([X, cond]))
            
        return a, b
    

    def forward(self, x, conditions):
        x_upper = x[:, :self.upper ]
        x_lower = x[:, self.upper: ]
        
        s, t = self.get_transforms(x_upper, conditions)

        z_upper = x_upper
        z_lower = torch.exp(s) * x_lower + t

        z = torch.hstack([z_upper, z_lower]) @ self.rotation

        return z 
    
    def inverse(self, x, conditions):
        backrot_X = x @ self.rotation.T

        x_upper = backrot_X[:, :self.upper ]
        x_lower = backrot_X[:, self.upper: ]

        s, t = self.get_transforms(x_upper, conditions)

        z_upper = x_upper
        z_lower = (x_lower - t) * torch.exp(-s)

        z = torch.hstack([z_upper, z_lower])

        return z
    
class RealNVP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, blocks, condition_size = 0, coupling_layer = "Affine", printing = False):
        super(RealNVP, self).__init__()

        assert(hidden_size > input_size)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.blocks = blocks
        self.condition_size = condition_size

        if coupling_layer=="Affine":
            self.coupling_blocks = torch.nn.ModuleList([AffineCouplingLayer(self.input_size,self.hidden_size, self.condition_size) for i in range(self.blocks-1)])
            self.coupling_blocks.append(AffineCouplingLayer(self.input_size, self.hidden_size, self.condition_size, rotate=False))

        if printing == True:
            print("RealNVP initialization: -----------------------------------")
            print(f"input_size: {input_size}")
            print(f"hidden_size: {hidden_size}")
            print(f"coupling blocks: {self.coupling_blocks}")
            print("------------------------------------------------------------")

        self.Qs = torch.linalg.qr(torch.randn((self.blocks-1, self.input_size, self.input_size)))[0]

    def forward(self, x, conditions = None):
        for i, coupling_layer in enumerate(self.coupling_blocks):
            x = coupling_layer.forward(x, conditions)
            
        return x 
    
    def inverse(self, x, conditions = None):
        for i in range(self.blocks):
            x = self.coupling_blocks[self.blocks-1-i].inverse(x, conditions)

        return x

    def sample(self, num_samples, conditions):
        normal_samples = torch.randn((num_samples, self.input_size))
            
        samples = self.inverse(normal_samples, conditions)
        
        return samples

class LightningINN(L.LightningModule):
    def __init__(self, inn: RealNVP, encoder):
        super().__init__()

        self.inn = inn
        self.encoder = encoder

    def training_step(self, batch):
        x, conditions = batch    
        conditions = self.encoder(conditions)[0]  
                            
        size = x.shape[0]
        z = x
        log_det_jac = 0
        for block in self.inn.coupling_blocks:
            log_det_jac += torch.sum(block.get_transforms(z[:, :block.upper], conditions)[0])
            z = block(z, conditions)
            
        loss = ( 1 / size ) * ( ( torch.sum(torch.square(z)) / 2 ) - log_det_jac )
        
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

