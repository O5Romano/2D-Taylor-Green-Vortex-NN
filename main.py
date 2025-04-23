
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)



v=torch.tensor([0.1])
L = torch.pi
T = 10
num_points = 10000
epochs = 23000

class NeuralNet(nn.Module):

    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, 3) # u,v,p

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
                nn.init.constant_(m.bias, 0.1)
    
    def forward(self, x,y, t):
        inputs = torch.cat((x,y, t), dim=1)
        x = torch.tanh(self.fc1(inputs))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        x = torch.tanh(self.fc6(x))
        x = self.fc7(x)
        return x

model = NeuralNet()

def velocity_loss(model,x0,y0,x_pi,y_pi,x,y,t):
    f1 = model(x,y0,t)
    f2 = model(x,y_pi,t)
    f3 = model(x0,y,t)
    f4 = model(x_pi,y,t)
    g = torch.exp(-2*v*t)
    return ((f1[:,0:1]-(torch.sin(x))*g)**2).sum() + ((f2[:,0:1]+(torch.sin(x))*g)**2).sum() + ((f3[:,0:1])**2).sum() + ((f4[:,0:1])**2).sum() + ((f1[:,1:2])**2).sum() + ((f2[:,1:2])**2).sum() + ((f3[:,1:2]+(torch.sin(y))*g)**2).sum() + ((f4[:,1:2]-(torch.sin(y))*g)**2).sum()

    

def loss_conservation(model,x,y,t):
     f = model(x,y,t) 
     fu = f[:,0:1]
     fv = f[:,1:2]
     fu_x = torch.autograd.grad(fu,x,create_graph=True,grad_outputs=torch.ones_like(fu))[0]
     fv_y = torch.autograd.grad(fv,y,create_graph=True,grad_outputs=torch.ones_like(fu))[0]
     return ((fu_x+fv_y)**2).sum()
def equation_loss(model,x,y,t,rho = torch.tensor([1.])):
    f = model(x,y,t)
    fu = f[:,0:1]
    fv = f[:,1:2]
    fu_t = torch.autograd.grad(fu, t, grad_outputs=torch.ones_like(fu), create_graph=True)[0]
    fv_t = torch.autograd.grad(fv, t, grad_outputs=torch.ones_like(fv), create_graph=True)[0]


    fu_x = torch.autograd.grad(fu, x, create_graph=True,grad_outputs=torch.ones_like(fu))[0]
    fu_y = torch.autograd.grad(fu, y, create_graph=True,grad_outputs=torch.ones_like(fu))[0]
    fv_y = torch.autograd.grad(fv, y, create_graph=True,grad_outputs=torch.ones_like(fv))[0]
    fv_x = torch.autograd.grad(fv, x, create_graph=True,grad_outputs=torch.ones_like(fv))[0]
    fu_xx = torch.autograd.grad(fu_x, x, create_graph=True,grad_outputs=torch.ones_like(fu_x))[0]
    fu_yy = torch.autograd.grad(fu_y, y, create_graph=True,grad_outputs=torch.ones_like(fu_y))[0]
    fv_yy = torch.autograd.grad(fv_y, y, create_graph=True,grad_outputs=torch.ones_like(fv_y))[0]
    fv_xx = torch.autograd.grad(fv_x, x, create_graph=True,grad_outputs=torch.ones_like(fv_x))[0]

    fp_x = torch.autograd.grad(f[:, 2:3], x, create_graph=True,grad_outputs=torch.ones_like(f[:, 2:3]))[0]
    fp_y =torch.autograd.grad(f[:, 2:3], y, create_graph=True,grad_outputs=torch.ones_like(f[:, 2:3]))[0]


    return ((fu_t+fu*fu_x+fv*fu_y + (1/rho)*fp_x-v*(fu_xx+fu_yy))**2).sum() + ((fv_t+fu*fv_x+fv*fv_y+(1/rho)*fp_y-v*(fv_xx+fv_yy))**2).sum()  

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50,T_mult=2)



x_bc_0 = torch.zeros(num_points,1)
x_bc_L = torch.ones(num_points,1) * L
y_bc_0 = torch.zeros(num_points,1)
y_bc_L = torch.ones(num_points,1) * L
x_bc_0.requires_grad = True
x_bc_L.requires_grad = True
y_bc_0.requires_grad = True
y_bc_L.requires_grad = True


for epoch in range(epochs):

    if epoch%1000 == 0: 
        x_pde = torch.rand(num_points, 1) * L
        y_pde = torch.rand(num_points, 1) * L
        t_pde = torch.rand(num_points, 1) * T
        x_pde.requires_grad = True
        y_pde.requires_grad = True
        t_pde.requires_grad = True


    optimizer.zero_grad()

    loss_pde = equation_loss(model,x_pde,y_pde,t_pde)
    loss_cons = loss_conservation(model,x_pde,y_pde,t_pde)
    loss_vel = velocity_loss(model,x_bc_0,y_bc_0,x_bc_L,y_bc_L,x_pde,y_pde,t_pde)

    cost = loss_pde+loss_cons+loss_vel
    cost.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step()

    scheduler.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {cost.item()}')

torch.save(model.state_dict(), 'checkpoint.pth')

#model.load_state_dict(torch.load('checkpoint.pth'))

model.eval()
step = 1000
X, Y = np.meshgrid(np.linspace(0,L,step),np.linspace(0,L,step))
X_tensor = (torch.tensor(X).flatten().unsqueeze(1)).to(torch.float32)
Y_tensor = (torch.tensor(Y).flatten().unsqueeze(1)).to(torch.float32)
output = model(X_tensor,Y_tensor,torch.zeros_like(X_tensor).to(torch.float32))
U = output[:,0]
V = output[:,1]
plt.streamplot(X,Y,U.cpu().detach().numpy().reshape(X.shape),V.cpu().detach().numpy().reshape(X.shape))
plt.plot()
plt.show()