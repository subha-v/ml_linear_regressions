import torch
import torch.nn as nn
# LINEAR REGRESSION
# f(x)=3x
#f(x) w*x
# we are going to learn the value of 3 from the data

x=torch.tensor([[1.],[2],[3],[4]]) #we now have a rank 2 tensor
y=torch.tensor([[3.],[6],[9],[12]])

x_test = torch.tensor([5.]) #we are seeing if we learn the correct value for x=5

class LinearModel(nn.Module): # we are creating a class for a linear model
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.lin = nn.Linear(input_dim, output_dim) #we take a value from an input dimension to an out put dimension
# we could have wrote input_dim =1 and output_dim=1

    def forward(self, x):
        return self.lin(x) #this creates a linear model and returns it
            #it's the exact thing as w*x
            #but it also adds something like y=mx to y=mx+b
            #this plugs in x to the linear model!!!!!!!!!!
   
model = LinearModel(1,1)


loss_func=nn.MSELoss()


num_iters = 70
lr = 0.1 #how big of a step u take

print(f"Prediction: f(5) = {model(x_test).item():.3f}") #.item so we can print out the value inside hte tensor
optimizer = torch.optim.SGD(model.parameters(),lr) #this is like our leg that gets down the hill


for i in range(num_iters):
    y_pred = model(x)
    loss = loss_func(y_pred,y)
#now compute gradiant
    loss.backward() #we can find the gradient for any function using this value

    #update the weights
    optimizer.step()#this optimizer thing helps u take a step
    
    optimizer.zero_grad() #take my gradient an zero it in place 
    #u need to make this zero so we can use this in new locations


    
    print(f"Iter {i+1}: loss = {loss:.8f}")


print(f"Prediction: f(5) = {model(x_test).item():.3}")




