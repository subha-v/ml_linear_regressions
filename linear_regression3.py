import torch
import torch.nn as nn
# LINEAR REGRESSION
# f(x)=3x
#f(x) w*x
# we are going to learn the value of 3 from the data

x=torch.tensor([1.,2,3,4])
y=torch.tensor([3.,6,9,12])

#want the computer to learn the parameter is 3

w = torch.tensor(0.0, requires_grad=True) #this means wwhen we do a backward call, we are creating the gradiant attribute so we can take the derivative with respect to it
#we want to get hte gradiant of hte loss respect to w
#we are trying to find the loss that corresponds to the best w


def model(x):
    return w*x

loss_func=nn.MSELoss()


num_iters = 20
lr = 0.1 #how big of a step u take

print(f"Prediction: f(4) = {model(4):.3f}")
optimizer = torch.optim.SGD([w],lr) #this is like our leg that gets down the hill



for i in range(num_iters):
    y_pred = model(x)
    loss = loss_func(y_pred,y)
#now compute gradiant
    loss.backward() #we can find the gradient for any function using this value

    #update the weights
    optimizer.step()#this optimizer thing helps u take a step
    
    optimizer.zero_grad() #take my gradient an zero it in place 
    #u need to make this zero so we can use this in new locations


    
    print(f"Iter {i+1}: w = {w:.3f}, loss = {loss:.8f}")

    if(abs(loss)<1e-6):
        break
    

print(f"Prediction: f(4) = {model(4):.3}")




