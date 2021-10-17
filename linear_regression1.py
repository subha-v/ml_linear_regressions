import torch

# LINEAR REGRESSION
# f(x)=3x
#f(x) w*x
# we are going to learn the value of 3 from the data

x=torch.tensor([1.,2,3,4])
y=torch.tensor([3.,6,9,12])

#want the computer to learn the parameter is 3

w=0.0

def model(x):
    return w*x

def loss_func(y_pred, y):
    # mean squared loss function
    return((y_pred-y)**2).mean()

def gradient(x,y_pred,y):
    return((2*x) * (y_pred-y)).mean() #check papper for why this is true, its the derivative

print(f"Prediction: f(4) = {model(4):.3f}")


num_iters = 80
lr = 0.01

for i in range(num_iters):
    y_pred = model(x)
    loss = loss_func(y_pred,y)
#now compute gradiant
    grad = gradient(x,y_pred,y) #this computes the derivative 

    #update the weights
    w = w- lr * grad #we are making the model learn
    
    print(f"Iter {i+1}: w = {w:.3f}, loss = {loss:.8f}")

    if(abs(loss)<1e-6):
        break
    

print(f"Prediction: f(4) = {model(4):.3}")




