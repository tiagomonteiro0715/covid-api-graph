from pickle import FALSE
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch #create tensors to raw data and weights
import torch.nn as nn # make tensores part of nn
from torch.optim import SGD #stochastic gradient descend
'''
## Ver recursos e de codigo que goste, fazer projetos

usar seaborn e matplotlib

https://pytorch.org/docs/stable/distributions.html

https://github.com/fossasia/visdom

https://github.com/ritchieng/the-incredible-pytorch#Tutorials

https://github.com/tomgoldstein/loss-landscape

https://github.com/ritchieng/the-incredible-pytorch

covid api predcit

https://github.com/yunjey/pytorch-tutorial

https://stackoverflow.com/questions/33962226/common-causes-of-nans-during-training-of-neural-networks?noredirect=1&lq=1

https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/linear_regression/main.py


desenhar na descrição do projeto isto: https://alexlenail.me/NN-SVG/index.html

colocar também grafico final - o peso é o declice. 
Nota: será que é preciso multiplicar o peso por alguma coisa para acertar nos valores?


criar um lugar na net com jupyter notebooks bons

usar seaborn para obter bons gráficos
dividir o codigo em varios ficheiros
fazer isto mesmo um bom projeto e acabar documentação
'''

#https://documenter.getpostman.com/view/10808728/SzS8rjbc#00030720-fae3-4c72-8aea-ad01ba17adf8
# Getting the information from the api and converting it to json format
response_info = requests.get("https://api.covid19api.com/summary").json()

# Creating a list of lists with the countries, total confirmed cases and total deaths.
country_list = []
for country in response_info["Countries"]:
    country_list.append([country["Country"], country["TotalConfirmed"], country["TotalDeaths"]])

# Creating a dataframe with the information from the api and then sorting it by the number of deaths
df = pd.DataFrame(data=country_list, columns=["Country", "Total_Confirmed", "TotalDeaths"])
df_sort = df.sort_values('TotalDeaths',ascending=False)

# Creating a list of the countries, total confirmed cases and total deaths.
confirmed = df_sort['Total_Confirmed'].tolist()
Deaths = df_sort['TotalDeaths'].tolist()
countries = df_sort['Country'].tolist()

# Creating a list of the countries, total confirmed cases and total deaths sorted
deathSorted = []
coutriesSorted = []
confirmedSorted = []

numOfCountries = 175#number of cauntries é como o batch size
for i in range (numOfCountries):
    deathSorted.append(Deaths[i])
    coutriesSorted.append(countries[i])
    confirmedSorted.append(confirmed[i])


# Converting the list of countries into a numpy array.
coutriesSorted = np.array(coutriesSorted)

# Converting the list of countries into a tensor
deathTensor = torch.tensor(deathSorted, dtype=torch.float64)
confirmedTensor = torch.tensor(confirmedSorted, dtype=torch.float64)

# The number of countries that we are using to train the model.
input_size = numOfCountries
output_size = numOfCountries


# Telling pytorch that we want to calculate the derivative of the loss function with respect to the
# tensors.
deathTensor.requires_grad_(True)
confirmedTensor.requires_grad_(True)


# The class creates a neural network with one hidden layer and one output layer. The hidden layer has
# one neuron and the output layer has one neuron. The hidden layer has a weight and a bias and the
# output layer has a bias
class BasicNN_train(nn.Module):
    def __init__(self):#creates and inicializes weights and biases + all objects will have it
        """
        It creates a class called BasicNN_train, which inherits from the nn.Module class. 
        
        The __init__ function is a special function that is called when an object is created. 
        
        The super function is used to inherit the __init__ function from the nn.Module class. 
        
        The nn.Parameter function is used to create a parameter that can be optimized. 
        
        The requires_grad function is set to True, which means that the parameter will be optimized. 
        
        The weight and bias parameters are created and initialized to 0.0. 
        
        The final_bias parameter is created and initialized to 0.0. 
        
        The final_bias parameter is not used in the forward function, but it is used in the loss function. 
        
        The final_bias parameter is used
        """
        super(BasicNN_train, self).__init__()
        self.weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.final_bias = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        
    def forward(self, input):#Makes foward pass trought neural network to see what input will give out
        """
        > The function takes in an input, multiplies it by the weight, adds the bias, and then adds the
        final bias
        
        :param input: The input to the neural network
        """
        output = (input*self.weight) +self.bias
        modelOutput = output + self.final_bias

        return modelOutput


# Creating a model and an optimizer. The model is a neural network with one hidden layer and one
# output layer. The hidden layer has one neuron and the output layer has one neuron. The hidden layer
# has a weight and a bias and the output layer has a bias. The optimizer is used to optimize the
# parameters of the neural network.
model = BasicNN_train()
optimizer = SGD(model.parameters(), lr=0.000000000000000001)


# Training the model.
for epoch in range(501):
    total_loss = 0

# Calculating the loss function for each country.
    for iteration in range(len(confirmedTensor)):
        input_i = confirmedTensor[iteration]
        targets_i = deathTensor[iteration]
        output_i = model(input_i)


        loss = (output_i-targets_i)**2#this is the square residual loss function. there are others losso functions
        loss.backward()#calculate de derivative of the loss function with the respect of the parameters I want to optiize
        #loss.backwars adds derivatives  - accumulates the derivatives
        total_loss += float(loss)

# Updating the parameters of the model.
    optimizer.step()
# Setting the gradient to zero.
    optimizer.zero_grad()


# Printing the weight, bias and final bias of the model.
    print("Step: " + str(epoch) + " Final Bias: " + str(model.weight.data))
    print("Step: " + str(epoch) + " Final Bias: " + str(model.bias.data))
    print("Step: " + str(epoch) + " Final Bias: " + str(model.final_bias.data) + "\n")
    print("\nTotal loss: " + str(total_loss)+"\n\n")

# Saving the model to a file called model.pth
torch.save(model.state_dict(), "model.pth")

# Printing the output of the model when the input is 50000.
print(model(50000))

def barPlot(countries, deaths):
    """
    > The function takes two lists as input, one for the x-axis and one for the y-axis, and plots a bar
    chart
    
    :param countries: a list of countries
    :param deaths: a list of integers
    """
    fig, ax = plt.subplots()
    ax.bar(countries, deaths)
    plt.show()

def pieplot(countries, deaths):
    """
    This function takes in two lists, one of countries and one of deaths, and plots a pie chart of the
    deaths in each country
    
    :param countries: list of countries
    :param deaths: a list of the number of deaths in each country
    """
    fig, ax = plt.subplots()
    ax.pie(deathSorted, radius=3, center=(4, 4), wedgeprops={"linewidth": 1, "edgecolor": "white"}, frame=True)
    plt.show()

df.to_excel("country.xlsx")
