import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch #create tensors to raw data and weights
import torch.nn as nn # make tensores part of nn
from torch.optim import SGD #stochastic gradient descend

'''
# Links para resolver o problema

usar seaborn e matplotlib

https://pytorch.org/docs/stable/distributions.html

https://github.com/fossasia/visdom

https://github.com/ritchieng/the-incredible-pytorch#Tutorials

https://github.com/tomgoldstein/loss-landscape

https://github.com/ritchieng/the-incredible-pytorch

covid api predcit

https://github.com/yunjey/pytorch-tutorial

https://stackoverflow.com/questions/33962226/common-causes-of-nans-during-training-of-neural-networks?noredirect=1&lq=1

gerar com https://pytorch.org/docs/stable/generated/torch.randn.html dados para ver se é das arrays que isto não está a funcionar


https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/linear_regression/main.py

criar um lugar na net com jupyter notebooks bons

'''

#instalar e usar seaborn

#https://documenter.getpostman.com/view/10808728/SzS8rjbc#00030720-fae3-4c72-8aea-ad01ba17adf8
response_info = requests.get("https://api.covid19api.com/summary").json()

country_list = []
for country in response_info["Countries"]:
    country_list.append([country["Country"], country["TotalConfirmed"], country["TotalDeaths"]])

df = pd.DataFrame(data=country_list, columns=["Country", "Total_Confirmed", "TotalDeaths"])
df_sort = df.sort_values('TotalDeaths',ascending=False)

confirmed = df_sort['Total_Confirmed'].tolist()
Deaths = df_sort['TotalDeaths'].tolist()
countries = df_sort['Country'].tolist()

deathSorted = []
coutriesSorted = []
confirmedSorted = []


numOfCountries = 20
for i in range (numOfCountries):
    deathSorted.append(Deaths[i])
    coutriesSorted.append(countries[i])
    confirmedSorted.append(confirmed[i])

#-------------------------------------------------------------------------------------
#Modelo começa aqui
deathSorted = np.array(deathSorted)
confirmedSorted = np.array(confirmedSorted)
coutriesSorted = np.array(coutriesSorted)


input_size = numOfCountries
output_size = numOfCountries


deathTensor = torch.from_numpy(deathSorted)
confirmedTensor = torch.from_numpy(confirmedSorted)

class BasicNN_train(nn.Module):
    def __init__(self):#creates and inicializes weights and biases + all objects will have it
        super(BasicNN_train, self).__init__()
        self.weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.final_bias = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        
    def forward(self, input):#Makes foward pass trought neural network to see what input will give out
        output = (input*self.weight) +self.bias
        modelOutput = output + self.final_bias

        return modelOutput


model = BasicNN_train()
optimizer = SGD(model.parameters(), lr=0.0001)

for epoch in range(100):
    total_loss = 0

    for iteration in range(len(confirmedTensor)):
        input_i = confirmedTensor[iteration]
        targets_i = deathTensor[iteration]

        input_i = input_i.detach()

        targets_i = targets_i.detach()
        
        output_i = model(input_i)

        print("iteration: " + str(iteration) + "  " + "output_i: " + str(output_i))

        loss = (output_i-targets_i)**2#this is the square residual loss function. there are others losso functions

        optimizer.zero_grad()
        loss.backward()#calculate de derivative of the loss function with the respect of the parameters I want to optiize
        #loss.backwars adds derivatives  - accumulates the derivatives
        optimizer.step()
        total_loss += float(loss)
        
    if (total_loss < 0.0001):
        print("Num stepts: " + str(epoch))
        break
    #print("total loss: " + str(total_loss))

    #print("Step: " + str(epoch) + " Final Bias: " + str(model.final_bias.data) + "\n")

#-------------------------------------------------------------------------------------

def barPlot(countries, deaths):
    fig, ax = plt.subplots()
    ax.bar(countries, deaths)
    plt.show()

def pieplot(countries, deaths):
    fig, ax = plt.subplots()
    ax.pie(deathSorted, radius=3, center=(4, 4), wedgeprops={"linewidth": 1, "edgecolor": "white"}, frame=True)
    plt.show()


df.to_excel("country.xlsx")
