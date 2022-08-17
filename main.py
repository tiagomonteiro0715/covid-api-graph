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


criar um lugar na net com jupyter notebooks bons

usar seaborn para obter bons gráficos
dividir o codigo em varios ficheiros
fazer isto mesmo um bom projeto e acabar documentação
'''

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

'''
vão todos os paises para depois se escolhar batch size melhor

'''
numOfCountries = 175#number of cauntries é como o batch size
for i in range (numOfCountries):
    deathSorted.append(Deaths[i])
    coutriesSorted.append(countries[i])
    confirmedSorted.append(confirmed[i])


coutriesSorted = np.array(coutriesSorted)

deathTensor = torch.tensor(deathSorted, dtype=torch.float64)
confirmedTensor = torch.tensor(confirmedSorted, dtype=torch.float64)

input_size = numOfCountries
output_size = numOfCountries


deathTensor.requires_grad_(True)
confirmedTensor.requires_grad_(True)


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
optimizer = SGD(model.parameters(), lr=0.000000000000000001)


for epoch in range(501):
    total_loss = 0

    for iteration in range(len(confirmedTensor)):
        input_i = confirmedTensor[iteration]
        targets_i = deathTensor[iteration]
        output_i = model(input_i)


        loss = (output_i-targets_i)**2#this is the square residual loss function. there are others losso functions
        loss.backward()#calculate de derivative of the loss function with the respect of the parameters I want to optiize
        #loss.backwars adds derivatives  - accumulates the derivatives
        total_loss += float(loss)

    optimizer.step()
    optimizer.zero_grad()


    print("Step: " + str(epoch) + " Final Bias: " + str(model.weight.data))
    print("Step: " + str(epoch) + " Final Bias: " + str(model.bias.data))
    print("Step: " + str(epoch) + " Final Bias: " + str(model.final_bias.data) + "\n")
    print("\nTotal loss: " + str(total_loss)+"\n\n")

torch.save(model.state_dict(), "model.pth")

print(model(50000))

def barPlot(countries, deaths):
    fig, ax = plt.subplots()
    ax.bar(countries, deaths)
    plt.show()

def pieplot(countries, deaths):
    fig, ax = plt.subplots()
    ax.pie(deathSorted, radius=3, center=(4, 4), wedgeprops={"linewidth": 1, "edgecolor": "white"}, frame=True)
    plt.show()


df.to_excel("country.xlsx")
