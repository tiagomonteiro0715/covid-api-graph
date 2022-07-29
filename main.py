import requests
import pandas as pd
import matplotlib.pyplot as plt

#https://documenter.getpostman.com/view/10808728/SzS8rjbc#00030720-fae3-4c72-8aea-ad01ba17adf8
response_info = requests.get("https://api.covid19api.com/summary").json()

country_list = []
for country in response_info["Countries"]:
    country_list.append([country["Country"], country["TotalConfirmed"], country["TotalDeaths"]])

df = pd.DataFrame(data=country_list, columns=["Country", "Total_Confirmed", "TotalDeaths"])
df_sort = df.sort_values('TotalDeaths',ascending=False)

Deaths = df_sort['TotalDeaths'].tolist()
countries = df_sort['Country'].tolist()

deathsorted = []
coutriesorted = []
for i in range (5):
    deathsorted.append(Deaths[i])
    coutriesorted.append(countries[i])


def barPlot(countries, deaths):
    fig, ax = plt.subplots()
    ax.bar(countries, deaths, width=0.5, log = True)
    plt.show()


def pieplot(countries, deaths):
    fig, ax = plt.subplots()
    ax.pie(deathsorted, radius=4, center=(4, 4), wedgeprops={"linewidth": 1, "edgecolor": "white"}, frame=True)
    plt.show()

barPlot(coutriesorted, deathsorted)
df.to_excel("country.xlsx")
