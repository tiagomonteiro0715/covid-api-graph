import requests
import pandas as pd

response_info = requests.get("https://api.covid19api.com/summary").json()

country_list = []
for country in response_info["Countries"]:
    country_list.append([country["Country"], country["TotalConfirmed"], country["TotalDeaths"]])

df = pd.DataFrame(data=country_list, columns=["Country", "Total_Confirmed", "TotalDeaths"])

df.to_csv("country.csv")
