import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from progplot import BarWriter

#df = pd.read_csv("athlete_events.csv")
#new_df = pd.concat([df,pd.get_dummies(df["Medal"],dummy_na=True)],axis=1)
#bw = BarWriter(new_df)
#bw.set_data("NOC", "Year", "Gold", resample="4y")
#bw.write_video(test=True, sort=True, use_top_x=15, display_top_x=15, time_in_seconds=20)


#df = pd.read_csv("https://covid.ourworldindata.org/data/owid-covid-data.csv")
#df = df[["location","date","new_deaths"]]
#new_df = df[df["location"].isin(["International","World"])==False]
#bw = BarWriter(new_df)
#bw.set_data("location", "date", "new_deaths", resample="1d")
#bw.category_values
#bw.write_video(test=False, sort=True, use_top_x=15, display_top_x=15,time_in_seconds=40)


#df = pd.read_csv("obesity-cleaned.csv", index_col=0)
#new_df = df [df["Sex"]=="Both sexes"]
#new_df["ob"] = new_df["Obesity (%)"].apply(lambda x: x.split(" ")[0])
#new_df = new_df[new_df["ob"] != "No"]
#bw = BarWriter(new_df)
#bw.set_data("Country", "Year", "ob", resample="1y", groupby_agg="mean", resample_agg="mean",output_agg="4rolling")
#bw.set_display_settings(sort=True, use_top_x=25, display_top_x=25, time_in_seconds=30, palette="twilight_shifted",fps=50, x_title="Obesity % by Country", x_label="Average % of Obese Adults", dateformat="%Y",dpi=75)
#bw.write_video()

#df = pd.read_csv("athlete_events.csv")
#new_df = pd.concat([df,pd.get_dummies(df["Medal"],dummy_na=True)],axis=1)
#new_df["total_meds"] = new_df["Bronze"] + new_df["Gold"] + new_df["Silver"]
#from progplot import BarWriter, LineWriter
#bw = BarWriter(new_df)
#bw.set_data("NOC", "Year", "total_meds", resample="1y", groupby_agg="sum", resample_agg="sum",output_agg="cumsum")
#bw.set_display_settings(sort=True, use_top_x=10, display_top_x=10,time_in_seconds=30,palette="twilight_shifted",fps=50,x_title="Obesity % by Country", x_label="Average % of Obese Adults", dateformat="%Y",dpi=75)

#bw.test_chart()
#bw.write_video()

os.system("wget 'https://flagpedia.net/data/flags/w320.zip'")
import zipfile
with zipfile.ZipFile("w320.zip", 'r') as zip_ref:
    zip_ref.extractall("./icons/flags/")

df_country = pd.read_csv("covid_19_clean_complete.csv")
df_country["Date"] = pd.to_datetime(df_country["Date"])
df_country

codes = pd.read_html("https://www.iban.com/country-codes",attrs = {'id': 'myTable'})

df_country = df_country.merge(codes[0],left_on="Country/Region", right_on="Country")
df_country = df_country[["Country/Region","Date","Confirmed","Deaths","Alpha-2 code"]]
df_country

countries = list(df_country.loc[:, "Country/Region"].unique())
codes = list(df_country.loc[:, "Alpha-2 code"].unique())

image_dict = {country:f"./icons/flags/{str(code).lower()}.png" for country,code in zip(countries,codes)}
image_dict

bw = BarWriter()

bw.set_data(df_country, "Country/Region", "Date", "Deaths", resample="1d", groupby_agg="sum", resample_agg="sum",output_agg=None)

bw.set_display_settings(time_in_seconds=45, video_file_name = "deathsbycountrywithflag.mp4")

bw.set_chart_options(x_tick_format="{:,.0f}", dateformat="%Y-%m-%d",
                     palette="summer",
                     title="Top 15 Countries by Total Deaths <mindatetime> to <currentdatetime>",
                     use_top_x=30, display_top_x=15,
                     border_size=2, border_colour=(0.12,0.12,0.12),
                     font_scale=1.6,
                     use_data_labels="end", convert_bar_to_image=True,image_dict={})  # <--- Add image_dict and set convert_bar_to_image=True
bw.test_chart(100)