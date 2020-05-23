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

df = pd.read_csv("./examples/athlete_events.csv")

#read the region data
noc = pd.read_csv("./examples/noc_regions.csv")

#merge the two
olymp_df = df.merge(noc,left_on="NOC",right_on="NOC")

#select categories we want
olymp_df = olymp_df[["region","Age","Height","Weight","Year","Sport","Medal"]]

#fix the medal column

olymp_df = pd.concat([olymp_df,pd.get_dummies(olymp_df["Medal"])],axis=1)

olymp_df["Total"] = olymp_df["Bronze"] + olymp_df["Gold"] + olymp_df["Silver"]

olymp_df.drop("Medal",axis=1, inplace=True)
olymp_df["Year"] = pd.to_datetime(olymp_df["Year"],format="%Y")



from progplot import BarWriter

bw = BarWriter()

bw.set_data(data=olymp_df, category_col="region", timeseries_col="Year", value_col="Age", groupby_agg="mean", resample_agg="mean", output_agg="4rolling", resample = "4y")

bw.set_display_settings(time_in_seconds=45, video_file_name = "mean_height_by_country.mp4")


bw.set_chart_options(x_tick_format="{:,.2f}",
                     palette="Pastel1",
                     title="Top 10 Rolling Mean Height <rollingdatetime> to <currentdatetime>",dateformat="%Y",
                     y_label="State",
                     use_top_x=20, display_top_x=10,
                     border_size=2, border_colour=(0.3,0.3,0.3),
                     font_scale=1.3,
                     use_data_labels="end",
                     squeeze_lower_x=0.2)
bw.test_chart(27)
