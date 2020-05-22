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

df = pd.read_csv("usa_county_wise.csv")
df["Date"] = pd.to_datetime(df["Date"])
from progplot import BarWriter

bw = BarWriter()

bw.set_data(df, "Province_State", "Date", "Deaths", resample="1w", groupby_agg="sum", resample_agg="mean",output_agg=None)

bw.set_display_settings(time_in_seconds=10, video_file_name = "deathsbystate.mp4")

bw.set_chart_options(x_tick_format="{:,.0f}", dateformat="%Y-%m",
                     palette="copper",
                     title="Top 15 Weekly Deaths <currentdatetime>", y_label="State",
                     use_top_x=30, display_top_x=15,
                     border_size=2, border_colour=(0.12,0.12,0.12),
                     font_scale=1.6, title_font_size=18,x_label_font_size=16,
                     use_data_labels="end")

bw.test_chart()
bw.write_video()