# %%

import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from progplot import BarWriter

df = pd.read_csv("./examples/athlete_events.csv")

# read the region data
noc = pd.read_csv("./examples/noc_regions.csv")

# merge the two
olymp_df = df.merge(noc, left_on="NOC", right_on="NOC")

# select categories we want
olymp_df = olymp_df[["region", "Age", "Height", "Weight", "Year", "Sport", "Medal", "NOC"]]

# fix the medal column

olymp_df = pd.concat([olymp_df, pd.get_dummies(olymp_df["Medal"])], axis=1)

olymp_df["Total"] = olymp_df["Bronze"] + olymp_df["Gold"] + olymp_df["Silver"]

olymp_df.drop("Medal", axis=1, inplace=True)
olymp_df["Year"] = pd.to_datetime(olymp_df["Year"], format="%Y")

from progplot import BarWriter

bw = BarWriter()

bw.set_data(data=olymp_df, category_col="region", timeseries_col="Year", value_col="Total", groupby_agg="sum", resample_agg="sum", output_agg="cumsum", resample = "4y")
bw.set_display_settings(time_in_seconds=30, video_file_name = "total_medals_by_country.mp4")


# %%

bw.set_chart_options(x_tick_format="{:,.0f}",
                     palette="Pastel1",
                     title="Top 5 Countries by Total Medals from <mindatetime> to <currentdatetime>",dateformat="%Y",
                     y_label="State",
                     use_top_x=20, display_top_x=5,
                     border_size=2, border_colour=(0.3,0.3,0.3),
                     font_scale=1.3,
                     use_data_labels="end",
                     squeeze_lower_x="1000") # <----------- HERE either enter the percentace lower than the minimum data value you want the x value to be. OR the absolute value i.e 1000. *** NOTE: Will change to the nearest MAJOR TICK MARK

bw.test_chart(7)



