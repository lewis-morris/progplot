import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ProgPlot import BarWriter

df = pd.read_csv("athlete_events.csv")
new_df = pd.concat([df,pd.get_dummies(df["Medal"],dummy_na=True)],axis=1)
bw = BarWriter(new_df)
bw.set_data("NOC", "Year", "Gold", agg_type="sum")
bw.write_video(test=True, sort=True, use_top_x=20, display_top_x=8)