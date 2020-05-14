import datetime

import datetime

import PIL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from ProgPlot.functions import convert, get_bar


class BarWriter:

    def __init__(self, df=None, fps=30):
        """
        Init of the bar writer.


        """
        self.df = df.reset_index(drop=True)

        self.category_col = None
        self.timeseries_col = None
        self.value_col = None
        self.category_values = None

        self.agg_type = None

        self.__allow_missing_date = None
        self.__resample = None

    def set_data(self, category_col=None, timeseries_col=None, value_col=None,
                 allow_missing_date=False, agg_type="count", resample=None):

        ## seconds (int) playback time
        ## category_col (str) the name of the coloumn to group by
        ## date_col (str) the name of the column with time series
        ## value_col (str) the name of the column with values to measure
        ## allow_missing_date (bool) True will only display the categoircal value on the chart when the first instance of the time series appears
        ## False will show it from the start with 0 value.
        ## use_top_x (int or None) None will use all categories Int will filter the top x values
        ## agg_type (str)  "count","mean" or "sum"

        assert type(
            category_col) == str and category_col in self.df.columns, "category_col is not str or not in dataframe columns"
        self.category_col = category_col

        assert type(
            timeseries_col) == str and timeseries_col in self.df.columns, "timeseries_col is not str or not in dataframe columns"
        self.timeseries_col = timeseries_col

        self.df[timeseries_col] = self.df[timeseries_col].apply(
            lambda x: pd.to_datetime(str(x)) if (type(x) == str or type(x) == int) else x)

        self.df = self.df.sort_values(timeseries_col)

        assert type(
            value_col) == str and value_col in self.df.columns, "value_col is not str or not in dataframe columns"
        self.value_col = value_col

        assert type(allow_missing_date) == bool, "allow_missing_date is not bool"
        self.__allow_missing_date = allow_missing_date

        assert type(agg_type) == str and agg_type in ["count", "mean",
                                                      "sum"], 'agg_type is not str or not in ["count","mean","sum"]'
        self.agg_type = agg_type

        self.__resample = self.__check_resample_valid(resample)

        self.category_values = self.df.groupby([self.category_col]).count().sort_values(self.value_col).index

    def __check_resample_valid(self, resample):

        if resample == None:
            return None

        str_type = []
        num_type = []
        for x in resample:
            try:
                num_type.append(int(x))
            except:
                str_type.append(x)

        txt = "".join(str_type).lower()
        if len(num_type) == 0:
            num = 1
        else:
            num_type = [str(x) for x in num_type]
            num = int("".join(num_type))

        assert txt in ["days", "minutes", "hours", "seconds", "weeks"], 'resample type is not valid'

        return (txt, num)

    def add_missing_dates(self, date_list, df):

        # make sure index
        df = df.set_index(df[self.timeseries_col])
        # get missing dates
        missing_dates = pd.concat([pd.Series(df.index), pd.Series(date_list)]).drop_duplicates(keep=False)

        # set cat name
        cat = df.iloc[0, 0]
        temp_df = df.reset_index(drop=True).groupby(self.timeseries_col).count().reset_index()
        temp_df = temp_df.reindex(missing_dates.to_list(), fill_value=0)
        temp_df[self.category_col] = cat
        temp_df[self.timeseries_col] = temp_df.index

        return pd.concat([df, temp_df]).sort_values(self.timeseries_col).reset_index(drop=True)

    def create_new_frame(self):

        # trim self.df by categories for speedier execution
        self.__data_df_temp = self.df[self.df[self.category_col].isin(self.category_values)][
            [self.category_col, self.timeseries_col, self.value_col]]

        max_date = pd.to_datetime(self.df[self.timeseries_col].max())
        min_date = pd.to_datetime(self.df[self.timeseries_col].min())
        date_diff = self.__data_df_temp[self.timeseries_col].drop_duplicates().diff().median()
        print(max_date, min_date, date_diff)
        # get dates
        dt_index = pd.DatetimeIndex(pd.date_range(min_date, periods=(max_date - min_date) / date_diff, freq=date_diff))

        # final_df = pd.DataFrame()

        for i, (df, cat) in enumerate(self.get_unique()):

            df = df[[self.timeseries_col, self.value_col, self.category_col]]

            df = self.add_missing_dates(dt_index, df)

            if self.__resample != None:
                temp_df = self.do_agg(df, cat)
            else:
                temp_df = df.copy()

            # temp_df = self.cumlative_df_temp.merge(temp_df,left_index=True,right_index=True)
            # temp_df.reindex(self.dt_index.index, fill_value=0)

            if i == 0:
                final_df = temp_df.copy()
            else:
                final_df = pd.concat([final_df, temp_df])

        final_df = final_df.reset_index(drop=True)
        final_df.columns = [self.timeseries_col, self.value_col, self.category_col]

        return final_df

    def get_unique(self):
        for cat in self.category_values:
            yield self.__data_df_temp[self.__data_df_temp[self.category_col] == cat], cat

    def do_agg(self, df, cat):

        if self.agg_type == "count":
            df = pd.DataFrame({"Value": df.set_index(df[self.timeseries_col]).resample(self.get_time_delta()).count()[
                self.value_col].cumsum()})
        elif self.agg_type == "mean":
            df = pd.DataFrame({"Value": df.set_index(df[self.timeseries_col]).resample(self.get_time_delta()).mean()[
                self.value_col].cumsum()})
        elif self.agg_type == "sum":
            df = pd.DataFrame({"Value": df.set_index(df[self.timeseries_col]).resample(self.get_time_delta()).sum()[
                self.value_col].cumsum()})

        df[self.category_col] = cat

        return df

    def get_time_delta(self):

        if self.__resample[0] == "days":
            return datetime.timedelta(days=self.__resample[1])
        elif self.__resample[0] == "weeks":
            return datetime.timedelta(weeks=self.__resample[1])
        elif self.__resample[0] == "months":
            return datetime.timedelta(months=self.__resample[1])
        elif self.__resample[0] == "years":
            return datetime.timedelta(years=self.__resample[1])
        elif self.__resample[0] == "minutes":
            return datetime.timedelta(minutes=self.__resample[1])
        elif self.__resample[0] == "seconds":
            return datetime.timedelta(seconds=self.__resample[1])
        elif self.__resample[0] == "hours":
            return datetime.timedelta(hours=self.__resample[1])

    def show_video(self):

        assert type(self._last_video_save) == str, "Video not rendered"

        return HTML(f"""
                    <video alt="test" controls>
                        <source src="{self._last_video_save}" type="video/mp4">
                    </video>
                """)

    def __assert_sort(self, sort):
        assert sort == True, "Cant display_top_x or use top_x while sort == False"

    def write_video(self, use_top_x=None, display_top_x=None, palette="magma", sort=True, x_label="", x_title="",
                    fps=30,
                    file_name="output.webm", dateformat=None, output_html=True, test=False, **kwargs):

        # save file name
        self._last_video_save = file_name

        if type(use_top_x) == int:
            self.__assert_sort(sort)
            self.category_values = self.df.groupby([self.category_col]).count().sort_values(self.value_col).tail(
                use_top_x).index

        # prepare df for redndering
        print("Creating resampled video dataframe. This may take a moment.")
        self.__video_df = self.create_new_frame()

        # unique values for palette colours
        if palette == None:
            uniques = list(self.category_values)
            palette = dict(zip(uniques, sns.color_palette(palette, n_colors=len(uniques))))

        # start timers
        start = datetime.datetime.now()

        # get unique dates
        unique_dates = self.__video_df[self.timeseries_col].unique()
        # squeeze if TEST mode
        if test:
            unique_dates = unique_dates[0:100]

        for i, dte in enumerate(unique_dates):

            # get dates df SORTED IF NEEDED
            if sort:
                temp_df = self.__get_temp_df_sort_values(dte)
            else:
                temp_df = temp_df = self.__get_temp_df_sort_values(dte, False)

            # filter display_top_x value to only SHOW the top x values.
            if type(display_top_x) == int:
                self.__assert_sort(sort)
                temp_df = temp_df.tail(display_top_x)

            # get plot
            fig = plt.figure(figsize=(18, 8), dpi=100)
            ax = sns.barplot(y=self.category_col,
                             x=self.value_col,
                             data=temp_df,
                             palette=palette,
                             **kwargs)
            ax.set_xlabel(x_label)

            if dateformat == None:
                ax.set_title(
                    f"{x_title} From {pd.to_datetime(min(self.__video_df[self.timeseries_col]))} To {pd.to_datetime(temp_df[self.timeseries_col].iloc[0])}")
            else:
                ax.set_title(
                    f"{x_title} From {pd.to_datetime(min(self.__video_df[self.timeseries_col])).strftime(dateformat)} To {pd.to_datetime(temp_df[self.timeseries_col].iloc[0]).strftime(dateformat)}")

            # save fig and reread as np array
            fig.savefig("temp_out.jpg")
            plt.close(fig)
            img = cv2.imread("temp_out.jpg")

            if i == 0:
                # set writer
                fourcc = cv2.VideoWriter_fourcc(*'VP80')
                out = cv2.VideoWriter("./" + file_name, fourcc=fourcc, fps=fps, frameSize=(img.shape[1], img.shape[0]))

            out.write(img)

            # LOGGING
            self.__write_log(start, i, unique_dates)

        # early test stopping
        if test:
            print("Testing Complete, sample file saved.")
            output_html = True

        # finalize file
        out.release()
        print("\nVideo creation complete")

        # return HTML output
        if output_html:
            return HTML(f"""
                    <video alt="test" controls>
                        <source src="{file_name}" type="video/mp4">
                    </video>
                """)

    def __write_log(self, start, i, unique_dates):

        time_end = datetime.datetime.now()
        total_time = (time_end - start).total_seconds()
        seconds_per = (total_time / (i + 1))
        seconds_left = (len(unique_dates) - (i + 1)) * seconds_per
        print(
            f"\rWriting Frame {i:>4}/{len(unique_dates)}  Render Speed: {(i + 1) / total_time:.1f}fps  Time taken: {convert(total_time)}  Time left: {convert(seconds_left)} {get_bar(i, len(unique_dates) - 1)}",
            end="")

    def __get_temp_df_sort_values(self, dte, sort=True):
        # filter the df by date and sort if needed
        if sort:
            return self.__video_df[self.__video_df[self.timeseries_col] == dte].sort_values(self.value_col)
        else:
            return self.__video_df[self.__video_df[self.timeseries_col] == dte]
