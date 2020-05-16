import datetime

import datetime

import PIL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from IPython.display import HTML
import ffmpy

from ProgPlot.functions import convert, get_bar

class __base_writer:
    
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
        self.__video_options = {}

        self.__resample = None

    def set_data(self, category_col, timeseries_col, value_col,
                 groupby_agg="sum", resample_agg="sum", output_agg="cumsum", resample=None):

        ## seconds (int) playback time
        ## category_col (str) the name of the coloumn to group by
        ## date_col (str) the name of the column with time series
        ## value_col (str) the name of the column with values to measure
        ## allow_missing_date (bool) True will only display the categoircal value on the chart when the first instance of the time series appears
        ## False will show it from the start with 0 value.
        ## use_top_x (int or None) None will use all categories Int will filter the top x values
        ## agg_type (str)  "count","mean" or "sum"

        self.__video_options = {}

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

        assert type(groupby_agg) == str and groupby_agg in ["count", "mean",
                                                      "sum"], 'groupby_agg is not str or not in ["count","mean","sum"]'
        self.groupby_agg =groupby_agg

        assert type(resample_agg) == str and resample_agg in ["count", "mean",
                                                      "sum"], 'resample_agg is not str or not in ["count","mean","sum"]'
        self.resample_agg=resample_agg

        assert type(output_agg) == str and output_agg == "cumsum" or output_agg.find("rolling") >= 0, 'output_agg is not str or not in ["rolling", "cumsum"]'
        self.output_agg=output_agg

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

        assert txt in ["d", "m", "h", "s", "w","y","rolling"], 'resample type is not valid'

        return (txt, num)

    def add_missing_dates(self, date_list, df):

        # make sure index
        df = df.set_index(df[self.timeseries_col])
        # get missing dates
        missing_dates = pd.concat([pd.Series(df.index), pd.Series(date_list)]).drop_duplicates(keep=False)

        # set cat name
        cat = df[self.category_col].unique()[0]
        temp_df = df.reset_index(drop=True).groupby(self.timeseries_col).count().reset_index()
        temp_df = temp_df.reindex(missing_dates.to_list(), fill_value=0)
        temp_df[self.category_col] = cat
        temp_df[self.timeseries_col] = temp_df.index
        temp_df.reset_index(drop=True,inplace=True)
        for x in range(len(temp_df[self.timeseries_col])):
            if temp_df.loc[x,self.timeseries_col] in df[self.timeseries_col]:
                temp_df.drop(x, inplace=True)
        return pd.concat([df, temp_df]).sort_values(self.timeseries_col).reset_index(drop=True)

    def create_new_frame(self):

        # trim self.df by categories for speedier execution
        self.__data_df_temp = self.df[self.df[self.category_col].isin(self.category_values)][
            [self.category_col, self.timeseries_col, self.value_col]]

        self.max_date = pd.to_datetime(self.df[self.timeseries_col].max())
        self.min_date = pd.to_datetime(self.df[self.timeseries_col].min())
        self.date_diff = self.__data_df_temp[self.timeseries_col].drop_duplicates().diff().median()

        # get dates
        #TODO check this is working correctly - was not generating enough dates
        dt_index = pd.DatetimeIndex(pd.date_range(self.min_date,self.max_date, freq=self.__resample))

        # final_df = pd.DataFrame()

        for i, (df, cat) in enumerate(self.get_unique()):

            df = df[[self.timeseries_col, self.value_col, self.category_col]]

            df = self.add_missing_dates(dt_index, df)

            if self.__resample != None:
                temp_df = self.do_agg(df, cat, True)
            else:
                temp_df = self.do_agg(df, cat, False)

            # temp_df = self.cumlative_df_temp.merge(temp_df,left_index=True,right_index=True)
            # temp_df.reindex(self.dt_index.index, fill_value=0)

            if i == 0:
                final_df = temp_df.copy()
            else:
                final_df = pd.concat([final_df, temp_df])

        final_df = final_df.reset_index()
        final_df.columns = [self.timeseries_col, self.value_col, self.category_col]

        return final_df

    def fix_dates(self,df):
        df = df.reset_index(drop=True)
        for x in range(len(df)):
            try:
                diff = df.loc[x + 1, self.timeseries_col] - df.loc[x, self.timeseries_col]
                print(diff)
                if diff * .95 < self.date_diff:
                    df.loc[x, self.value_col] += df.loc[x + 1, self.value_col]
                    df.drop(x + 1, inplace=True)
            except Exception as e:
                print(e)
        return df
    def get_unique(self):
        for cat in self.category_values:
            yield self.__data_df_temp[self.__data_df_temp[self.category_col] == cat], cat

    def do_agg(self, df, cat, resample=False):

        df[self.value_col] = df[self.value_col].astype(float)


        if self.groupby_agg == "sum":
            df = df.groupby([self.timeseries_col,self.category_col]).sum().reset_index()
        elif self.groupby_agg == "mean":
            try:
                df = df.groupby([self.timeseries_col, self.category_col]).mean().reset_index()
            except pd.core.base.DataError:
                pass

        elif self.groupby_agg == "count":
            df = df.groupby([self.timeseries_col, self.category_col]).count().reset_index()
        else:
            pass

        if self.resample_agg == "sum":
            df = df.set_index(df[self.timeseries_col]).resample(self.__resample).sum()
        elif self.resample_agg == "mean":
            df[self.value_col] = df[self.value_col].apply(lambda x: np.nan if x == 0 else float(x))
            df = df.set_index(df[self.timeseries_col])[self.value_col].resample(self.__resample).mean()
            df = pd.DataFrame({self.value_col:df})
        elif self.resample_agg == "count":
            df = df.set_index(df[self.timeseries_col]).resample(self.__resample).count()
        else:
            pass

        if self.output_agg == "cumsum":
            df = pd.DataFrame({"Value":df[self.value_col].cumsum()})

        #TODO add this
        elif self.output_agg.find("rolling") >= 0:
            _, num = self.__check_resample_valid(self.output_agg)
            df = df.rolling(window=num).mean()
            df.dropna(inplace=True)
        #TODO add this
        else:
            pass

        df[self.category_col] = cat

        return df

    def create_gif(self):


        ff = ffmpy.FFmpeg(
            inputs={"output.webm": None},
            outputs={"cash.gif": None})
        ff.run()

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


    def set_display_settings(self, use_top_x=None, display_top_x=None,
                           palette="magma", sort=True, x_label="",
                           x_title="", fps=30, time_in_seconds=None,
                           file_name="output.webm", dateformat=None,
                           palette_keep=True, figsize=(14, 8), dpi=100,
                             **kwargs):

        # save file name
        self._last_video_save = file_name

        # prepare df for redndering
        print("Creating resampled video dataframe. This may take a moment.")
        self.__video_df = self.create_new_frame()

        if type(use_top_x) == int:
            self.__assert_sort(sort)
            max_date = self.__video_df[self.timeseries_col].tail(1).item()
            self.category_values = \
            self.__video_df[self.__video_df[self.timeseries_col] == max_date].sort_values(self.value_col)[
                self.category_col].iloc[-use_top_x:]
            self.__video_df = self.__video_df[self.__video_df[self.category_col].isin(self.category_values)]

        # unique values for palette colours
        if palette_keep:
            uniques = list(self.category_values)
            palette = dict(zip(uniques, sns.color_palette(palette, n_colors=len(uniques))))

        # get unique dates
        unique_dates = self.__video_df[self.timeseries_col].unique()

        ## get times per frame.
        if not time_in_seconds == None:
            total_frames = time_in_seconds * fps
            frames_per_image = int(total_frames/len(unique_dates))
        else:
            frames_per_image = 1

        #set options for
        self.__video_options = {"unique_dates":unique_dates,
                                "frames_per_image":frames_per_image,
                                "display_top_x":display_top_x,
                                "use_top_x":use_top_x,
                                "palette":palette,
                                "sort":sort,
                                "dateformat":dateformat,
                                "x_label":x_label,
                                "x_title":x_title,
                                "**kwargs":kwargs,
                                "looptimes":0,
                                "file_name":file_name,
                                "figsize":figsize,
                                "dpi":dpi,
                                "fps":fps}

    def test_chart(self,frame_no=None, as_pil=True):

        assert self.__video_options != {}, "Please set video settings first"

        if frame_no == None:
            frame_no = np.random.randint(0,len(self.__video_options["unique_dates"])-1)

        df_date = self.get_date_df(frame_no)

        img = self.get_chart(df_date)

        if as_pil:
            return PIL.Image.fromarray(img)
        else:
            return img


    def write_video(self,  output_html=True, test=False):

        assert self.__video_options != {}, "Please set video settings first"

        self.__video_options["looptimes"] = 0
        self.__video_options['starttime'] = datetime.datetime.now()

        # squeeze if TEST mode
        if test:
            dte_len = len(self.__video_options["unique_dates"])
            self.__video_options["unique_dates"] = self.__video_options["unique_dates"][0:int(dte_len*.05)]

        for i, dte in enumerate(self.__video_options["unique_dates"]):

            df_date = self.get_date_df(i)

            img = self.get_chart(df_date)

            if i == 0:
                # set writer
                fourcc = cv2.VideoWriter_fourcc(*'VP80')
                out = cv2.VideoWriter("./" + self.__video_options["file_name"], fourcc=fourcc,
                                      fps=self.__video_options["fps"], frameSize=(img.shape[1], img.shape[0]))

            out = self.write_extra_frames(i, out, img, df_date)

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
                        <source src="{self.__video_options["file_name"]}" type="video/mp4">
                    </video>
                """)

    def get_date_df(self, i):
            # get dates df SORTED IF NEEDED
        self.__assert_sort(self.__video_options["sort"])

        if self.__video_options["sort"]:
            temp_df = self.__get_temp_df_sort_values(self.__video_options["unique_dates"][i])
        else:
            temp_df = self.__get_temp_df_sort_values(self.__video_options["unique_dates"][i])

        # filter display_top_x value to only SHOW the top x values.
        if type(self.__video_options["display_top_x"]) == int:
            self.__assert_sort(self.__video_options["sort"])
            temp_df = temp_df.tail(self.__video_options["display_top_x"])

        return temp_df

    def get_chart(self, df_date):

        # get plot
        fig = plt.figure(figsize=self.__video_options["figsize"] , dpi=self.__video_options["dpi"])
        ax = sns.barplot(y=self.category_col,
                         x=self.value_col,
                         data=df_date,
                         palette=self.__video_options['palette'],
                         **self.__video_options['**kwargs'])
        ax.set_xlabel(self.__video_options['x_label'])

        if self.__video_options['dateformat'] == None:
            ax.set_title(
                f"{self.__video_options['x_title']} From {pd.to_datetime(min(self.__video_df[self.timeseries_col]))} To {pd.to_datetime(df_date[self.timeseries_col].iloc[0])}")
        else:
            ax.set_title(
                f"{self.__video_options['x_title']} From {pd.to_datetime(min(self.__video_df[self.timeseries_col])).strftime(self.__video_options['dateformat'])} To"
                f" {pd.to_datetime(df_date[self.timeseries_col].iloc[0]).strftime(self.__video_options['dateformat'])}")

        # save fig and reread as np array
        fig.savefig("temp_out.jpg")
        plt.close(fig)
        return cv2.imread("temp_out.jpg")

    def write_extra_frames(self,i,out_writer,img, df_date):



        times = self.__video_options['frames_per_image']

        last = False

        for x in range(times):


            #if first frame write the original data
            if x ==0:
                out_writer.write(img)
                try:
                    df_date1 = self.get_date_df(i + 1)
                    temp_df = df_date.merge(df_date1.set_index(self.category_col)[self.value_col],
                                            left_on=self.category_col,
                                            right_index=True)
                    val_diff = (temp_df[self.value_col + "_y"] - temp_df[self.value_col + "_x"]) / (times - 1)
                except IndexError:
                    last = True

            # if not increment towards the next date evenly
            else:
                if not last:
                    temp_df[self.value_col] = temp_df[self.value_col + "_x"] + (val_diff * x)
                    if self.__video_options["sort"]:
                        temp_df = temp_df.sort_values(self.value_col)
                    img = self.get_chart(temp_df)
                out_writer.write(img)

            #increment loops
            self.__video_options['looptimes'] += 1

            # LOGGING
            self.__write_log(self.__video_options['starttime'],
                             self.__video_options['looptimes'],
                             len(self.__video_options['unique_dates'])*times)

        return out_writer

    def __write_log(self, start, i, unique_dates):

        time_end = datetime.datetime.now()
        total_time = (time_end - start).total_seconds()
        seconds_per = (total_time / (i + 1))
        seconds_left = (unique_dates - (i + 1)) * seconds_per
        print(
            f"\rWriting Frame {i:>4}/{unique_dates}  Render Speed: {(i + 1) / total_time:.1f}fps  Time taken: {convert(total_time)}  Time left: {convert(seconds_left)} {get_bar(i, unique_dates - 1)}",
            end="")

    def __get_temp_df_sort_values(self, dte):
        # filter the df by date and sort if needed

        sort = self.__video_options['sort']

        if sort:
            return self.__video_df[self.__video_df[self.timeseries_col] == dte].sort_values(self.value_col)
        else:
            return self.__video_df[self.__video_df[self.timeseries_col] == dte]

    def __get_time(self,seconds,fps,df):
        number_of_frames = len(df[self.timeseries_col].unique())



