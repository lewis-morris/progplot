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


class _base_writer:

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
        self._video_options = {}

        self._resample = None
        self._keep_history = None

    def set_data(self, category_col, timeseries_col, value_col,
                 groupby_agg="sum", resample_agg="sum", output_agg="cumsum", resample=None):
        """
        Used to set the data prior to chart creation.

        :param category_col: (str) the name of pandas column that holds the categorical data
        :param timeseries_col: (str) the name of pandas column that holds the timeseries data
        :param value_col: (str) the name of pandas column that holds the value data you want to measure - if aggregation is set to "count" this can be a coloumn containing strings
        :param groupby_agg: (str / None) the aggregation function used to group identical datetimes with. "count", "sum", "mean" or None
        :param resample_agg: (str / None) the aggregation function to used to resample by datetime. "count", "sum", "mean" or None
        :param output_agg: (str / None) how to deal with the final output data. "cumsum" for a cumlative sum over the datetimes, "xrolling" for a rolling window where x is the number of windows i.e "4rolling", "6rolling", or None.
        :param resample: (str) pandas resample arg - resampling is necessary to normalize the dates. Use options (y,m,d,h,s,m,ms etc) i.e "2d" for sampling every 2 days, "1y" for every year. "6m" for six months etc.
        :return: None
        """

        self._video_options = {}

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
        self.groupby_agg = groupby_agg

        assert type(resample_agg) == str and resample_agg in ["count", "mean",
                                                              "sum"], 'resample_agg is not str or not in ["count","mean","sum"]'
        self.resample_agg = resample_agg

        assert type(output_agg) == str and output_agg == "cumsum" or output_agg.find(
            "rolling") >= 0, 'output_agg is not str or not in ["rolling", "cumsum"]'
        self.output_agg = output_agg

        self._resample = self._check_resample_valid(resample)

        self.category_values = self.df.groupby([self.category_col]).count().sort_values(self.value_col).index

    def _check_resample_valid(self, resample):

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

        assert txt in ["d", "m", "h", "s", "w", "y", "rolling"], 'resample type is not valid'

        return (txt, num)

    def _add_missing_dates(self, date_list, df):
        # used to add missing dates to timeseries do categories do not dissapear from chart

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
        temp_df.reset_index(drop=True, inplace=True)
        for x in range(len(temp_df[self.timeseries_col])):
            if temp_df.loc[x, self.timeseries_col] in df[self.timeseries_col]:
                temp_df.drop(x, inplace=True)
        return pd.concat([df, temp_df]).sort_values(self.timeseries_col).reset_index(drop=True)

    def _create_new_frame(self):

        # used to aggegate the data etc

        # trim self.df by categories for speedier execution
        self._data_df_temp = self.df[self.df[self.category_col].isin(self.category_values)][
            [self.category_col, self.timeseries_col, self.value_col]]

        self.max_date = pd.to_datetime(self.df[self.timeseries_col].max())
        self.min_date = pd.to_datetime(self.df[self.timeseries_col].min())
        self.date_diff = self._data_df_temp[self.timeseries_col].drop_duplicates().diff().median()

        # get dates
        dt_index = pd.DatetimeIndex(pd.date_range(self.min_date, self.max_date, freq=self._resample))

        # final_df = pd.DataFrame()

        for i, (df, cat) in enumerate(self._get_unique()):

            df = df[[self.timeseries_col, self.value_col, self.category_col]]

            df = self._add_missing_dates(dt_index, df)

            if self._resample != None:
                temp_df = self._do_agg(df, cat, True)
            else:
                temp_df = self._do_agg(df, cat, False)

            # temp_df = self.cumlative_df_temp.merge(temp_df,left_index=True,right_index=True)
            # temp_df.reindex(self.dt_index.index, fill_value=0)

            if i == 0:
                final_df = temp_df.copy()
            else:
                final_df = pd.concat([final_df, temp_df])

        final_df = final_df.reset_index()
        final_df.columns = [self.timeseries_col, self.value_col, self.category_col]

        return final_df

    def _fix_dates(self, df):
        # relic - not used

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

    def _get_unique(self):
        for cat in self.category_values:
            yield self._data_df_temp[self._data_df_temp[self.category_col] == cat], cat

    def _do_agg(self, df, cat, resample=False):

        # used to aggregate the data if needed.

        df[self.value_col] = df[self.value_col].astype(float)

        if self.groupby_agg == "sum":
            df = df.groupby([self.timeseries_col, self.category_col]).sum().reset_index()
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
            df = df.set_index(df[self.timeseries_col]).resample(self._resample).sum()
        elif self.resample_agg == "mean":
            df[self.value_col] = df[self.value_col].apply(lambda x: np.nan if x == 0 else float(x))
            df = df.set_index(df[self.timeseries_col])[self.value_col].resample(self._resample).mean()
            df = pd.DataFrame({self.value_col: df})
            df = df.fillna(0)

        elif self.resample_agg == "count":
            df = df.set_index(df[self.timeseries_col]).resample(self._resample).count()
        else:
            pass

        if self.output_agg == "cumsum":
            df = pd.DataFrame({"Value": df[self.value_col].cumsum()})

        elif self.output_agg.find("rolling") >= 0:
            _, num = self._check_resample_valid(self.output_agg)
            df = df.rolling(window=num).mean()
            df.dropna(inplace=True)
        else:
            pass

        df[self.category_col] = cat

        return df

    def get_time_delta(self):

        if self._resample[0] == "days":
            return datetime.timedelta(days=self._resample[1])
        elif self._resample[0] == "weeks":
            return datetime.timedelta(weeks=self._resample[1])
        elif self._resample[0] == "months":
            return datetime.timedelta(months=self._resample[1])
        elif self._resample[0] == "years":
            return datetime.timedelta(years=self._resample[1])
        elif self._resample[0] == "minutes":
            return datetime.timedelta(minutes=self._resample[1])
        elif self._resample[0] == "seconds":
            return datetime.timedelta(seconds=self._resample[1])
        elif self._resample[0] == "hours":
            return datetime.timedelta(hours=self._resample[1])

    def _assert_sort(self, sort):
        assert sort == True, "Cant display_top_x or use top_x while sort == False"

    def set_display_settings(self, use_top_x=None, display_top_x=None, palette="magma", palette_keep=True, sort=True,
                             x_label="", x_title="", fps=30, dateformat=None, figsize=(14, 8), dpi=100,
                             time_in_seconds=None, video_file_name="output.webm", **kwargs):
        """

        :param use_top_x: (int) Amount of categories to keep when generating chart. None uses all data. (the amount of data is sometimes to much to make visually appealing charts so trimming the data down is beneficial)
        :param display_top_x: (int) Amount of categories to display on chart. Differs from use_top_x - you can set use_top_x to 20 and display_top_x to 10. Then some categories *MIGHT* fall off the bottom of the chart and be replaced with a previously unseen value that was not "Displayed" before but was present in the data.
        :param palette: (str) matplotlib style palette name
        :param palette_keep: (bool) If True chart colours are pinned to categories False colours are pinned to positions on the chart.
        :param sort: (bool) True data is sorted and chart positions change. I.e the highest values are reordered to the bottom of the chart.
        :param x_label: (str) Label for x Axis
        :param x_title: (str) Label for title that will prefix the default title "from MAXDATETIME to CURRENTDATETIME"
        :param dateformat: (str) formatting for x_title datetime display. in strftime format.
        :param figsize: (tuple) matplotlib figsize
        :param dpi: (int) matplotlib dpi value
        :param fps: (int) expected fps of video
        :param time_in_seconds: (int) rough expected running time of video in seconds if NONE then each datetime is displayed for 1 frame. This sometimes creates very FAST videos if there is limitied data.
        :param video_file_name: (str) desired output file - must be "xxx.webm"
        :param kwargs: matplotlib kwargs
        :return:
        """
        # save file name
        self._last_video_save = video_file_name

        # prepare df for redndering
        print("Creating resampled video dataframe. This may take a moment.")
        self._video_df = self._create_new_frame()

        if type(use_top_x) == int:
            self._assert_sort(sort)
            max_date = self._video_df[self.timeseries_col].tail(1).item()
            self.category_values = \
                self._video_df[self._video_df[self.timeseries_col] == max_date].sort_values(self.value_col)[
                    self.category_col].iloc[-use_top_x:]
            self._video_df = self._video_df[self._video_df[self.category_col].isin(self.category_values)]

        # unique values for palette colours
        if palette_keep:
            uniques = list(self.category_values)
            palette = dict(zip(uniques, sns.color_palette(palette, n_colors=len(uniques))))

        # get unique dates
        unique_dates = self._video_df[self.timeseries_col].unique()

        ## get times per frame.
        if not time_in_seconds == None:
            total_frames = time_in_seconds * fps
            frames_per_image = int(total_frames / len(unique_dates))
        else:
            frames_per_image = 1

        # set options for
        self._video_options = {"unique_dates": unique_dates,
                               "frames_per_image": frames_per_image,
                               "display_top_x": display_top_x,
                               "use_top_x": use_top_x,
                               "palette": palette,
                               "sort": sort,
                               "dateformat": dateformat,
                               "x_label": x_label,
                               "x_title": x_title,
                               "**kwargs": kwargs,
                               "looptimes": 0,
                               "video_file_name": video_file_name,
                               "gif_file_name": None,
                               "figsize": figsize,
                               "dpi": dpi,
                               "fps": fps}

    def test_chart(self, frame_no=None, as_pil=True):
        """
        Use prior to video creation to test output.
        :param frame_no: (int / None) if None a random position on the timeline is selected.
        :param as_pil: (bool) True outputs a PIL Image False outputs a np.array
        :return:
        """
        assert self._video_options != {}, "Please set video settings first"

        if frame_no == None:
            frame_no = np.random.randint(0, len(self._video_options["unique_dates"]) - 1)

        df_date = self._get_date_df(frame_no)

        img = self.get_chart(df_date)

        if as_pil:
            return PIL.Image.fromarray(img)
        else:
            return img

    def write_video(self, output_html=True):
        """
        Renders Video and saves to file - all settings need to be set piror to calling this function.

        :param output_html: For Jupyter - will output the video as HTML

        :return:
        """
        assert self._video_options != {}, "Please set video settings first"

        self._video_options["looptimes"] = 0
        self._video_options['starttime'] = datetime.datetime.now()

        # squeeze if TEST mode

        for i, dte in enumerate(self._video_options["unique_dates"]):

            df_date = self._get_date_df(i)

            img = self.get_chart(df_date)

            if i == 0:
                # set writer
                fourcc = cv2.VideoWriter_fourcc(*'VP80')
                out = cv2.VideoWriter("./" + self._video_options["video_file_name"], fourcc=fourcc,
                                      fps=self._video_options["fps"], frameSize=(img.shape[1], img.shape[0]))

            out = self.write_extra_frames(i, out, img, df_date)

        # finalize file
        out.release()
        print("\nVideo creation complete")

        # return HTML output
        if output_html:
            self.show_video()

    def create_gif(self, show_html=True):
        """
        Converts video file to gif

        :param show_html: (bool) after creation show html ?
        :return:
        """
        assert self._video_options != {}, "Please set video settings first"

        file_name = self._video_options["video_file_name"]
        new_file_name = file_name.split(".")[:-1] + ".gif"
        ff = ffmpy.FFmpeg(
            inputs={file_name: None},
            outputs={new_file_name: None})
        ff.run()

        if show_html:
            self.show_gif()

    def show_video(self):
        """
        Shows video in Jupyter
        :return:
        """
        assert type(self._last_video_save) == str, "Video not rendered"

        return HTML(f"""
                    <video alt="test" controls>
                        <source src="{self._last_video_save}" type="video/mp4">
                    </video>
                """)

    def show_gif(self):
        """
        Shows gif in Jupyter
        :return:
        """
        assert type(self._last_video_save) == str, "Video not rendered"

        return HTML(f"""
                    < img
                    src = self.
                    alt = "this slowpoke moves"
                    width = 250 / >
                """)

    def _write_log(self, start, i, unique_dates):

        time_end = datetime.datetime.now()
        total_time = (time_end - start).total_seconds()
        seconds_per = (total_time / (i + 1))
        seconds_left = (unique_dates - (i + 1)) * seconds_per
        print(
            f"\rWriting Frame {i:>4}/{unique_dates}  Render Speed: {(i + 1) / total_time:.1f}fps  Time taken: {convert(total_time)}  Time left: {convert(seconds_left)} {get_bar(i, unique_dates - 1)}",
            end="")

    def _get_temp_df_sort_values(self, dte, all_values=False):
        # filter the df by date and sort if needed

        sort = self._video_options['sort']

        if sort:
            if all_values:
                return self._video_df[self._video_df[self.timeseries_col] <= dte].sort_values(self.value_col)
            else:
                return self._video_df[self._video_df[self.timeseries_col] == dte].sort_values(self.value_col)
        else:
            if all_values:
                return self._video_df[self._video_df[self.timeseries_col] <= dte]
            else:
                return self._video_df[self._video_df[self.timeseries_col] == dte]

    def _get_time(self, seconds, fps, df):
        number_of_frames = len(df[self.timeseries_col].unique())

    def _get_date_df(self, i):
        # get dates df SORTED IF NEEDED
        self._assert_sort(self._video_options["sort"])

        if self._video_options["sort"]:
            temp_df = self._get_temp_df_sort_values(self._video_options["unique_dates"][i], self._keep_history)
        else:
            temp_df = self._get_temp_df_sort_values(self._video_options["unique_dates"][i], self._keep_history)

        # filter display_top_x value to only SHOW the top x values.
        if type(self._video_options["display_top_x"]) == int:
            self._assert_sort(self._video_options["sort"])
            temp_df = temp_df.tail(self._video_options["display_top_x"])

        return temp_df


class BarWriter(_base_writer):

    def __init__(self, df):
        super().__init__(df)
        self._keep_history = False

    def get_chart(self, df_date):

        # get plot
        fig = plt.figure(figsize=self._video_options["figsize"], dpi=self._video_options["dpi"])
        ax = sns.barplot(y=self.category_col,
                         x=self.value_col,
                         data=df_date,
                         palette=self._video_options['palette'],
                         **self._video_options['**kwargs'])
        ax.set_xlabel(self._video_options['x_label'])

        if self._video_options['dateformat'] == None:
            ax.set_title(
                f"{self._video_options['x_title']} From {pd.to_datetime(min(self._video_df[self.timeseries_col]))} To {pd.to_datetime(df_date[self.timeseries_col].iloc[0])}")
        else:
            ax.set_title(
                f"{self._video_options['x_title']} From {pd.to_datetime(min(self._video_df[self.timeseries_col])).strftime(self._video_options['dateformat'])} To"
                f" {pd.to_datetime(df_date[self.timeseries_col].iloc[0]).strftime(self._video_options['dateformat'])}")

        plt.tight_layout()

        # save fig and reread as np array
        fig.savefig("temp_out.jpg")
        plt.close(fig)
        return cv2.imread("temp_out.jpg")

    def write_extra_frames(self, i, out_writer, img, df_date):

        times = self._video_options['frames_per_image']

        last = False

        for x in range(times):

            # if first frame write the original data
            if x == 0:
                out_writer.write(img)
                try:
                    df_date1 = self._get_date_df(i + 1)
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
                    if self._video_options["sort"]:
                        temp_df = temp_df.sort_values(self.value_col)
                    img = self.get_chart(temp_df)
                out_writer.write(img)

            # increment loops
            self._video_options['looptimes'] += 1

            # LOGGING
            self._write_log(self._video_options['starttime'],
                            self._video_options['looptimes'],
                            len(self._video_options['unique_dates']) * times)

        return out_writer

class LineWriter(_base_writer):

    def __init__(self, df):
        super().__init__(df)
        self._keep_history = True

    def write_video(self, output_html=True):
        """
        Renders Video and saves to file - all settings need to be set piror to calling this function.

        :param output_html: For Jupyter - will output the video as HTML

        :return:
        """
        assert self._video_options != {}, "Please set video settings first"

        self._video_options["looptimes"] = 0
        self._video_options['starttime'] = datetime.datetime.now()

        # squeeze if TEST mode

        df_date = self._video_df
        fig,ax = self.get_chart(self._video_df)

        for i, dte in enumerate(self._video_options["unique_dates"]):

            img = self.set_lim_and_save(ax,fig,dte)

            if i == 0:
                # set writer
                fourcc = cv2.VideoWriter_fourcc(*'VP80')
                out = cv2.VideoWriter("./" + self._video_options["video_file_name"], fourcc=fourcc,
                                      fps=self._video_options["fps"], frameSize=(img.shape[1], img.shape[0]))

            out = self.write_extra_frames(i, out, img, df_date)

        # finalize file
        out.release()
        print("\nVideo creation complete")

        # return HTML output
        if output_html:
            self.show_video()


    def get_chart(self, df_date):

        # get plot
        fig = plt.figure(figsize=self._video_options["figsize"], dpi=self._video_options["dpi"])
        ax = sns.lineplot(y=self.value_col,
                         x=self.timeseries_col,
                         data=df_date,
                         hue= self.category_col,
                         palette=self._video_options['palette'],
                         estimator=None,
                         **self._video_options['**kwargs'])

        ax.set_xlabel(self._video_options['x_label'])

        if self._video_options['dateformat'] == None:
            ax.set_title(
                f"{self._video_options['x_title']} From {pd.to_datetime(min(self._video_df[self.timeseries_col]))} To {pd.to_datetime(df_date[self.timeseries_col].iloc[0])}")
        else:
            ax.set_title(
                f"{self._video_options['x_title']} From {pd.to_datetime(min(self._video_df[self.timeseries_col])).strftime(self._video_options['dateformat'])} To"
                f" {pd.to_datetime(df_date[self.timeseries_col].iloc[0]).strftime(self._video_options['dateformat'])}")

        plt.legend(bbox_to_anchor=(1.0125, 1), loc=2, borderaxespad=0.)
        return fig,ax

    def test_chart(self, frame_no=None, as_pil=True):
        """
        Use prior to video creation to test output.
        :param frame_no: (int / None) if None a random position on the timeline is selected.
        :param as_pil: (bool) True outputs a PIL Image False outputs a np.array
        :return:
        """
        assert self._video_options != {}, "Please set video settings first"

        if frame_no == None:
            frame_no = np.random.randint(0, len(self._video_options["unique_dates"]) - 1)

        fig,ax  = self.get_chart(self._video_df)
        dates = self._video_df[self.timeseries_col].unique()

        img = self.set_lim_and_save(ax,fig,dates[frame_no])

        if as_pil:
            return PIL.Image.fromarray(img)
        else:
            return img

    def set_lim_and_save(self, ax, fig, dte_to):

        plt.tight_layout()
        lims = ax.get_xlim()
        to_ord = pd.to_datetime(dte_to).toordinal()
        if str(lims[1]).find(".") >= 0 and str(to_ord).find(".") == -1:
            add = float(str(lims[1]).split(".")[-1])/10
        else:
            add = 0

        self._video_df
        ax.set_xlim(lims[0],to_ord + add )

        df = self._video_df[self._video_df[self.timeseries_col] == dte_to][self.value_col]
        maxx = df.max()
        minn = df.min()
        ax.set_ylim(minn-(maxx*.025), maxx*1.05)
        # save fig and reread as np array
        fig.savefig("temp_out.jpg")
        plt.close(fig)
        return cv2.imread("temp_out.jpg")

    def write_extra_frames(self, i, out_writer, img, df_date):

        times = self._video_options['frames_per_image']

        last = False

        for x in range(times):

            # if first frame write the original data

            out_writer.write(img)

            # increment loops
            self._video_options['looptimes'] += 1

            # LOGGING
            self._write_log(self._video_options['starttime'],
                            self._video_options['looptimes'],
                            len(self._video_options['unique_dates']) * times)

        return out_writer