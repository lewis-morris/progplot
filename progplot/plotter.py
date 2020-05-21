import datetime
import PIL
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import warnings
import matplotlib.font_manager as fontman
from PIL import ImageFont
from PIL import ImageDraw
try:
    from IPython.display import HTML
except:
    warnings.warn("You may not be able to display outputs if not using jupyter/ipython")

from matplotlib.ticker import StrMethodFormatter
from progplot.functions import convert, get_bar, gather_image_and_rough_reshape, remove_rects_get_pos, \
    get_bar_appended_chart, SubCan, try_merge
import matplotlib.dates as mdates
from random import shuffle
import os
import matplotlib.patheffects as PathEffects


class _base_writer:

    def __init__(self, verbose=1):
        """
        Init of the bar writer.
        """
        self.category_col = None
        self.timeseries_col = None
        self.value_col = None
        self.category_values = None

        self.agg_type = None
        self._video_options = {}

        self._resample = None
        self._keep_history = None
        self._verbose = verbose

    def set_data(self, data, category_col, timeseries_col, value_col,
                 groupby_agg="sum", resample_agg="sum", output_agg="cumsum", resample=None):
        """
        Used to set the data prior to chart creation.
        At this point a new dataframe is created to suit the settings. Depending on your choices this could take a
        while to complete.
        :param data (pandas Dataframe): input data, must have datetime values, categorical values and numerical values (or other if using count aggregation)
        :param category_col: (str) the name of pandas column that holds the categorical data
        :param timeseries_col: (str) the name of pandas column that holds the timeseries data
        :param value_col: (str) the name of pandas column that holds the value data you want to measure - if aggregation is set to "count" this can be a coloumn containing strings
        :param groupby_agg: (str / None) the aggregation function used to group identical datetimes with. "count", "sum", "mean" or None
        :param resample_agg: (str / None) the aggregation function to used to resample by datetime. "count", "sum", "mean" or None
        :param output_agg: (str / None) how to deal with the final output data. "cumsum" for a cumlative sum over the datetimes, "xrolling" for a rolling window where x is the number of windows i.e "4rolling", "6rolling", or None.
        :param resample: (str) pandas resample arg - resampling is necessary to normalize the dates. Use options (y,m,d,h,s,m,ms etc) i.e "2d" for sampling every 2 days, "1y" for every year. "6m" for six months etc.
        :return: None
        """

        if self._verbose == 1:
            print("Creating resampled video dataframe (aggregating/ resampling). This may take a moment.")

        assert type(data) == pd.DataFrame, "Input data must be a data frame"
        self.df = data

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

        assert (type(groupby_agg) == str and groupby_agg in ["count", "mean", "sum"]) or groupby_agg == None, \
            'groupby_agg is not str or not in ["count","mean","sum"]'

        self.groupby_agg = groupby_agg

        assert (type(resample_agg) == str and resample_agg.lower() in ["count", "mean", "sum"]) or resample_agg == None, \
            'resample_agg is not str or not in ["count","mean","sum"]'

        self.resample_agg = resample_agg

        assert (type(output_agg) == str and (output_agg in [
            "cumsum"] or "rolling" in output_agg.lower())) or output_agg == None, 'output_agg is not None or not in ["cumsum"] or like "4rolling"]'

        self.resample_agg = resample_agg

        self.output_agg = output_agg

        self._resample = self._check_resample_valid(resample)

        self.category_values = list(self.df.groupby([self.category_col]).count().reset_index().
                                    sort_values([self.value_col, self.category_col])[self.category_col])

        # prepare df for redndering

        self._video_df_base = self._create_new_frame()

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

        self._max_date = pd.to_datetime(self.df[self.timeseries_col].max())
        self._min_date = pd.to_datetime(self.df[self.timeseries_col].min())
        self._date_diff = self._data_df_temp[self.timeseries_col].drop_duplicates().diff().median()

        # get dates
        dt_index = pd.DatetimeIndex(pd.date_range(self._min_date, self._max_date, freq=self._resample))

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
        elif self.output_agg == None or self.output_agg.lower() == "None":
            pass
        elif self.output_agg.find("rolling") >= 0:
            _, num = self._check_resample_valid(self.output_agg)
            df = df.rolling(window=num).mean()
            df.dropna(inplace=True)
        else:
            pass

        df[self.category_col] = cat

        return df

    def _get_time_delta(self):

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

    def set_chart_options(self, use_top_x=None, display_top_x=None, x_tick_format=None, y_tick_format=None,
                          dateformat=None, y_label=True, y_label_font_size=None, x_label=True, x_label_font_size=None,
                          title=None, title_font_size=None, use_data_labels=None, figsize=(14, 8),
                          dpi=100, border_size=None, border_colour=(0, 0, 0), palette="magma", palette_keep=True,
                          palette_random=True, tight_layout=True, sort=True, seaborn_style="whitegrid",
                          seaborn_context="paper", font_scale=1.1, convert_bar_to_image=False, image_dict=None):
        """
        Used to set chart options - to be called before video creation to test the output of the chart.
        Two important values are use_top_x and display_top_x:
        use_top_x trims the dataframe down to the top x values.
        display_top_x shows only x categories on the chart.
        You could then therefor keep 20 categories, only showing 10. The effect? The lowest values contend for there
        position and some may disappear off the end of the chart and be replaced with new values IF their value is higher.
        :param use_top_x: (int) Amount of categories to keep when generating chart. None uses all data.
               (the amount of data is sometimes to much to make visually appealing charts so trimming the data down
               is beneficial)
        :param display_top_x: (int) Amount of categories to display on chart. Differs from use_top_x - you can set
               use_top_x to 20 and display_top_x to 10. Then some categories *MIGHT* fall off the bottom of the
               chart and be replaced with a previously unseen value that was not "Displayed" before but was present
               in the data.
        :param x_tick_format: (str) string formatting i.e "£{:.2f}" £2.25 or "%Y-%d" date time like formatting
                              None = no formatting
        :param y_tick_format: (str) string formatting i.e "£{:.2f}" £2.25 or "%Y-%d" date time like formatting
                              None = no formatting
        :param dateformat: (str) formatting for title datetime display. in strftime format.
        :param figsize: (tuple) matplotlib figsize
        :param dpi: (int) matplotlib dpi value
        :param border_size: (int/None) None = no border Int = size of border around each bar
        :param border_colour: (tuple) (r,g,b) colour or bar - not applicable if border_size == None
        :param y_label: (str/ bool) Label for y Axis or Bool True = column name from DataFrame False = None
        :param y_label_font_size: (int) font size for y_label (None = style default)
        :param x_label: (str/ bool)  Label for x Axis or Bool True = column name from DataFrame False = None
        :param x_label_font_size: (int) font size for x_label (None = style default)
        :param title: (str) Label for title that will prefix the default title "from MAXDATETIME to CURRENTDATETIME"
        :param title_font_size: (int) font size for title (None = style default)
        :param use_data_labels: (str/ None) No datalabels if None - "base" datalabels at base of bar "end" for at end
        :param palette: (str) matplotlib style palette name
        :param palette_keep: (bool) If True chart colours are pinned to categories False colours are pinned to positions
               on the chart.
        :param palette_random: (bool) When the palette colours are created by default they are randomised and assigned
               to each categorical value. This is because depending on the palette type and amount of data is can
               sometimes be hard to determine the movement category (moving up and down the chart) when sort == True.
               Randomising colours can help visualise the movement somewhat.
        :param tight_layout: (bool) use tight layout on plot to make sure all text fits. Sometimes causes chart to
               move when animated.
        :param sort: (bool) True data is sorted and chart positions change. I.e the highest values are reordered to
               the bottom of the chart.
        :param seaborn_style (str) default "whitegrid" - options darkgrid, whitegrid, dark, white, ticks
        :param seaborn_context (str) default "paper  - None, or one of {paper, notebook, talk, poster}
        :param font_scale (float) default 1.1
        :param convert_bar_to_image (bool) if True the bargraph bars will be mapped to images specified in "image_dict"
        :param image_dict (dict) dictionary of images to be mapped to bars k: categorical value v: file path of image
               i.e {"test":"./test.jpg"}
        :return:
        """

        sns.set_style(seaborn_style)
        sns.set_context(seaborn_context, font_scale=font_scale)

        ##trim_dataframe
        if type(use_top_x) == int:
            self._assert_sort(sort)
            max_date = self._video_df_base[self.timeseries_col].tail(1).item()
            self.category_values = \
                self._video_df_base[self._video_df_base[self.timeseries_col] == max_date].sort_values(
                    [self.value_col, self.category_col])[
                    self.category_col].iloc[-use_top_x:]
            self._video_df = self._video_df_base[self._video_df_base[self.category_col].isin(self.category_values)]
        else:
            self._video_df = self._video_df_base.copy()

        max_len = int(self._video_df[self.category_col].str.len().max() * 1.1)
        self._video_df.loc[:][self.category_col] = self._video_df[self.category_col].apply(
            lambda x: x.rjust(max_len) if len(x) < max_len else x)
        # unique values for palette colours

        uniques = list(self._video_df[self.category_col].unique())

        if palette_keep:
            colours = sns.color_palette(palette, n_colors=len(uniques))

            if palette_random:
                shuffle(uniques)
                shuffle(colours)

            palette = dict(zip(uniques, colours))

        w, h = figsize[0] * dpi, figsize[1] * dpi

        self._chart_options = {"x_label_font_size": x_label_font_size,
                               "x_label": x_label,
                               "y_label_font_size": y_label_font_size,
                               "y_label": y_label,
                               "title": title,
                               "figsize": figsize,
                               "dpi": dpi,
                               "x_tick_format": x_tick_format,
                               "y_tick_format": y_tick_format,
                               "palette": palette,
                               "dateformat": dateformat,
                               "tight_layout": tight_layout,
                               "display_top_x": display_top_x,
                               "use_top_x": use_top_x,
                               "sort": sort,
                               "border_size": border_size,
                               "border_colour": border_colour,
                               "title_font_size": title_font_size,
                               "use_data_labels": use_data_labels,
                               "convert_bar_to_image": convert_bar_to_image,
                               "image_dict": gather_image_and_rough_reshape(image_dict, w, h,
                                                                            display_top_x,
                                                                            figsize[0]*dpi,
                                                                            [uni.strip() for uni in uniques]) if convert_bar_to_image == True else False
                               }

    def _set_chart_axis(self, date_df):

        # set labels

        # set x label if needed
        if self._chart_options['x_label'] == False:
            self._ax.xaxis.label.set_visible(False)
        elif self._chart_options['x_label'] == True:
            pass
        else:
            self._ax.set_ylabel(self._chart_options['x_label'])
        # set font size
        if self._chart_options["x_label_font_size"] != None:
            self._ax.set_xlabel(self._ax.get_xlabel(), fontsize=self._chart_options['x_label_font_size'])

        # set y label if needed
        if self._chart_options['y_label'] == False:
            self._ax.yaxis.label.set_visible(False)
        elif self._chart_options['y_label'] == True:
            pass
        else:
            self._ax.set_ylabel(self._chart_options['y_label'])

        # set font size
        if self._chart_options["y_label_font_size"] != None:
            self._ax.set_ylabel(self._ax.get_ylabel(), fontsize=self._chart_options['y_label_font_size'])

        # set chart title
        if self._chart_options['dateformat'] == None:

           self._ax.set_title(
                f"{self._chart_options['title']} From {pd.to_datetime(min(self._video_df[self.timeseries_col]))} To {pd.to_datetime(date_df[self.timeseries_col].iloc[0])}")
        else:
            self._ax.set_title(
                f"{self._chart_options['title']} From {pd.to_datetime(min(self._video_df[self.timeseries_col])).strftime(self._chart_options['dateformat'])} To"
                f" {pd.to_datetime(date_df[self.timeseries_col].iloc[0]).strftime(self._chart_options['dateformat'])}")

        # set fontsize of title
        if self._chart_options['title_font_size'] != None:
            self._ax.set_title(self._ax.get_title(), fontsize=self._chart_options['title_font_size'])

        # check and set x_tick_format
        fmtX = None

        if self._chart_options["x_tick_format"] != None and self._chart_options["x_tick_format"].find("%") >= 0:
            fmtX = mdates.DateFormatter(self._chart_options["x_tick_format"])
        elif self._chart_options["x_tick_format"] != None:
            formatting = self._add_x_to_formatting_for_matplotlib(self._chart_options["x_tick_format"])
            fmtX = StrMethodFormatter(formatting)

        if fmtX != None:
            self._ax.xaxis.set_major_formatter(fmtX)

        # check and set y_tick_format
        fmtY = None

        if self._chart_options["y_tick_format"] != None and self._chart_options["y_tick_format"].find("%") >= 0:
            fmtY = mdates.DateFormatter(self._chart_options["y_tick_format"])
        elif self._chart_options["y_tick_format"] != None:
            formatting = self._add_x_to_formatting_for_matplotlib(self._chart_options["y_tick_format"])
            fmtY = StrMethodFormatter(formatting)

        if fmtY != None:
            self._ax.yaxis.set_major_formatter(fmtY)

    def _add_x_to_formatting_for_matplotlib(self, text):
        text = self._chart_options["x_tick_format"]
        if text.find("x") == -1:
            pos = text.find("{") + 1
            return text[:pos] + "x" + text[pos:]

    def set_display_settings(self, fps=30, time_in_seconds=None, video_file_name="output.webm", codec="VP90"):
        """
        Used to set the video settings for rendering
        :param fps: (int) expected fps of video
        :param time_in_seconds: (int) rough expected running time of video in seconds if NONE then each datetime is displayed for 1 frame. This sometimes creates very FAST videos if there is limitied data.
        :param video_file_name: (str) desired output file - must be "xxx.webm"
        :param codec: (str) from list "VP80", "VP90", "XVID", "MP4V".  DEFAULT is VP90 and is open source web format for videos. VP80/90 must have ".webm" for file extension / XVID ".avi" MP4V ".mp4"
        :return:
        """
        # save file name
        self._last_video_save = video_file_name

        # get unique dates
        unique_dates = self._video_df_base[self.timeseries_col].unique()

        ## get times per frame.
        if not time_in_seconds == None:
            total_frames = time_in_seconds * fps
            frames_per_image = int(total_frames / len(unique_dates))
        else:
            frames_per_image = 1

        # set options for
        self._video_options = {"unique_dates": unique_dates,
                               "frames_per_image": frames_per_image,
                               "looptimes": 0,
                               "fourcc": codec,
                               "video_file_name": video_file_name,
                               "gif_file_name": None,
                               "fps": fps}

    def test_chart(self, frame_no=None, as_pil=True):
        """
        Use prior to video creation to test output.
        :param frame_no: (int / None) if None a random position on the timeline is selected.
        :param as_pil: (bool) True outputs a PIL Image False outputs a np.array
        :return:
        """
        assert self._chart_options != {}, "Please set chart options first"

        if frame_no == None:
            frame_no = np.random.randint(0, len(self._video_options["unique_dates"]) - 1)

        df_date = self._get_date_df(frame_no)

        img = self._get_chart(df_date)

        if as_pil:
            return PIL.Image.fromarray(img)
        else:
            return img

    def write_video(self, output_html=True, limit_frames=None):

        """
        Renders Video and saves to file - all settings need to be set piror to calling this function.
        :param output_html: For Jupyter - will output the video as HTML
        :param limit_frames: To limit frames to x number for testing i.e 20 will only render the first 20 frames.
        :return:
        """

        assert self._video_options != {}, "Please set video options first"
        assert self._chart_options != {}, "Please set chart options first"

        self._video_options["looptimes"] = 0
        self._video_options['starttime'] = datetime.datetime.now()

        # squeeze if TEST mode

        for i, dte in enumerate(self._video_options["unique_dates"]):

            df_date = self._get_date_df(i)

            img = self._get_chart(df_date)

            if i == 0:
                # set writer
                fourcc = cv2.VideoWriter_fourcc(*self._video_options["fourcc"])
                self._out = cv2.VideoWriter("./" + self._video_options["video_file_name"], fourcc=fourcc,
                                            fps=self._video_options["fps"], frameSize=(img.shape[1], img.shape[0]))

            self._write_extra_frames(i, img, df_date)

            if limit_frames != None:
                if self._video_options['looptimes'] > limit_frames:
                    break

        # finalize file
        self._out.release()

        del self._out

        if self._verbose == 1:
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

        if self._verbose == 1:
            print("Converting video file to gif.")

        file_name = self._video_options["video_file_name"]
        new_file_name = "".join(file_name.split(".")[:-1]) + ".gif"

        try:
            os.remove(new_file_name)
        except:
            pass

        out = os.system(f"ffmpeg -i {self._video_options['video_file_name']} {new_file_name}")

        self._video_options["gif_file_name"] = new_file_name

        if self._verbose == 1 and out == 0:
            print("Video file converted to gif.")
        elif out > 0:
            print("Error Generating GIF. Have you got ffmpeg installed?")

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
                    <img
                    src = {self._video_options["gif_file_name"]}
                    alt = "Gif Output">
                """)

    def _set_tight_layout(self):

        plt.tight_layout()
        self._ax.get_figure().canvas.draw()
        chart_x1 = self._ax.get_window_extent().x1
        ticks = [tick.get_position()[0] for tick in self._ax.get_xticklabels() if
                 tick.get_window_extent().x1 < chart_x1]
        self._ax.set_xticks(ticks)
        plt.tight_layout()

    def _get_numpy(self,trans = False):

        # fig.savefig("temp_out.png")
        # plt.close(fig)
        # return cv2.imread("temp_out.png")[:,:,::-1]

        can = SubCan(self._fig)
        plt.close(self._fig)

        if trans:
            arr = can.get_arr()[:, :, :]
        else:
            arr = can.get_arr()[:, :, :3]

        return arr

    def _write_log(self, start, i, unique_dates):

        time_end = datetime.datetime.now()
        total_time = (time_end - start).total_seconds()
        seconds_per = (total_time / (i + 1))
        seconds_left = (unique_dates - (i + 1)) * seconds_per
        if self._verbose == 1:
            print(f"\rWriting Frame {i:>4}/{unique_dates}  Render Speed: {(i + 1) / total_time:.1f}fps  Time taken:"
                  f" {convert(total_time)}  Time left: {convert(seconds_left)} {get_bar(i, unique_dates - 1)}", end="")

    def _get_temp_df_sort_values(self, dte, all_values=False):
        # filter the df by date and sort if needed

        sort = self._chart_options['sort']

        if sort:
            if all_values:
                return self._video_df[self._video_df[self.timeseries_col] <= dte].sort_values(
                    [self.value_col, self.category_col])
            else:
                return self._video_df[self._video_df[self.timeseries_col] == dte].sort_values(
                    [self.value_col, self.category_col])
        else:
            if all_values:
                return self._video_df[self._video_df[self.timeseries_col] <= dte]
            else:
                return self._video_df[self._video_df[self.timeseries_col] == dte]

    def _get_time(self, seconds, fps, df):
        number_of_frames = len(df[self.timeseries_col].unique())

    def _get_date_df(self, i, vals=False):
        # get dates df SORTED IF NEEDED
        self._assert_sort(self._chart_options["sort"])

        if self._chart_options["sort"]:
            temp_df = self._get_temp_df_sort_values(self._video_options["unique_dates"][i], self._keep_history)
        else:
            temp_df = self._get_temp_df_sort_values(self._video_options["unique_dates"][i], self._keep_history)

        # filter display_top_x value to only SHOW the top x values.
        if type(vals) == pd.core.series.Series:
            self._assert_sort(self._chart_options["sort"])
            temp_df = temp_df[temp_df[self.category_col].isin(vals)]

        elif type(self._chart_options["display_top_x"]) == int:
            self._assert_sort(self._chart_options["sort"])
            temp_df = temp_df.tail(self._chart_options["display_top_x"])

        else:
            pass

        return temp_df

    def _add_text_values(self, data, fontdict={"size": "large", "color": "#F6F6F6"}, formatting=None):

        # if float(label_txt) != 0:
        fonts = {}

        try:
            self._fig.canvas.draw()
        except:
            raise ValueError("Issue with formatting needs to be in the format {:.2f} etc or None")

        if self._chart_options["convert_bar_to_image"]:
            fontdict["color"] = (.965,.965,.965)

        for i, rect in enumerate(self._ax.patches):

            # remove value if data = 0 and start =
            if data[i] != 0 and rect.get_window_extent().x0 == self._ax.get_window_extent().x0:

                label_txt = data[i]

                # format data
                if self._chart_options["x_tick_format"] != None:
                    label_txt = self._chart_options["x_tick_format"].format(label_txt)

                ext = self._ax.get_window_extent()

                extra_perc = 0.005

                extra = self._ax.get_xlim()[1] * extra_perc

                # locations for base
                yloc_middle_bar = rect.get_y() + rect.get_height() / 2
                xloc_begg_bar = rect.get_bbox().x0 + extra
                xloc_end_bar = rect.get_bbox().x1 + extra
                xloc_inside_bar = rect.get_bbox().x1 - extra

                if self._chart_options["use_data_labels"] == "base":
                    txt = self._ax.text(xloc_begg_bar, yloc_middle_bar, label_txt, verticalalignment='center',
                                        horizontalalignment="left",
                                        fontdict=fontdict)

                elif self._chart_options["use_data_labels"] == "end":

                    txt = self._ax.text(xloc_end_bar, yloc_middle_bar, label_txt, verticalalignment='center',
                                        horizontalalignment="left",
                                        fontdict=fontdict)
                    inc = 0.005

                    if txt.get_window_extent().x1 > self._ax.get_window_extent().x1:
                        txt.set_visible(False)
                        txt = self._ax.text(xloc_inside_bar, yloc_middle_bar, label_txt, verticalalignment='center',
                                            horizontalalignment="right", fontdict=fontdict)

                    if txt.get_window_extent().x0 == 0:
                        pass
                else:
                    pass

                # strokes?

                stroke = int(self._fig.get_window_extent().x1 * 0.0015)
                txt.set_path_effects([PathEffects.withStroke(linewidth=stroke * 1,
                                                             foreground=self._chart_options["border_colour"])])

                fonts[i] = txt

        return fonts

class BarWriter(_base_writer):

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self._keep_history = False

    def _get_chart(self, df_date):

        plt.ioff()

        # get plot
        self._fig = plt.figure(figsize=self._chart_options["figsize"], dpi=self._chart_options["dpi"])

        if self._chart_options["border_size"] == None:
            border = False
            self._ax = sns.barplot(y=self.category_col,
                                   x=self.value_col,
                                   data=df_date,
                                   palette=self._chart_options['palette'],
                                   zorder = 1)
        else:
            border = True
            self._ax = sns.barplot(y=self.category_col,
                                   x=self.value_col,
                                   data=df_date,
                                   palette=self._chart_options['palette'],
                                   edgecolor=self._chart_options['border_colour'],
                                   linewidth=self._chart_options["border_size"],
                                   zorder=1
                                   )

        ##squeeze values
        #minn = np.min(df_date[self.value_col]) * .9
        #if minn < 0:
        #    minn = 0
        #maxx = np.max(df_date[self.value_col]) * 1.07#

        #if minn != maxx :
        #    self._ax.set_xlim(minn, maxx)

        #self._ax.autoscale(enable=True, axis="x", tight=True)

        # set line to 0 if no value
        [x.set_linewidth(0) for x in self._ax.get_children() if
         type(x) == matplotlib.patches.Rectangle and x.get_width() == 0]

        self._set_chart_axis(df_date)


        # tight layout 100% recommended
        if self._chart_options["tight_layout"]:
            self._set_tight_layout()

        if self._chart_options["convert_bar_to_image"]:
            self._ax, self._fig, rect_dict = remove_rects_get_pos(self._ax, self._fig)
            nump = self._get_numpy(False)

            new_bars = get_bar_appended_chart(nump, rect_dict, self._chart_options["image_dict"], border,
                                          self._chart_options["border_size"],
                                          self._chart_options['border_colour'])

            ext = self._ax.get_window_extent().transformed(self._fig.dpi_scale_trans.inverted())
            extp0 = ext.p0 * self._chart_options['dpi']
            extp1 = ext.p1 * self._chart_options['dpi']

            new_bars = new_bars[int(self._fig.bbox.extents[3] - extp1[1]):int(self._fig.bbox.extents[3] - extp0[1]),
                   int(extp0[0]):int(extp1[0]), :][:, :, ::-1].copy()

            # TODO TRIED THIS METHOD AND COULD NOT GET IT TO WORK PROPERLY - NEEDS LOOKING AT AS BETTER
            #blank = np.ones((int(self._fig.bbox.height), int(self._fig.bbox.width), 3)) * 255
            #blank[int(self._fig.bbox.extents[3] - extp1[1]):int(self._fig.bbox.extents[3] - extp0[1]),
            #       int(extp0[0]):int(extp1[0]), :] = new_bars

            #new_bars = blank.astype(np.uint8).copy()

        if self._chart_options["use_data_labels"] != None:
            lst = list(df_date[self.value_col])

            fonts = self._add_text_values(lst)

            if self._chart_options["convert_bar_to_image"]:

                nump = nump.copy()

                nump[int(self._fig.bbox.extents[3] - extp1[1]):int(self._fig.bbox.extents[3] - extp0[1]),
                   int(extp0[0]):int(extp1[0]), :] = new_bars

                return self._draw_text(nump, fonts, 2)[:, :, ::-1]

                #TODO TRIED THIS METHOD AND COULD NOT GET IT TO WORK PROPERLY - NEEDS LOOKING AT AS BETTER

                #current_frame = self._get_numpy(False)

                #mask = cv2.cvtColor(cv2.absdiff(nump, current_frame), cv2.COLOR_BGR2GRAY)
                #mask = np.where(mask == 254, 255, mask)
                #mask = np.stack((mask) * 3, axis=-1)

                #mask = np.where(mask >5, 5, 255)/255

                #blank = np.ones((int(self._fig.bbox.height), int(self._fig.bbox.width), 3)) * 255

                #return ((nump1[:,:,::-1] * mask) + (nump[:, :, :3] * (1 - mask))).astype(np.uint8)

        nump = self._get_numpy(True)

        return nump

    def _draw_text(self, img, font_list={}, stroke=None):
        ## used to draw text onto image

        h,w,_ = img.shape

        pil_img = PIL.Image.fromarray(img)


        draw = ImageDraw.Draw(pil_img)

        for k,v in font_list.items():

            font_prob = v.get_font_properties()

            text = v.properties()["prop_tup"][2]
            font_loc = self._ax.transData.transform(v.get_position())
            font_loc[1] = (h - font_loc[1])-(v.get_window_extent().extents[3] -v.get_window_extent().extents[1])/2
            if v.get_ha() == "right":
                font_loc[0] = font_loc[0] - ((v.get_window_extent().extents[2] - v.get_window_extent().extents[0]))

            size = int(v.get_fontsize()*1.15)

            font = fontman.findfont(font_prob.get_family()[0].replace("-", " "))

            font = ImageFont.truetype(font, int(size))

            # draw with stroke

            text_col = (255, 255, 255)
            fill_col = (0, 0, 0)


            draw.text(font_loc, text, text_col, font=font, stroke_width=2, stroke_fill=fill_col, align=v.get_ha())

        return np.array(pil_img)

    def _write_extra_frames(self, i, img, df_date):

        times = self._video_options['frames_per_image']

        last = False

        for x in range(times):

            # if first frame write the original data
            if x == 0:
                self._out.write(img[:, :, ::-1])
                try:
                    df_date1 = self._get_date_df(i + 1, df_date[self.category_col])
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
                    if self._chart_options["sort"]:
                        temp_df = temp_df.sort_values([self.value_col, self.category_col])
                    img = self._get_chart(temp_df)
                self._out.write(img[:, :, ::-1])

            # increment loops
            self._video_options['looptimes'] += 1

            # LOGGING
            self._write_log(self._video_options['starttime'],
                            self._video_options['looptimes'],
                            len(self._video_options['unique_dates']) * times)


class LineWriter(_base_writer):

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self._keep_history = True

    def write_video(self, output_html=True, limit_frames=None):
        """
        Renders Video and saves to file - all settings need to be set piror to calling this function.
        :param output_html: (bool) For Jupyter - True will output the video as HTML after render
        :param limit_frames: To limit frames to x number for testing i.e 20 will only render the first 20 frames.
        """
        assert self._video_options != {}, "Please set video settings first"
        assert limit_frames == None or type(limit_frames) == int, "limit_frames is not None or Int"

        self._video_options["looptimes"] = 0
        self._video_options['starttime'] = datetime.datetime.now()

        # squeeze if TEST mode

        self._get_chart(self._video_df)

        for i, dte in enumerate(self._video_options["unique_dates"]):

            img = self.set_lim_and_save(ax, fig, dte)

            if i == 0:
                # set writer
                fourcc = cv2.VideoWriter_fourcc(*'VP90')

                out = cv2.VideoWriter("./" + self._video_options["video_file_name"], fourcc=fourcc,
                                      fps=self._video_options["fps"], frameSize=(img.shape[1], img.shape[0]))

            out = self._write_extra_frames(i, out, img, self._video_df)

            # check if early stopping
            if limit_frames != None:
                if self._video_options['looptimes'] > limit_frames:
                    break

        # finalize file
        out.release()
        if self._verbose == 1:
            print("\nVideo creation complete")

        # return HTML output
        if output_html:
            return self.show_video()

    def get_chart(self, df_date):

        # get plot
        fig = plt.figure(figsize=self._chart_options["figsize"], dpi=self._chart_options["dpi"])
        ax = sns.lineplot(y=self.value_col,
                          x=self.timeseries_col,
                          data=df_date,
                          hue=self.category_col,
                          palette=self._chart_options['palette'],
                          estimator=None)

        self._fig = fig
        self._ax = ax

        self._set_chart_axis(df_date)

        plt.legend(bbox_to_anchor=(1.0125, 1), loc=2, borderaxespad=0.)

        if self._chart_options["tight_layout"]:
            self._set_tight_layout()

    def test_chart(self, frame_no=None, as_pil=True):
        """
        Use prior to video creation to test output.
        :param frame_no: (int / None) if None a random position on the timeline is selected.
        :param out_type: (bool) True outputs a PIL Image False outputs a np.array
        :return:
        """
        assert self._video_options != {}, "Please set video settings first"

        if frame_no == None:
            frame_no = np.random.randint(0, len(self._video_options["unique_dates"]) - 1)

        self._get_chart(self._video_df)
        dates = self._video_df[self.timeseries_col].unique()

        img = self.set_lim_and_save(dates[frame_no])

        if as_pil:
            return PIL.Image.fromarray(img)
        else:
            return img

    def set_lim_and_save(self, dte_to):

        plt.tight_layout()
        lims = self._ax.get_xlim()
        to_ord = pd.to_datetime(dte_to).toordinal()
        if str(lims[1]).find(".") >= 0 and str(to_ord).find(".") == -1:
            add = float(str(lims[1]).split(".")[-1]) / 10
        else:
            add = 0

        self._video_df
        self._ax.set_xlim(lims[0], to_ord + add)

        df = self._video_df[self._video_df[self.timeseries_col] == dte_to][self.value_col]
        maxx = df.max()
        minn = df.min()
        self._ax.set_ylim(minn - (maxx * .025), maxx * 1.05)
        # save fig and reread as np array
        return self._get_numpy()

    def _write_extra_frames(self, i, out_writer, img, df_date):

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