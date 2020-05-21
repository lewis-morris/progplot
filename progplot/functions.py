import imp
import importlib

import PIL
import matplotlib
import numpy as np
import seaborn as sns
import cv2


import inspect



def convert(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%d:%02d:%02d" % (hour, minutes, seconds)


def get_bar(current, maxx, bar_length=30, bar_load="=", bar_blank="-"):
    perc = current / maxx
    bar = int(round(bar_length * perc, 0))
    blank = int(round(bar_length - (bar_length * perc), 0))
    return "[" + bar_load * bar + bar_blank * blank + "]" + f" {round(current / maxx * 100, 2)} % "


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


def get_pallete(as_pil=True):

    #used to draw palette options

    palettes = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges',
                'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
                'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter',
                'cool', 'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic', 'twilight', 'twilight_shifted', 'hsv', 'Pastel1',
                'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c',
                'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
                'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'nipy_spectral', 'gist_ncar']
    total = 0
    for x in range(6):
        for y in range(13):
            img = pad_arr(get_np_pallette(palettes[total]), 3, 0)
            img = write_text(palettes[total], img)
            if y == 0:
                vigm = img.copy()
            else:
                vigm = np.hstack([vigm, img.copy()])
            total += 1
        if x == 0:
            himg = vigm
        else:
            himg = np.vstack([himg, vigm])
    if as_pil:
        return PIL.Image.fromarray(cv2.rotate(himg, cv2.ROTATE_90_COUNTERCLOCKWISE))
    else:
        return cv2.rotate(himg, cv2.ROTATE_90_COUNTERCLOCKWISE)


def get_np_pallette(pal):

    palette = sns.color_palette(pal, n_colors=10)
    for i, x in enumerate(palette):
        base = np.ones([15, 100, 3]).astype(np.uint8)
        base[:, :, 0], base[:, :, 1], base[:, :, 2] = x[0] * 255, x[1] * 255, x[2] * 255

        if i == 0:
            img = base
        else:
            img = np.vstack([img, base])

    return img


def pad_arr(img, width, fill_val):

    base = np.zeros([x + (width * 2) if i <= 1 else x for i, x in enumerate(img.shape)])
    base[width:base.shape[0] - width, width:base.shape[1] - width, :] = img
    return base.astype(np.uint8)


def write_text(text, img):
    img1 = img
    img_rot = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img1 = cv2.putText(img_rot, text, (6, 50), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 6)
    img1 = cv2.putText(img_rot, text, (6, 50), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1)
    img_rot = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
    return img_rot

def tick_transform(x, pos):
        return '${:,.1f}M'.format(x * 1e-6)


def trim_img(img):
    try:
        mask = img[:, :, 3]
        non = np.nonzero(mask)

        w_min = np.min(non[0])
        w_max = np.max(non[0])
        h_min = np.min(non[1])
        h_max = np.max(non[1])

        return img[w_min:w_max, h_min:h_max, :]
    except:
        return img[:,:,::-1]


def gather_image_and_rough_reshape(image_dict, w, h, items, width, keys):
    new_img_dict = {}

    if image_dict == None:
        image_dict = {}

    #add missing keys
    for unique in keys:
        if unique not in image_dict.keys():
            image_dict[unique] = None

    #rough reshape

    try:
        for k, v in image_dict.items():

            img = cv2.imread(v, cv2.IMREAD_UNCHANGED)

            # defualt to error image if not found
            if type(img) != np.ndarray:
                img = cv2.imread("/".join(importlib.util.find_spec("progplot").origin.split("/")[:-1]) + "/error.png", cv2.IMREAD_UNCHANGED)

            img = trim_img(img)

            ratio = img.shape[1] / img.shape[0]
            roughsize = (h / items) * .8

            img = cv2.resize(img, dsize=(int(roughsize * ratio), int(roughsize)), fx=0, fy=0,
                                         interpolation=cv2.INTER_AREA)
            base_img = img

            while base_img.shape[1] < width:
                base_img = np.hstack([base_img, img])

            new_img_dict[k.strip()] = base_img
    except Exception as e:
        print(e)
        return {}

    return new_img_dict


def get_actual_size(pt0, pt1, img):

    ratio = img.shape[1] / img.shape[0]

    height = pt0[1] - pt1[1]
    width = pt1[0] - pt0[0]

    new_img = cv2.resize(img, dsize=(int(height * ratio), int(height)), fx=0, fy=0, interpolation=cv2.INTER_AREA)

    return new_img[:height, :width, :]


def remove_rects_get_pos(ax, fig):

    rect_dict = {}

    w, h = fig.canvas.get_width_height()

    for i, (rect, label) in enumerate(zip(ax.patches, ax.get_yticklabels())):
        rect.set_visible(False)
        rect_dict[label.get_text().strip()] = [(int(rect.get_window_extent().x0), int(h - rect.get_window_extent().y0)),
                                       (int(rect.get_window_extent().x1), int(h - rect.get_window_extent().y1))]
    return ax, fig, rect_dict


def get_bar_appended_chart(cht_img, rect_dict, image_dict, add_rect=False, rect_width=3, rect_colour=(124, 124, 124)):

    for k, v in rect_dict.items():
        # get the image in the shape of the rect from matplotlib- duplicating the image to match width while preserving height

        # check if image in dict
        try:
            single_bar_img = image_dict[k]
        except:
            #used if key not found - will default to error.png
            pass
            single_bar_img = None

        #if type(single_bar_img) != np.ndarray:
        #    single_bar_img = cv2.imread("/".join(importlib.util.find_spec("progplot").origin.split("/")[:-1]) + "/error.png", cv2.IMREAD_UNCHANGED)
        #    single_bar_img = trim_img(single_bar_img)

        bar_img = get_actual_size(v[0], v[1], single_bar_img)

        # if jpg overlay image.
        if bar_img.shape[2] == 3:
            try:
                cht_img = cht_img.copy()
                cht_img[v[1][1]:v[0][1], v[0][0]:v[1][0], :] = bar_img
            except ValueError as e:
                cht_img = cht_img.copy()
                cht_img[v[1][1]:v[0][1], v[0][0]:v[1][0], :] = bar_img
        # if png

        else:
            # set alpha
            h, w, _ = cht_img.shape
            alpha = np.zeros((h, w))

            alpha = try_merge(alpha, bar_img[:, :, 3], v)

            #alpha[v[1][1]:v[0][1], v[0][0]:v[1][0]] = bar_img[:, :, 3]
            alpha = alpha.astype(float) / 255
            alpha = np.stack((alpha,) * 3, axis=-1)

            # set foreground
            foreground = np.zeros(cht_img.shape)

            foreground = try_merge(foreground, bar_img[:, :, :3], v, True)
            #foreground[v[1][1]:v[0][1], v[0][0]:v[1][0], :] = bar_img[:, :, :3]

            foreground = foreground.astype(float)
            foreground = foreground[:, :, ::-1]

            _chart_img = cht_img.astype(float)

            cht_img = (_chart_img * (1 - alpha)) + (foreground * (alpha))

        if add_rect:
            if v[0][0] < cht_img.shape[1]*.4:
                cht_img = cv2.rectangle(cht_img.astype(np.uint8), v[0], v[1], rect_colour, rect_width)

    if cht_img.dtype != np.uint8:
        cht_img = cht_img.astype(np.uint8)

    return cht_img

def try_merge(img1,img,v,double=False):

    if not double:
        try:
            img1[v[1][1]:v[0][1], v[0][0]:v[1][0]] = img
            return img1
        except:
            pass

        try:
            img1[v[1][1]:v[0][1], v[0][0]:v[1][0]-1] = img
            return img1
        except:
            pass

        try:
            img1[v[1][1]:v[0][1], v[0][0]:v[1][0]+1] = img
            return img1
        except:
            pass

        try:
            img1[v[1][1]:v[0][1], v[0][0]:v[1][0]-2] = img
            return img1
        except:
            pass

        try:
            img1[v[1][1]:v[0][1], v[0][0]:v[1][0]+2] = img
            return img1
        except:
            pass
    else:
        try:
            img1[v[1][1]:v[0][1], v[0][0]:v[1][0],:] = img
            return img1
        except:
            pass

        try:
            img1[v[1][1]:v[0][1], v[0][0]:v[1][0]-1,:] = img
            return img1
        except:
            pass

        try:
            img1[v[1][1]:v[0][1], v[0][0]:v[1][0]+1,:] = img
            return img1
        except:
            pass
        try:
            img1[v[1][1]:v[0][1], v[0][0]:v[1][0]-2,:] = img
            return img1
        except:
            pass

        try:
            img1[v[1][1]:v[0][1], v[0][0]:v[1][0]+2,:] = img
            return img1
        except:
            pass
    x=""

def convert_bar(plt, ax, fig, images_dict):
    """
    Used if you want to convert your own bar chart to an image bar.

    ONLY WORKS ON HORIZONTAL BARS CURRENTLY

    :param plt: matplotlib.plyplot object
    :param ax: axes object
    :param fig: figure object
    :param images_dict: (dict) in format k = Y data :  v = image file name  i.e {"test_category", "test.jpg"}
    :return:
    """

    w, h = fig.get_figwidth() * fig.dpi, fig.get_figheight()[1] * fig.dpi

    image_dict = gather_image_and_rough_reshape(images_dict, w, h, ax.get_yticklabels())
    fig.canvas.draw()
    rect_dict = {}
    ax, fig, rect_dict = remove_rects_get_pos(ax, fig)
    can = SubCan(fig)
    chart_img = can.get_arr()[:, :, :3]
    chart_img = get_bar_appended_chart(chart_img, rect_dict, image_dict, True, 2,(30,30,30))
    plt.close(fig)
    return PIL.Image.fromarray(chart_img)



class SubCan(matplotlib.backends.backend_agg.FigureCanvasAgg):

    def __init__(self, fig):
        super().__init__(fig)

    def get_arr(self, *args,
                metadata=None, pil_kwargs=None,
                **kwargs):

        buf, size = self.print_to_buffer()

        return np.frombuffer(buf, np.uint8).reshape((size[1], size[0], 4))