import PIL
import numpy as np
import seaborn as sns
import cv2

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



def create_gif():
    import ffmpy
    ff = ffmpy.FFmpeg(
        inputs={"output.webm": None},
        outputs={"cash.gif": None})
    ff.run()


def get_pallete(as_pil=True):
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
        base = np.ones([12, 80, 3]).astype(np.uint8)
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