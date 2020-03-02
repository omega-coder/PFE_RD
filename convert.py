from __future__ import division, print_function

import logging
import os
from multiprocessing.pool import Pool
import datetime
import numpy as np
from PIL import Image, ImageFilter

N_PROC = 4

def timestamped_filename(filename, fmt='%m-%d-%y-%H:%M:%S-{filename}'):
    return datetime.datetime.now().strftime(fmt).format(filename=filename)



log_filename = timestamped_filename("converter") 


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M:%S',
    filename="./logs/{}.log".format(log_filename),
    filemode='w')

console = logging.StreamHandler()
console.setLevel(logging.WARNING)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
logging.info("Started logging in file {}.log".format(log_filename))


def convert(fname, cs):
    img = Image.open(fname)
    blurred = img.filter(ImageFilter.BLUR)
    ba = np.array(blurred)
    h, w, _ = ba.shape

    if w > 1.2 * h:
        left_max = ba[:, :w // 32, :].max(axis=(0, 1)).astype(int)
        right_max = ba[:, -w // 32:, :].max(axis=(0, 1)).astype(int)
        max_bg = np.maximum(left_max, right_max)

        foreground = (ba > max_bg + 10).astype(np.uint8)
        bbox = Image.fromarray(foreground).getbbox()

        if bbox is None:
            print('bbox none for {} (???)'.format(fname))
        else:
            left, upper, right, lower = bbox
            if right - left < 0.8 * h or lower - upper < 0.8 * h:
                print('bbox too small for {}'.format(fname))
                bbox = None
    else:
        bbox = None

    if bbox is None:
        bbox = square_bbox(img)

    cropped = img.crop(bbox)
    resized = cropped.resize([cs, cs])
    return resized


def square_bbox(img):
    w, h = img.size
    left = max((w - h) // 2, 0)
    upper = 0
    right = min(w - (w - h) // 2, w)
    lower = h
    return (left, upper, right, lower)


def convert_square(fname, cs):
    img = Image.open(fname)
    bbox = square_bbox(img)
    cropped = img.crop(bbox)
    resized = cropped.resize([cs, cs])
    return resized


def get_convert_fname(fname, extension, directory, convert_directory):
    return fname.replace('jpeg', extension).replace(directory,
                                                    convert_directory)


def process(args):
    fun, arg = args
    directory, convert_directory, fname, cs, extension = arg
    convert_fname = get_convert_fname(fname, extension, directory,
                                      convert_directory)
    if not os.path.exists(convert_fname):
        img = fun(fname, cs)
        save(img, convert_fname)


def save(img, fname):
    logging.info("Saving image ...")
    img.save(fname, quality=90)


def main(directory, convert_directory, cs, extension):

    try:
        os.mkdir(convert_directory)
    except OSError:
        logging.exception("Exception while creating {} directory".format(convert_directory))
        pass

    filenames = [
        os.path.join(dp, f) for dp, dn, fn in os.walk(directory) for f in fn
        if f.endswith('jpeg')
    ]
    filenames = sorted(filenames)

    logging.info("Resizing images in {} to {}."
          "".format(directory, convert_directory))

    n = len(filenames)
    batchsize = 200
    batches = n // batchsize + 1
    pool = Pool(N_PROC)

    args = []

    for f in filenames:
        args.append(
            (convert, (directory, convert_directory, f, cs, extension)))

    for i in range(batches):
        logging.info("batch {:>2} / {}".format(i + 1, batches))
        pool.map(process, args[i * batchsize:(i + 1) * batchsize])

    pool.close()

    logging.info("Finished jobs")


if __name__ == '__main__':
    main(directory='data/train',
         convert_directory='data/train_converted',
         cs=224,
         extension='jpeg')
