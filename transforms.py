import numpy as np
import warnings
import cv2
import edt
from skimage.filters import gaussian
from scipy.ndimage import median_filter, binary_dilation
import fastremap

import logging

transforms_logger = logging.getLogger(__name__)
transforms_logger.setLevel(logging.DEBUG)


def _taper_mask(ly=224, lx=224, sig=7.5):
    bsize = max(224, max(ly, lx))
    xm = np.arange(bsize)
    xm = np.abs(xm - xm.mean())
    mask = 1 / (1 + np.exp((xm - (bsize / 2 - 20)) / sig))
    mask = mask * mask[:, np.newaxis]
    mask = mask[
        bsize // 2 - ly // 2 : bsize // 2 + ly // 2 + ly % 2,
        bsize // 2 - lx // 2 : bsize // 2 + lx // 2 + lx % 2,
    ]
    return mask


def unaugment_tiles(y, unet=False):
    """reverse test-time augmentations for averaging

    Parameters
    ----------

    y: float32
        array that's ntiles_y x ntiles_x x chan x Ly x Lx where chan = (dY, dX, cell prob)

    unet: bool (optional, False)
        whether or not unet output or cellpose output

    Returns
    -------

    y: float32

    """
    for j in range(y.shape[0]):
        for i in range(y.shape[1]):
            if j % 2 == 0 and i % 2 == 1:
                y[j, i] = y[j, i, :, ::-1, :]
                if not unet:
                    y[j, i, 0] *= -1
            elif j % 2 == 1 and i % 2 == 0:
                y[j, i] = y[j, i, :, :, ::-1]
                if not unet:
                    y[j, i, 1] *= -1
            elif j % 2 == 1 and i % 2 == 1:
                y[j, i] = y[j, i, :, ::-1, ::-1]
                if not unet:
                    y[j, i, 0] *= -1
                    y[j, i, 1] *= -1
    return y


def average_tiles(y, ysub, xsub, Ly, Lx):
    """average results of network over tiles

    Parameters
    -------------

    y: float, [ntiles x nclasses x bsize x bsize]
        output of cellpose network for each tile

    ysub : list
        list of arrays with start and end of tiles in Y of length ntiles

    xsub : list
        list of arrays with start and end of tiles in X of length ntiles

    Ly : int
        size of pre-tiled image in Y (may be larger than original image if
        image size is less than bsize)

    Lx : int
        size of pre-tiled image in X (may be larger than original image if
        image size is less than bsize)

    Returns
    -------------

    yf: float32, [nclasses x Ly x Lx]
        network output averaged over tiles

    """
    Navg = np.zeros((Ly, Lx))
    yf = np.zeros((y.shape[1], Ly, Lx), np.float32)
    # taper edges of tiles
    mask = _taper_mask(ly=y.shape[-2], lx=y.shape[-1])
    for j in range(len(ysub)):
        yf[:, ysub[j][0] : ysub[j][1], xsub[j][0] : xsub[j][1]] += y[j] * mask
        Navg[ysub[j][0] : ysub[j][1], xsub[j][0] : xsub[j][1]] += mask
    yf /= Navg
    return yf


def make_tiles(imgi, bsize=224, augment=False, tile_overlap=0.1):
    """make tiles of image to run at test-time

    if augmented, tiles are flipped and tile_overlap=2.
        * original
        * flipped vertically
        * flipped horizontally
        * flipped vertically and horizontally

    Parameters
    ----------
    imgi : float32
        array that's nchan x Ly x Lx

    bsize : float (optional, default 224)
        size of tiles

    augment : bool (optional, default False)
        flip tiles and set tile_overlap=2.

    tile_overlap: float (optional, default 0.1)
        fraction of overlap of tiles

    Returns
    -------
    IMG : float32
        array that's ntiles x nchan x bsize x bsize

    ysub : list
        list of arrays with start and end of tiles in Y of length ntiles

    xsub : list
        list of arrays with start and end of tiles in X of length ntiles


    """

    nchan, Ly, Lx = imgi.shape
    if augment:
        bsize = np.int32(bsize)
        # pad if image smaller than bsize
        if Ly < bsize:
            imgi = np.concatenate((imgi, np.zeros((nchan, bsize - Ly, Lx))), axis=1)
            Ly = bsize
        if Lx < bsize:
            imgi = np.concatenate((imgi, np.zeros((nchan, Ly, bsize - Lx))), axis=2)
        Ly, Lx = imgi.shape[-2:]
        # tiles overlap by half of tile size
        ny = max(2, int(np.ceil(2.0 * Ly / bsize)))
        nx = max(2, int(np.ceil(2.0 * Lx / bsize)))
        ystart = np.linspace(0, Ly - bsize, ny).astype(int)
        xstart = np.linspace(0, Lx - bsize, nx).astype(int)

        ysub = []
        xsub = []

        # flip tiles so that overlapping segments are processed in rotation
        IMG = np.zeros((len(ystart), len(xstart), nchan, bsize, bsize), np.float32)
        for j in range(len(ystart)):
            for i in range(len(xstart)):
                ysub.append([ystart[j], ystart[j] + bsize])
                xsub.append([xstart[i], xstart[i] + bsize])
                IMG[j, i] = imgi[
                    :, ysub[-1][0] : ysub[-1][1], xsub[-1][0] : xsub[-1][1]
                ]
                # flip tiles to allow for augmentation of overlapping segments
                if j % 2 == 0 and i % 2 == 1:
                    IMG[j, i] = IMG[j, i, :, ::-1, :]
                elif j % 2 == 1 and i % 2 == 0:
                    IMG[j, i] = IMG[j, i, :, :, ::-1]
                elif j % 2 == 1 and i % 2 == 1:
                    IMG[j, i] = IMG[j, i, :, ::-1, ::-1]
    else:
        tile_overlap = min(0.5, max(0.05, tile_overlap))
        bsizeY, bsizeX = min(bsize, Ly), min(bsize, Lx)
        bsizeY = np.int32(bsizeY)
        bsizeX = np.int32(bsizeX)
        # tiles overlap by 10% tile size
        ny = 1 if Ly <= bsize else int(np.ceil((1.0 + 2 * tile_overlap) * Ly / bsize))
        nx = 1 if Lx <= bsize else int(np.ceil((1.0 + 2 * tile_overlap) * Lx / bsize))
        ystart = np.linspace(0, Ly - bsizeY, ny).astype(int)
        xstart = np.linspace(0, Lx - bsizeX, nx).astype(int)

        ysub = []
        xsub = []
        IMG = np.zeros((len(ystart), len(xstart), nchan, bsizeY, bsizeX), np.float32)
        for j in range(len(ystart)):
            for i in range(len(xstart)):
                ysub.append([ystart[j], ystart[j] + bsizeY])
                xsub.append([xstart[i], xstart[i] + bsizeX])
                IMG[j, i] = imgi[
                    :, ysub[-1][0] : ysub[-1][1], xsub[-1][0] : xsub[-1][1]
                ]

    return IMG, ysub, xsub, Ly, Lx


# needs to have a wider range to avoid weird effects with few cells in frame
# also turns out previous fomulation can give negative numbers
def normalize99(img, lower=0.01, upper=99.99, skel=False):
    """normalize image so 0.0 is 0.01st percentile and 1.0 is 99.99th percentile"""
    X = img.copy()
    if skel:
        print("running kevin version of normalize99")
        X = np.interp(X, (np.percentile(X, lower), np.percentile(X, upper)), (0, 1))
    else:
        x01 = np.percentile(X, 1)
        x99 = np.percentile(X, 99)
        X = (X - x01) / (x99 - x01)
    return X


def move_axis(img, m_axis=-1, first=True):
    """move axis m_axis to first or last position"""
    if m_axis == -1:
        m_axis = img.ndim - 1
    m_axis = min(img.ndim - 1, m_axis)
    axes = np.arange(0, img.ndim)
    if first:
        axes[1 : m_axis + 1] = axes[:m_axis]
        axes[0] = m_axis
    else:
        axes[m_axis:-1] = axes[m_axis + 1 :]
        axes[-1] = m_axis
    img = img.transpose(tuple(axes))
    return img


# This was edited to fix a bug where single-channel images of shape (y,x) would be
# transposed to (x,y) if x<y, making the labels no longer correspond to the data.
def move_min_dim(img, force=False):
    """move minimum dimension last as channels if < 10, or force==True"""
    if (
        len(img.shape) > 2
    ):  # only makese sense to do this if channel axis is already present
        min_dim = min(img.shape)
        if min_dim < 10 or force:
            if img.shape[-1] == min_dim:
                channel_axis = -1
            else:
                channel_axis = (img.shape).index(min_dim)
            img = move_axis(img, m_axis=channel_axis, first=False)
    return img


def update_axis(m_axis, to_squeeze, ndim):
    if m_axis == -1:
        m_axis = ndim - 1
    if (to_squeeze == m_axis).sum() == 1:
        m_axis = None
    else:
        inds = np.ones(ndim, bool)
        inds[to_squeeze] = False
        m_axis = np.nonzero(np.arange(0, ndim)[inds] == m_axis)[0]
        if len(m_axis) > 0:
            m_axis = m_axis[0]
        else:
            m_axis = None
    return m_axis


def convert_image(
    x,
    channels,
    channel_axis=None,
    z_axis=None,
    do_3D=False,
    normalize=True,
    invert=False,
    nchan=2,
    skel=False,
):
    """return image with z first, channels last and normalized intensities"""

    # squeeze image, and if channel_axis or z_axis given, transpose image
    if x.ndim > 3:
        to_squeeze = np.array([int(isq) for isq, s in enumerate(x.shape) if s == 1])
        # remove channel axis if number of channels is 1
        if len(to_squeeze) > 0:
            channel_axis = (
                update_axis(channel_axis, to_squeeze, x.ndim)
                if channel_axis is not None
                else channel_axis
            )
            z_axis = (
                update_axis(z_axis, to_squeeze, x.ndim)
                if z_axis is not None
                else z_axis
            )
        x = x.squeeze()

    # put z axis first
    if z_axis is not None and x.ndim > 2:
        x = move_axis(x, m_axis=z_axis, first=True)
        if channel_axis is not None:
            channel_axis += 1
        if x.ndim == 3:
            x = x[..., np.newaxis]

    # put channel axis last
    if channel_axis is not None and x.ndim > 2:
        x = move_axis(x, m_axis=channel_axis, first=False)
    elif x.ndim == 2:
        x = x[:, :, np.newaxis]

    if do_3D:
        if x.ndim < 3:
            transforms_logger.critical("ERROR: cannot process 2D images in 3D mode")
            raise ValueError("ERROR: cannot process 2D images in 3D mode")
        elif x.ndim < 4:
            x = x[..., np.newaxis]

    if channel_axis is None:
        x = move_min_dim(x)

    if x.ndim > 3:
        transforms_logger.info(
            "multi-stack tiff read in as having %d planes %d channels"
            % (x.shape[0], x.shape[-1])
        )

    if channels is not None:
        channels = channels[0] if len(channels) == 1 else channels
        if len(channels) < 2:
            transforms_logger.critical("ERROR: two channels not specified")
            raise ValueError("ERROR: two channels not specified")
        x = reshape(x, channels=channels)

    else:
        # code above put channels last
        if x.shape[-1] > nchan:
            transforms_logger.warning(
                'WARNING: more than %d channels given, use "channels" input for specifying channels - just using first %d channels to run processing'
                % (nchan, nchan)
            )
            x = x[..., :nchan]

        if not do_3D and x.ndim > 3:
            transforms_logger.critical("ERROR: cannot process 4D images in 2D mode")
            raise ValueError("ERROR: cannot process 4D images in 2D mode")

        if x.shape[-1] < nchan:
            x = np.concatenate(
                (x, np.tile(np.zeros_like(x), (1, 1, nchan - 1))), axis=-1
            )

    if normalize or invert:
        x = normalize_img(x, invert=invert, skel=skel)

    return x


def reshape(data, channels=[0, 0], chan_first=False):
    """reshape data using channels

    Parameters
    ----------

    data : numpy array that's (Z x ) Ly x Lx x nchan
        if data.ndim==8 and data.shape[0]<8, assumed to be nchan x Ly x Lx

    channels : list of int of length 2 (optional, default [0,0])
        First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
        Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
        For instance, to train on grayscale images, input [0,0]. To train on images with cells
        in green and nuclei in blue, input [2,3].

    invert : bool
        invert intensities

    Returns
    -------
    data : numpy array that's (Z x ) Ly x Lx x nchan (if chan_first==False)

    """
    data = data.astype(np.float32)
    if data.ndim < 3:
        data = data[:, :, np.newaxis]
    elif data.shape[0] < 8 and data.ndim == 3:
        data = np.transpose(data, (1, 2, 0))

    # use grayscale image
    if data.shape[-1] == 1:
        data = np.concatenate((data, np.zeros_like(data)), axis=-1)
    else:
        if channels[0] == 0:
            data = data.mean(axis=-1, keepdims=True)
            data = np.concatenate((data, np.zeros_like(data)), axis=-1)
        else:
            chanid = [channels[0] - 1]
            if channels[1] > 0:
                chanid.append(channels[1] - 1)
            data = data[..., chanid]
            for i in range(data.shape[-1]):
                if np.ptp(data[..., i]) == 0.0:
                    if i == 0:
                        warnings.warn("chan to seg' has value range of ZERO")
                    else:
                        warnings.warn(
                            "'chan2 (opt)' has value range of ZERO, can instead set chan2 to 0"
                        )
            if data.shape[-1] == 1:
                data = np.concatenate((data, np.zeros_like(data)), axis=-1)
    if chan_first:
        if data.ndim == 4:
            data = np.transpose(data, (3, 0, 1, 2))
        else:
            data = np.transpose(data, (2, 0, 1))
    return data


def normalize_img(img, axis=-1, invert=False, skel=False):
    """normalize each channel of the image so that so that 0.0=1st percentile
    and 1.0=99th percentile of image intensities

    optional inversion

    Parameters
    ------------

    img: ND-array (at least 3 dimensions)

    axis: channel axis to loop over for normalization

    Returns
    ---------------

    img: ND-array, float32
        normalized image of same size

    """
    if img.ndim < 3:
        error_message = "Image needs to have at least 3 dimensions"
        transforms_logger.critical(error_message)
        raise ValueError(error_message)

    img = img.astype(np.float32)
    img = np.moveaxis(img, axis, 0)
    for k in range(img.shape[0]):
        if np.ptp(img[k]) > 0.0:
            img[k] = normalize99(img[k], skel=skel)
            if invert:
                img[k] = -1 * img[k] + 1
    img = np.moveaxis(img, 0, axis)
    return img


def reshape_train_test(
    train_data, train_labels, test_data, test_labels, channels, normalize, skel=False
):
    """check sizes and reshape train and test data for training"""
    nimg = len(train_data)
    # check that arrays are correct size
    if nimg != len(train_labels):
        error_message = "train data and labels not same length"
        transforms_logger.critical(error_message)
        raise ValueError(error_message)
        return
    if train_labels[0].ndim < 2 or train_data[0].ndim < 2:
        error_message = "training data or labels are not at least two-dimensional"
        transforms_logger.critical(error_message)
        raise ValueError(error_message)
        return

    if train_data[0].ndim > 3:
        error_message = (
            "training data is more than three-dimensional (should be 2D or 3D array)"
        )
        transforms_logger.critical(error_message)
        raise ValueError(error_message)
        return

    # check if test_data correct length
    if not (
        test_data is not None
        and test_labels is not None
        and len(test_data) > 0
        and len(test_data) == len(test_labels)
    ):
        test_data = None

    # make data correct shape and normalize it so that 0 and 1 are 1st and 99th percentile of data
    train_data, test_data, run_test = reshape_and_normalize_data(
        train_data,
        test_data=test_data,
        channels=channels,
        normalize=normalize,
        skel=skel,
    )

    if train_data is None:
        error_message = "training data do not all have the same number of channels"
        transforms_logger.critical(error_message)
        raise ValueError(error_message)
        return

    if not run_test:
        transforms_logger.info(
            "NOTE: test data not provided OR labels incorrect OR not same number of channels as train data"
        )
        test_data, test_labels = None, None

    return train_data, train_labels, test_data, test_labels, run_test


def reshape_and_normalize_data(
    train_data, test_data=None, channels=None, normalize=True, skel=False
):
    """inputs converted to correct shapes for *training* and rescaled so that 0.0=1st percentile
    and 1.0=99th percentile of image intensities in each channel

    Parameters
    --------------

    train_data: list of ND-arrays, float
        list of training images of size [Ly x Lx], [nchan x Ly x Lx], or [Ly x Lx x nchan]

    test_data: list of ND-arrays, float (optional, default None)
        list of testing images of size [Ly x Lx], [nchan x Ly x Lx], or [Ly x Lx x nchan]

    channels: list of int of length 2 (optional, default None)
        First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
        Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
        For instance, to train on grayscale images, input [0,0]. To train on images with cells
        in green and nuclei in blue, input [2,3].

    normalize: bool (optional, True)
        normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

    Returns
    -------------

    train_data: list of ND-arrays, float
        list of training images of size [2 x Ly x Lx]

    test_data: list of ND-arrays, float (optional, default None)
        list of testing images of size [2 x Ly x Lx]

    run_test: bool
        whether or not test_data was correct size and is useable during training

    """

    # if training data is less than 2D
    run_test = False
    for test, data in enumerate([train_data, test_data]):
        if data is None:
            return train_data, test_data, run_test
        nimg = len(data)
        for i in range(nimg):
            data[i] = move_min_dim(data[i], force=True)
            if channels is not None:
                data[i] = reshape(data[i], channels=channels, chan_first=True)
            if data[i].ndim < 3:
                data[i] = data[i][np.newaxis, :, :]
            if normalize:
                data[i] = normalize_img(data[i], axis=0, skel=skel)
        nchan = [data[i].shape[0] for i in range(nimg)]
        transforms_logger.info("%s channels = %d" % (["train", "test"][test], nchan[0]))
    run_test = True
    return train_data, test_data, run_test


def resize_image(
    img0, Ly=None, Lx=None, rsz=None, interpolation=cv2.INTER_LINEAR, no_channels=False
):
    """resize image for computing flows / unresize for computing dynamics

    Parameters
    -------------

    img0: ND-array
        image of size [y x x x nchan] or [Lz x y x x x nchan] or [Lz x y x x]

    Ly: int, optional

    Lx: int, optional

    rsz: float, optional
        resize coefficient(s) for image; if Ly is None then rsz is used

    interpolation: cv2 interp method (optional, default cv2.INTER_LINEAR)

    Returns
    --------------

    imgs: ND-array
        image of size [Ly x Lx x nchan] or [Lz x Ly x Lx x nchan]

    """
    if Ly is None and rsz is None:
        error_message = "must give size to resize to or factor to use for resizing"
        transforms_logger.critical(error_message)
        raise ValueError(error_message)

    if Ly is None:
        # determine Ly and Lx using rsz
        if not isinstance(rsz, list) and not isinstance(rsz, np.ndarray):
            rsz = [rsz, rsz]
        if no_channels:
            Ly = int(img0.shape[-2] * rsz[-2])
            Lx = int(img0.shape[-1] * rsz[-1])
        else:
            Ly = int(img0.shape[-3] * rsz[-2])
            Lx = int(img0.shape[-2] * rsz[-1])

    if (img0.ndim > 2 and no_channels) or (img0.ndim == 4 and not no_channels):
        if no_channels:
            imgs = np.zeros((img0.shape[0], Ly, Lx), np.float32)
        else:
            imgs = np.zeros((img0.shape[0], Ly, Lx, img0.shape[-1]), np.float32)
        for i, img in enumerate(img0):
            imgs[i] = cv2.resize(img, (Lx, Ly), interpolation=interpolation)
    else:
        imgs = cv2.resize(img0, (Lx, Ly), interpolation=interpolation)
    return imgs


def pad_image_ND(img0, div=16, extra=1):
    """pad image for test-time so that its dimensions are a multiple of 16 (2D or 3D)

    Parameters
    -------------

    img0: ND-array
        image of size [nchan (x Lz) x Ly x Lx]

    div: int (optional, default 16)

    Returns
    --------------

    I: ND-array
        padded image

    ysub: array, int
        yrange of pixels in I corresponding to img0

    xsub: array, int
        xrange of pixels in I corresponding to img0

    """
    Lpad = int(div * np.ceil(img0.shape[-2] / div) - img0.shape[-2])
    xpad1 = extra * div // 2 + Lpad // 2
    xpad2 = extra * div // 2 + Lpad - Lpad // 2
    Lpad = int(div * np.ceil(img0.shape[-1] / div) - img0.shape[-1])
    ypad1 = extra * div // 2 + Lpad // 2
    ypad2 = extra * div // 2 + Lpad - Lpad // 2

    if img0.ndim > 3:
        pads = np.array([[0, 0], [0, 0], [xpad1, xpad2], [ypad1, ypad2]])
    else:
        pads = np.array([[0, 0], [xpad1, xpad2], [ypad1, ypad2]])

    I = np.pad(img0, pads, mode="constant")

    Ly, Lx = img0.shape[-2:]
    ysub = np.arange(xpad1, xpad1 + Ly)
    xsub = np.arange(ypad1, ypad1 + Lx)
    return I, ysub, xsub


def random_rotate_and_resize(
    X,
    Y=None,
    scale_range=1.0,
    gamma_range=0.5,
    xy=(224, 224),
    do_flip=True,
    rescale=None,
    unet=False,
    inds=None,
    depth=0,
    skel=False,
):
    """augmentation by random rotation and resizing

    X and Y are lists or arrays of length nimg, with dims channels x Ly x Lx (channels optional)

    Parameters
    ----------
    X: LIST of ND-arrays, float
        list of image arrays of size [nchan x Ly x Lx] or [Ly x Lx]

    Y: LIST of ND-arrays, float (optional, default None)
        list of image labels of size [nlabels x Ly x Lx] or [Ly x Lx]. The 1st channel
        of Y is always nearest-neighbor interpolated (assumed to be masks or 0-1 representation).
        If Y.shape[0]==3 and not unet, then the labels are assumed to be [cell probability, Y flow, X flow].
        If unet, second channel is dist_to_bound.

    scale_range: float (optional, default 1.0)
        Range of resizing of images for augmentation. Images are resized by
        (1-scale_range/2) + scale_range * np.random.rand()

    xy: tuple, int (optional, default (224,224))
        size of transformed images to return

    do_flip: bool (optional, default True)
        whether or not to flip images horizontally

    rescale: array, float (optional, default None)
        how much to resize images by before performing augmentations

    unet: bool (optional, default False)

    Returns
    -------
    imgi: ND-array, float
        transformed images in array [nimg x nchan x xy[0] x xy[1]]

    lbl: ND-array, float
        transformed labels in array [nimg x nchan x xy[0] x xy[1]]

    scale: array, float
        amount each image was resized by

    """
    if inds is None:  # only relevant when debugging
        inds = np.arange(nimg)

    # backwards compatibility; completely 'stock', no gamma augmentation or any other extra frills.
    if not skel:
        return original_random_rotate_and_resize(
            X,
            Y=[Y[i][1:] for i in inds],
            scale_range=scale_range,
            xy=xy,
            do_flip=do_flip,
            rescale=rescale,
            unet=unet,
        )

    if depth > 5:
        error_message = "Recusion depth exceeded. Check that your images contain cells."
        transforms_logger.critical(error_message)
        raise ValueError(error_message)
        return

    numpx = xy[0] * xy[1]

    dist_bg = 5  # background distance field is set to -dist_bg
    scale_range = max(
        0, min(2, float(scale_range))
    )  # limit overall range to [0,2] i.e. 1+-1
    nimg = len(X)

    # While in other parts of Cellpose channels are put last by default, here we have chan x Ly x Lx
    if X[0].ndim > 2:
        nchan = X[0].shape[0]
    else:
        nchan = 1
    imgi = np.zeros((nimg, nchan, xy[0], xy[1]), np.float32)

    lbl = []
    if Y is not None:
        for n in range(nimg):
            labels = Y[n].copy()
            if labels.ndim < 3:
                labels = labels[np.newaxis, :, :]
            dist = labels[1]
            dist[dist == 0] = -dist_bg
            if labels.shape[0] < 6:
                bd = 5.0 * (labels[1] == 1)
                bd[bd == 0] = -5.0
                labels = np.concatenate(
                    (labels, bd[np.newaxis, :])
                )  # add a boundary layer
            if labels.shape[0] < 7:
                mask = labels[0] > 0
                labels = np.concatenate(
                    (labels, mask[np.newaxis, :])
                )  # add a mask layer
            Y[n] = labels

        if Y[0].ndim > 2:
            nt = Y[0].shape[0] + 1  # (added one for weight array)
        else:
            nt = 1
        lbl = np.zeros((nimg, nt, xy[0], xy[1]), np.float32)

    scale = np.zeros((nimg, 2), np.float32)
    for n in range(nimg):
        img = X[n].copy()
        if Y is not None:
            labels = Y[n].copy()
            # We want the scale distibution to have a mean of 1
            # There may be a better way to skew the distribution to
            # interpolate the parameter space without skewing the mean
            ds = scale_range / 2
            scale[n, :] = np.random.uniform(low=1 - ds, high=1 + ds, size=2)
            if rescale is not None:
                scale[n, :] *= 1.0 / rescale[n]

        # image dimensions are always the last two in the stack
        Ly, Lx = img.shape[-2:]

        # generate random augmentation parameters
        dg = gamma_range / 2
        flip = np.random.choice([0, 1])
        theta = np.random.rand() * np.pi * 2

        # random translation, take the difference between the scaled dimensions and the crop dimensions
        dxy = np.maximum(
            0, np.array([Lx * scale[n, 1] - xy[1], Ly * scale[n, 0] - xy[0]])
        )
        # multiplies by a pair of random numbers from -.5 to .5 (different for each dimension)
        dxy = (
            np.random.rand(
                2,
            )
            - 0.5
        ) * dxy

        # create affine transform
        cc = np.array([Lx / 2, Ly / 2])
        # xy are the sizes of the cropped image, so this is the center coordinates minus half the difference
        cc1 = cc - np.array([Lx - xy[1], Ly - xy[0]]) / 2 + dxy
        # unit vectors from the center
        pts1 = np.float32([cc, cc + np.array([1, 0]), cc + np.array([0, 1])])
        # transformed unit vectors
        pts2 = np.float32(
            [
                cc1,
                cc1 + scale[n] * np.array([np.cos(theta), np.sin(theta)]),
                cc1
                + scale[n]
                * np.array([np.cos(np.pi / 2 + theta), np.sin(np.pi / 2 + theta)]),
            ]
        )
        M = cv2.getAffineTransform(pts1, pts2)

        if flip and do_flip:
            img = img[..., ::-1]
            if Y is not None:
                labels = labels[..., ::-1]
                if nt > 1 and not unet:
                    labels[3] = -labels[3]

        method = cv2.INTER_LINEAR
        # the mode determines what happens with out of bounds regions. If we recompute the flow, we can
        # reflect all the scalar quantities then take the derivative. If we just rotate the field, then
        # the reflection messes up the directions. For now, we are returning to the default of padding
        # with zeros. In the future, we may only predict a scalar field and can use reflection to fill
        # the entire FoV with data - or we can work out how to properly extend the flow field.
        #         mode = cv2.BORDER_DEFAULT # Does reflection
        mode = 0

        for k in range(nchan):
            I = cv2.warpAffine(img[k], M, (xy[1], xy[0]), borderMode=mode, flags=method)
            gamma = np.random.uniform(
                low=1 - dg, high=1 + dg
            )  # allow different gamma per channel
            imgi[n, k] = I ** gamma

        label_method = cv2.INTER_NEAREST
        if Y is not None:
            for k in [0, 1, 2, 3, 4, 5, 6]:  # was skipping 2 and 3, now not
                if not unet:
                    if k == 0:
                        l = labels[k]
                        lbl[n, k] = cv2.warpAffine(
                            l, M, (xy[1], xy[0]), borderMode=mode, flags=label_method
                        )

                        # check to make sure the region contains at least 10 cell pixels; if not, retry.
                        # far from the most efficient implmentation, but does not appear to increase training time.
                        cellpx = np.sum(lbl[n, 0] > 0)

                        if cellpx < 10 or cellpx == numpx:
                            return random_rotate_and_resize(
                                X,
                                Y=Y,
                                scale_range=scale_range,
                                gamma_range=gamma_range,
                                xy=xy,
                                do_flip=do_flip,
                                rescale=rescale,
                                unet=unet,
                                inds=inds,
                                depth=depth + 1,
                            )

                    else:
                        lbl[n, k] = cv2.warpAffine(
                            labels[k], M, (xy[1], xy[0]), borderMode=mode, flags=method
                        )
                else:
                    if k == 0:
                        lbl[n, k] = cv2.warpAffine(
                            labels[k], M, (xy[1], xy[0]), flags=cv2.INTER_NEAREST
                        )
                    else:
                        lbl[n, k] = cv2.warpAffine(
                            labels[k], M, (xy[1], xy[0]), flags=cv2.INTER_LINEAR
                        )

            # For a while I had the heat distribution carried through to re-compute the flow field, but it turns out that the interpolated field
            # gives better segmentation results. This may be because it reduces the importance of predictions right at skeletons and boundaries,
            # where more atrifacts tend to occur.
            if nt > 1 and not unet:
                v1 = lbl[n, 3].copy()  # x component
                v2 = lbl[n, 2].copy()  # y component
                dy = -v1 * np.sin(-theta) + v2 * np.cos(-theta)
                dx = v1 * np.cos(-theta) + v2 * np.sin(-theta)

                mask = lbl[n, 6]
                l = lbl[n, 0]
                dist = edt.edt(l, parallel=8)
                lbl[n, 5] = dist == 1

                lbl[n, 3] = 5.0 * dx * mask  # factor of 5 is applied here
                lbl[n, 2] = 5.0 * dy * mask

                # taking the derivative again rather than interpolating it, avoids a lot of artifacts
                # at centers and where cells meet, also allows for border reflections effortlessly
                #                 heat = np.exp(lbl[n,4].copy())
                #                 mu = np.stack(np.gradient(heat,edge_order=1))
                #                 mag = (mu**2).sum(axis=0)**0.5
                #                 mu = np.divide(mu, mag, out=np.zeros_like(mu), where=np.logical_and(mag!=0,~np.isnan(mag)))

                dist[dist <= 0] = -dist_bg
                lbl[n, 1] = dist

                bg_edt = edt.edt(
                    mask < 0.5, black_border=True
                )  # last arg gives weight to the border, which seems to always lose
                cutoff = 9
                lbl[n, 7] = (
                    gaussian(1 - np.clip(bg_edt, 0, cutoff) / cutoff, sigma=1) + 0.5
                )

    return (
        imgi,
        lbl,
        np.mean(scale),
    )  # for size training, must output scalar size (need to check this again)


# I have the skel flag here just in case, but it actually does not affect the tests
def normalize_field(mu, skel=True):
    if skel:
        mag = np.sqrt(np.nansum(mu ** 2, axis=0))
        mu = np.divide(
            mu,
            mag,
            out=np.zeros_like(mu),
            where=np.logical_and(mag != 0, ~np.isnan(mag)),
        )
    else:
        mu /= 1e-20 + (mu ** 2).sum(axis=0) ** 0.5
    return mu


def _X2zoom(img, X2=1):
    """zoom in image

    Parameters
    ----------
    img : numpy array that's Ly x Lx

    Returns
    -------
    img : numpy array that's Ly x Lx

    """
    ny, nx = img.shape[:2]
    img = cv2.resize(img, (int(nx * (2 ** X2)), int(ny * (2 ** X2))))
    return img


def _image_resizer(img, resize=512, to_uint8=False):
    """resize image

    Parameters
    ----------
    img : numpy array that's Ly x Lx

    resize : int
        max size of image returned

    to_uint8 : bool
        convert image to uint8

    Returns
    -------
    img : numpy array that's Ly x Lx, Ly,Lx<resize

    """
    ny, nx = img.shape[:2]
    if to_uint8:
        if img.max() <= 255 and img.min() >= 0 and img.max() > 1:
            img = img.astype(np.uint8)
        else:
            img = img.astype(np.float32)
            img -= img.min()
            img /= img.max()
            img *= 255
            img = img.astype(np.uint8)
    if np.array(img.shape).max() > resize:
        if ny > nx:
            nx = int(nx / ny * resize)
            ny = resize
        else:
            ny = int(ny / nx * resize)
            nx = resize
        shape = (nx, ny)
        img = cv2.resize(img, shape)
        img = img.astype(np.uint8)
    return img


def original_random_rotate_and_resize(
    X, Y=None, scale_range=1.0, xy=(224, 224), do_flip=True, rescale=None, unet=False
):
    """augmentation by random rotation and resizing
    X and Y are lists or arrays of length nimg, with dims channels x Ly x Lx (channels optional)
    Parameters
    ----------
    X: LIST of ND-arrays, float
        list of image arrays of size [nchan x Ly x Lx] or [Ly x Lx]
    Y: LIST of ND-arrays, float (optional, default None)
        list of image labels of size [nlabels x Ly x Lx] or [Ly x Lx]. The 1st channel
        of Y is always nearest-neighbor interpolated (assumed to be masks or 0-1 representation).
        If Y.shape[0]==3 and not unet, then the labels are assumed to be [cell probability, Y flow, X flow].
        If unet, second channel is dist_to_bound.
    scale_range: float (optional, default 1.0)
        Range of resizing of images for augmentation. Images are resized by
        (1-scale_range/2) + scale_range * np.random.rand()
    xy: tuple, int (optional, default (224,224))
        size of transformed images to return
    do_flip: bool (optional, default True)
        whether or not to flip images horizontally
    rescale: array, float (optional, default None)
        how much to resize images by before performing augmentations
    unet: bool (optional, default False)
    Returns
    -------
    imgi: ND-array, float
        transformed images in array [nimg x nchan x xy[0] x xy[1]]
    lbl: ND-array, float
        transformed labels in array [nimg x nchan x xy[0] x xy[1]]
    scale: array, float
        amount each image was resized by
    """
    scale_range = max(0, min(2, float(scale_range)))
    nimg = len(X)
    if X[0].ndim > 2:
        nchan = X[0].shape[0]
    else:
        nchan = 1
    imgi = np.zeros((nimg, nchan, xy[0], xy[1]), np.float32)

    lbl = []
    if Y is not None:
        if Y[0].ndim > 2:
            nt = Y[0].shape[0]
        else:
            nt = 1
        lbl = np.zeros((nimg, nt, xy[0], xy[1]), np.float32)

    scale = np.zeros(nimg, np.float32)
    for n in range(nimg):
        Ly, Lx = X[n].shape[-2:]

        # generate random augmentation parameters
        flip = np.random.rand() > 0.5
        theta = np.random.rand() * np.pi * 2
        scale[n] = (1 - scale_range / 2) + scale_range * np.random.rand()
        if rescale is not None:
            scale[n] *= 1.0 / rescale[n]
        dxy = np.maximum(0, np.array([Lx * scale[n] - xy[1], Ly * scale[n] - xy[0]]))
        dxy = (
            np.random.rand(
                2,
            )
            - 0.5
        ) * dxy

        # create affine transform
        cc = np.array([Lx / 2, Ly / 2])
        cc1 = cc - np.array([Lx - xy[1], Ly - xy[0]]) / 2 + dxy
        pts1 = np.float32([cc, cc + np.array([1, 0]), cc + np.array([0, 1])])
        pts2 = np.float32(
            [
                cc1,
                cc1 + scale[n] * np.array([np.cos(theta), np.sin(theta)]),
                cc1
                + scale[n]
                * np.array([np.cos(np.pi / 2 + theta), np.sin(np.pi / 2 + theta)]),
            ]
        )
        M = cv2.getAffineTransform(pts1, pts2)

        img = X[n].copy()
        if Y is not None:
            labels = Y[n].copy()
            if labels.ndim < 3:
                labels = labels[np.newaxis, :, :]

        if flip and do_flip:
            img = img[..., ::-1]
            if Y is not None:
                labels = labels[..., ::-1]
                if nt > 1 and not unet:
                    labels[2] = -labels[2]

        for k in range(nchan):
            I = cv2.warpAffine(img[k], M, (xy[1], xy[0]), flags=cv2.INTER_LINEAR)
            imgi[n, k] = I

        if Y is not None:
            for k in range(nt):
                if k == 0:
                    lbl[n, k] = cv2.warpAffine(
                        labels[k], M, (xy[1], xy[0]), flags=cv2.INTER_NEAREST
                    )
                else:
                    lbl[n, k] = cv2.warpAffine(
                        labels[k], M, (xy[1], xy[0]), flags=cv2.INTER_LINEAR
                    )

            if nt > 1 and not unet:
                v1 = lbl[n, 2].copy()
                v2 = lbl[n, 1].copy()
                lbl[n, 1] = -v1 * np.sin(-theta) + v2 * np.cos(-theta)
                lbl[n, 2] = v1 * np.cos(-theta) + v2 * np.sin(-theta)

    return imgi, lbl, scale
