import numpy as np
import cv2
from compute_masks import compute_masks
import transforms

def run_net(
    imgs,
    ort_session,
    nclasses=3,
    augment=False,
    tile_overlap=0.1,
    bsize=224,
    return_conv=False,
):
    # make image nchan x Ly x Lx for net
    imgs = np.transpose(imgs, (2, 0, 1))
    detranspose = (1, 2, 0)

    # pad image for net so Ly and Lx are divisible by 4
    imgs, ysub, xsub = transforms.pad_image_ND(imgs)
    # slices from padding
    #         slc = [slice(0, self.nclasses) for n in range(imgs.ndim)] # changed from imgs.shape[n]+1 for first slice size
    slc = [slice(0, imgs.shape[n] + 1) for n in range(imgs.ndim)]
    slc[-3] = slice(0, nclasses + 32 * return_conv + 1)
    slc[-2] = slice(ysub[0], ysub[-1] + 1)
    slc[-1] = slice(xsub[0], xsub[-1] + 1)
    slc = tuple(slc)

    # run network
    # imgs = np.expand_dims(imgs, axis=0)
    # y, style = ort_session.run(None, {'image': imgs})
    # y, style = y[0], style[0]
    y, style = run_tiled(
        imgs,
        ort_session,
        augment=augment,
        bsize=bsize,
        nclasses=nclasses,
        tile_overlap=tile_overlap,
        return_conv=return_conv,
    )

    style /= (style ** 2).sum() ** 0.5
    # slice out padding
    y = y[slc]
    # transpose so channels axis is last again
    y = np.transpose(y, detranspose)
    return y, style


def transforms_resize_image(
    img0, Ly=None, Lx=None, rsz=None, interpolation=cv2.INTER_LINEAR, no_channels=False
):
    """resize image for computing flows / unresize for computing dynamics"""
    if Ly is None and rsz is None:
        error_message = "must give size to resize to or factor to use for resizing"
        print(error_message)
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


def run_tiled(
    imgi,
    ort_session,
    batch_size=1,
    nclasses=3,
    augment=False,
    bsize=224,
    tile_overlap=0.1,
    return_conv=False,
):
    """run network in tiles of size [bsize x bsize]
    First image is split into overlapping tiles of size [bsize x bsize].
    If augment, tiles have 50% overlap and are flipped at overlaps.
    The average of the network output over tiles is returned.
    """
    IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(
        imgi, bsize=bsize, augment=augment, tile_overlap=tile_overlap
    )
    ny, nx, nchan, ly, lx = IMG.shape
    IMG = np.reshape(IMG, (ny * nx, nchan, ly, lx))
    batch_size = batch_size
    niter = int(np.ceil(IMG.shape[0] / batch_size))
    nout = nclasses + 32 * return_conv
    y = np.zeros((IMG.shape[0], nout, ly, lx))
    for k in range(niter):
        irange = np.arange(
            batch_size * k, min(IMG.shape[0], batch_size * k + batch_size)
        )
        y0, style = ort_session.run(None, {"image": IMG[irange]})
        y[irange] = y0.reshape(len(irange), y0.shape[-3], y0.shape[-2], y0.shape[-1])
        if k == 0:
            styles = style[0]
        styles += style.sum(axis=0)
    styles /= IMG.shape[0]
    if augment:
        y = np.reshape(y, (ny, nx, nout, bsize, bsize))
        y = transforms.unaugment_tiles(y, False)
        y = np.reshape(y, (-1, nout, bsize, bsize))

    yf = transforms.average_tiles(y, ysub, xsub, Ly, Lx)
    yf = yf[:, : imgi.shape[1], : imgi.shape[2]]
    styles /= (styles ** 2).sum() ** 0.5
    return yf, styles


def predict(
    x,
    ort_session,
    nclasses=3,
    resample=False,
    normalize=True,
    invert=False,
    rescale=1.0,
    dist_threshold=0.0,
    diam_threshold=12.0,
    flow_threshold=0.4,
    min_size=15,
    interp=False,
    cluster=False,
    do_3D=False,
    skel=False,
    calc_trace=False,
    verbose=False,
):
    shape = x.shape
    nimg = shape[0]
    iterator = range(nimg)
    if resample:
        dP = np.zeros((2, nimg, shape[1], shape[2]), np.float32)
        dist = np.zeros((nimg, shape[1], shape[2]), np.float32)
        bd = np.zeros_like(dist)
    else:
        dP = np.zeros(
            (2, nimg, int(shape[1] * rescale), int(shape[2] * rescale)), np.float32
        )
        dist = np.zeros(
            (nimg, int(shape[1] * rescale), int(shape[2] * rescale)), np.float32
        )
        bd = np.zeros_like(dist)
    for i in iterator:
        img = np.asarray(x[i])
        if normalize or invert:
            img = transforms.normalize_img(img, invert=invert, skel=skel)
        if rescale != 1.0:
            img = transforms_resize_image(img, rsz=rescale)
        yf, style = run_net(img, ort_session)
        print(img.shape, yf.shape, style.shape)
        dist[i] = yf[:, :, 2]
        dP[:, i] = yf[:, :, :2].transpose((2, 0, 1))
        if nclasses == 4:
            bd[i] = yf[:, :, 3]

    niter = 200 if do_3D else (1 / rescale * 200)
    masks = np.zeros((nimg, shape[1], shape[2]), np.uint16)
    p = np.zeros((2, nimg, shape[1], shape[2]) if not resample else dP.shape, np.uint16)
    #                 p = np.zeros(dP.shape, np.uint16)

    tr = [
        []
    ] * nimg  # trace may not work correctly with multiple images currently, still need to test it
    resize = [shape[1], shape[2]] if not resample else None
    for i in iterator:
        masks[i], p[:, i], tr[i] = compute_masks(
            dP[:, i],
            dist[i],
            bd[i],  # pi mismatch
            niter=niter,
            dist_threshold=dist_threshold,
            flow_threshold=flow_threshold,
            diam_threshold=diam_threshold,
            interp=interp,
            cluster=cluster,
            resize=resize,
            skel=skel,
            calc_trace=calc_trace,
            verbose=verbose,
        )
    return masks.squeeze() # dP.squeeze(), dist.squeeze(), p.squeeze(), bd.squeeze()


if __name__ == "__main__":
    import imageio
    import onnxruntime as ort
    model_path = "./cellpose_cyto.onnx"
    img = imageio.imread("https://www.cellpose.org/static/images/img02.png")
    img = img[None, :, :, 1:]
    ort_session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])
    masks = predict(img, ort_session)
    imageio.imwrite("output.png", masks.astype("uint8"))
    print("done!")
