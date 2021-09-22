import fastremap
from skimage import filters
import numpy as np
import dynamics
import edt
import cv2
from scipy.ndimage import find_objects, label, binary_fill_holes
from scipy.ndimage.morphology import binary_dilation, binary_opening

try:
    from sklearn.cluster import DBSCAN

    SKLEARN_ENABLED = True
except:
    SKLEARN_ENABLED = False

# merged deiameter functions
def utils_diameters(masks, skel=False, dist_threshold=1):
    if not skel:  # original 'equivalent area circle' diameter
        _, counts = np.unique(np.int32(masks), return_counts=True)
        counts = counts[1:]
        md = np.median(counts ** 0.5)
        if np.isnan(md):
            md = 0
        md /= (np.pi ** 0.5) / 2
        return md, counts ** 0.5
    else:  # new distance-field-derived diameter (aggrees with cicle but more general)
        dt = edt.edt(np.int32(masks))
        dt_pos = np.abs(dt[dt >= dist_threshold])
        return utils_dist_to_diam(np.abs(dt_pos)), None


# also used in models.py
def utils_dist_to_diam(dt_pos):
    return 6 * np.mean(dt_pos)


#     return np.exp(3/2)*gmean(dt_pos[dt_pos>=gmean(dt_pos)])


# Should work for 3D too. Could put into usigned integer form at the end...
# Also could use some parallelization
from skimage import measure


# Edited slightly to only remove small holes(under min_size) to avoid filling in voids formed by cells touching themselves
# (Masks show this, outlines somehow do not. Also need to find a way to split self-contact points).
def utils_fill_holes_and_remove_small_masks(
    masks, min_size=15, hole_size=3, scale_factor=1
):
    """fill holes in masks (2D/3D) and discard masks smaller than min_size (2D)

    fill holes in each mask using scipy.ndimage.morphology.binary_fill_holes

    Parameters
    ----------------

    masks: int, 2D or 3D array
        labelled masks, 0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]

    min_size: int (optional, default 15)
        minimum number of pixels per mask, can turn off with -1

    Returns
    ---------------

    masks: int, 2D or 3D array
        masks with holes filled and masks smaller than min_size removed,
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]

    """
    masks = dynamics.utils_format_labels(
        masks
    )  # not sure how this works with 3D... tests pass though

    # my slightly altered version below does not work well with 3D (vs test GT) so I need to test
    # to see if mine is actually better in general or needs to be toggled; for now, commenting out
    # #     min_size *= scale_factor
    #     hole_size *= scale_factor

    #     if masks.ndim > 3 or masks.ndim < 2:
    #         raise ValueError('masks_to_outlines takes 2D or 3D array, not %dD array'%masks.ndim)

    #     slices = find_objects(masks)
    #     j = 0
    #     for i,slc in enumerate(slices):
    #         if slc is not None:
    #             msk = masks[slc] == (i+1)
    #             npix = msk.sum()
    #             if min_size > 0 and npix < min_size:
    #                 masks[slc][msk] = 0
    #             else:
    #                 hsz = np.count_nonzero(msk)*hole_size/100 #turn hole size into percentage
    #                 #eventually the boundary output should be used to properly exclude real holes vs label gaps
    #                 if msk.ndim==3:
    #                     for k in range(msk.shape[0]):
    #                         padmsk = remove_small_holes(np.pad(msk[k],1,mode='constant'),hsz)
    #                         msk[k] = padmsk[1:-1,1:-1]
    #                 else:
    #                     padmsk = remove_small_holes(np.pad(msk,1,mode='constant'),hsz)
    #                     msk = padmsk[1:-1,1:-1]
    #                 masks[slc][msk] = (j+1)
    #                 j+=1
    #     return masks
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError(
            "fill_holes_and_remove_small_masks takes 2D or 3D array, not %dD array"
            % masks.ndim
        )
    slices = find_objects(masks)
    j = 0
    for i, slc in enumerate(slices):
        if slc is not None:
            msk = masks[slc] == (i + 1)
            npix = msk.sum()
            if min_size > 0 and npix < min_size:
                masks[slc][msk] = 0
            else:
                if msk.ndim == 3:
                    for k in range(msk.shape[0]):
                        msk[k] = binary_fill_holes(msk[k])
                else:
                    msk = binary_fill_holes(msk)
                masks[slc][msk] = j + 1
                j += 1
    return masks


# needs to have a wider range to avoid weird effects with few cells in frame
# also turns out previous fomulation can give negative numbers
def transforms_normalize99(img, lower=0.01, upper=99.99, skel=False):
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


def transforms_resize_image(
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


def compute_masks(
    dP,
    dist,
    bd,
    p=None,
    niter=200,
    dist_threshold=0.0,
    diam_threshold=12.0,
    flow_threshold=0.4,
    interp=True,
    cluster=False,
    do_3D=False,
    min_size=15,
    resize=None,
    skel=False,
    calc_trace=False,
    verbose=False,
    nclasses=3,
    gpu=False,
    device=None,
):
    """compute masks using dynamics from dP, dist, and boundary"""
    if skel:
        mask = filters.apply_hysteresis_threshold(
            dist, dist_threshold - 1, dist_threshold
        )  # good for thin features
    else:
        mask = (
            dist > dist_threshold
        )  # analog to original iscell=(cellprob>cellprob_threshold)

    if np.any(mask):  # mask at this point is a cell cluster binary map, not labels
        if not skel:  # use original algorthm
            if verbose:
                print("using original mask reconstruction algorithm")
            if p is None:
                p, inds, tr = dynamics.follow_flows(
                    dP * mask / 5.0,
                    mask=mask,
                    niter=niter,
                    interp=interp,
                    use_gpu=gpu,
                    device=device,
                    skel=skel,
                    calc_trace=calc_trace,
                )

            else:
                inds, tr = [], []
                if verbose:
                    print("p given")
            mask = dynamics.get_masks(
                p,
                iscell=mask,
                flows=dP,
                threshold=flow_threshold if not do_3D else None,
                use_gpu=gpu,
            )

        else:  # use new algorithm
            Ly, Lx = mask.shape
            if nclasses == 4:
                dt = np.abs(dist[mask])  # abs needed if the threshold is negative
                d = utils_dist_to_diam(dt)
                eps = np.std(dt) ** 0.5
                if verbose:
                    print(
                        "number of mask pixels",
                        np.sum(mask),
                        "image shape",
                        mask.shape,
                        "diameter metric is",
                        d,
                        "eps",
                        eps,
                    )

            else:  # backwards compatibility, doesn't help for *clusters* of thin/small cells
                d, e = utils_diameters(mask, skel)
                eps = np.sqrt(2)

            # save unaltered versions for later
            dP = dP.copy()

            # The mean diameter can inform whether or not the cells are too small to form contiguous blobs.
            # My first solution was to upscale everything before Euler integration to give pixels 'room' to
            # stay together. My new solution is much better: use a clustering algorithm on the sub-pixel coordinates
            # to assign labels. It works just as well and is faster because it doesn't require increasing the
            # number of points or taking time to upscale/downscale the data. Users can toggle cluster on manually or
            # by setting the diameter threshold higher than the average diameter of the cells.
            if d <= diam_threshold:
                cluster = True
                if verbose:
                    print("Turning on subpixel clustering for label continuity.")

            dP *= mask
            dx = dP[1].copy()
            dy = dP[0].copy()
            mag = np.sqrt(dP[1, :, :] ** 2 + dP[0, :, :] ** 2)
            #             # renormalize (i.e. only get directions from the network)
            dx[mask] = np.divide(
                dx[mask],
                mag[mask],
                out=np.zeros_like(dx[mask]),
                where=np.logical_and(mag[mask] != 0, ~np.isnan(mag[mask])),
            )
            dy[mask] = np.divide(
                dy[mask],
                mag[mask],
                out=np.zeros_like(dy[mask]),
                where=np.logical_and(mag[mask] != 0, ~np.isnan(mag[mask])),
            )

            # compute the divergence
            Y, X = np.nonzero(mask)
            pad = 1
            Tx = np.zeros((Ly + 2 * pad) * (Lx + 2 * pad), np.float64)
            Tx[Y * Lx + X] = np.reshape(dx.copy(), Ly * Lx)[Y * Lx + X]
            Ty = np.zeros((Ly + 2 * pad) * (Lx + 2 * pad), np.float64)
            Ty[Y * Lx + X] = np.reshape(dy.copy(), Ly * Lx)[Y * Lx + X]

            # Rescaling by the divergence
            div = np.zeros(Ly * Lx, np.float64)
            div[Y * Lx + X] = (
                Ty[(Y + 2) * Lx + X]
                + 8 * Ty[(Y + 1) * Lx + X]
                - 8 * Ty[(Y - 1) * Lx + X]
                - Ty[(Y - 2) * Lx + X]
                + Tx[Y * Lx + X + 2]
                + 8 * Tx[Y * Lx + X + 1]
                - 8 * Tx[Y * Lx + X - 1]
                - Tx[Y * Lx + X - 2]
            )
            div = transforms_normalize99(div, skel=True)
            div.shape = (Ly, Lx)
            # add sigmoid on boundary output to help push pixels away - the final bit needed in some cases!
            # specifically, places where adjacent cell flows are too colinear and therefore had low divergence
            #                 mag = div+1/(1+np.exp(-bd))
            mag = div
            dP[0] = dy * mag
            dP[1] = dx * mag

            p, inds, tr = dynamics.follow_flows(
                dP,
                mask,
                interp=interp,
                use_gpu=gpu,
                device=device,
                skel=skel,
                calc_trace=calc_trace,
            )

            newinds = p[:, inds[:, 0], inds[:, 1]].swapaxes(0, 1)
            mask = np.zeros((p.shape[1], p.shape[2]))

            # the eps parameter needs to be adjustable... maybe a function of the distance
            if cluster and SKLEARN_ENABLED:
                db = DBSCAN(eps=eps, min_samples=3, n_jobs=8).fit(newinds)
                labels = db.labels_
                mask[inds[:, 0], inds[:, 1]] = labels + 1
            else:
                newinds = np.rint(newinds).astype(int)
                skelmask = np.zeros_like(dist, dtype=bool)
                skelmask[newinds[:, 0], newinds[:, 1]] = 1

                # disconnect skeletons at the edge, 5 pixels in
                border_mask = np.zeros(skelmask.shape, dtype=bool)
                border_px = border_mask.copy()
                border_mask = binary_dilation(border_mask, border_value=1, iterations=5)

                border_px[border_mask] = skelmask[border_mask]
                if nclasses == 4:  # can use boundary to erase joined edge skelmasks
                    border_px[bd > -1] = 0
                    if verbose:
                        print("Using boundary output to split edge defects")
                else:  # otherwise do morphological opening to attempt splitting
                    border_px = binary_opening(border_px, border_value=0, iterations=3)

                skelmask[border_mask] = border_px[border_mask]

                LL = label(skelmask, connectivity=1)
                mask[inds[:, 0], inds[:, 1]] = LL[newinds[:, 0], newinds[:, 1]]

        # quality control - this got removed in recent version of cellpose??? or did I add it?
        #             if flow_threshold is not None and flow_threshold > 0 and dP is not None:
        #                 mask = dynamics.remove_bad_flow_masks(mask, dP, threshold=flow_threshold, skel=skel)

        if resize is not None:
            if verbose:
                print("resizing output with resize", resize)
            mask = transforms_resize_image(
                mask, resize[0], resize[1], interpolation=cv2.INTER_NEAREST
            )
            Ly, Lx = mask.shape
            pi = np.zeros([2, Ly, Lx])
            for k in range(2):
                pi[k] = cv2.resize(p[k], (Lx, Ly), interpolation=cv2.INTER_NEAREST)
            p = pi
    else:  # nothing to compute, just make it compatible
        print("No cell pixels found.")
        p = np.zeros([2, 1, 1])
        tr = []
        mask = np.zeros(resize)

    # moving the cleanup to the end helps avoid some bugs arising from scaling...
    # maybe better would be to rescale the min_size and hole_size parameters to do the
    # cleanup at the prediction scale, or switch depending on which one is bigger...
    mask = utils_fill_holes_and_remove_small_masks(mask, min_size=min_size)
    fastremap.renumber(
        mask, in_place=True
    )  # convenient to guarantee non-skipped labels
    return mask, p, tr
