import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from skimage.transform import resize
from scipy.ndimage import convolve

import SIFTTransform as sift


### Leung-Malik Filter Bank
def gaussian1d(sigma, mean, x, ord):
    x = np.array(x)
    x_ = x - mean
    var = sigma**2

    # Gaussian Function
    g1 = (1 / np.sqrt(2 * np.pi * var)) * (np.exp((-1 * x_ * x_) / (2 * var)))

    if ord == 0:
        g = g1
        return g
    elif ord == 1:
        g = -g1 * ((x_) / (var))
        return g
    else:
        g = g1 * (((x_ * x_) - var) / (var**2))
        return g


def gaussian2d(sup, scales):
    var = scales * scales
    shape = (sup, sup)
    n, m = [(i - 1) / 2 for i in shape]
    x, y = np.ogrid[-m : m + 1, -n : n + 1]
    g = (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(x * x + y * y) / (2 * var))
    return g


def log2d(sup, scales):
    var = scales * scales
    shape = (sup, sup)
    n, m = [(i - 1) / 2 for i in shape]
    x, y = np.ogrid[-m : m + 1, -n : n + 1]
    g = (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(x * x + y * y) / (2 * var))
    h = g * ((x * x + y * y) - var) / (var**2)
    return h


def normalize(fil):
    mean = np.mean(fil)
    std = np.std(fil)
    return (fil - mean) / std


def makefilter(scale, phasex, phasey, pts, sup):

    gx = gaussian1d(3 * scale, 0, pts[0, ...], phasex)
    gy = gaussian1d(scale, 0, pts[1, ...], phasey)

    image = gx * gy

    image = np.reshape(image, (sup, sup))
    image = normalize(image)
    return image


def makeLMfilters():
    sup = 49  # Support of the largest filter (must be odd)
    scalex = np.sqrt(2) ** np.array([1, 2, 3])  # Sigma_{x} for the oriented filters
    norient = 6
    nrotinv = 12

    nbar = len(scalex) * norient
    nedge = len(scalex) * norient
    nf = nbar + nedge + nrotinv
    F = np.zeros([sup, sup, nf])
    hsup = (sup - 1) / 2

    x = [np.arange(-hsup, hsup + 1)]
    # y = [np.arange(-hsup, hsup + 1)]
    y = [np.arange(hsup, -hsup - 1, -1)]

    [x, y] = np.meshgrid(x, y)

    orgpts = [x.flatten(), y.flatten()]
    orgpts = np.array(orgpts)

    count = 0
    for scale in range(len(scalex)):
        for orient in range(norient):
            angle = (np.pi * orient) / norient
            c = np.cos(angle)
            s = np.sin(angle)
            rotpts = [[c, -s], [s, c]]
            rotpts = np.array(rotpts)
            rotpts = np.dot(rotpts, orgpts)
            F[:, :, count] = makefilter(scalex[scale], 0, 1, rotpts, sup)
            F[:, :, count + nedge] = makefilter(scalex[scale], 0, 2, rotpts, sup)
            count = count + 1

    count = nbar + nedge
    scales = np.sqrt(2) ** np.array([1, 2, 3, 4])

    for i in range(len(scales)):
        F[:, :, count] = gaussian2d(sup, scales[i])
        count = count + 1

    for i in range(len(scales)):
        F[:, :, count] = log2d(sup, scales[i])
        count = count + 1

    for i in range(len(scales)):
        F[:, :, count] = log2d(sup, 3 * scales[i])
        count = count + 1

    return F


def create_ft_map(filter_banks, img, img_shape=(512, 512)):
    img = resize(img, (img_shape[0], img_shape[1]))

    # img = fft(img)

    filtered_imgs = np.zeros([filter_banks.shape[2], img.shape[0], img.shape[1]])

    for i in range(filter_banks.shape[2]):
        # filtered_imgs[i, :, :] = convolve(img, filter_banks[:, :, i]) # Very slow.
        # filtered_imgs[i, :, :] = ifft(
        #     img
        #     * fft(
        #         np.pad(
        #             filter_banks[:, :, i], ((231, 232), (232, 231)), constant_values=0
        #         )
        #     )
        # ) # Suffers from circular convolution problem.
        filtered_imgs[i, :, :] = scipy.signal.fftconvolve(
            img, filter_banks[:, :, i], mode="same"
        )

    return filtered_imgs


def fft(f):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(f)))


def ifft(f):
    return np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(f))))


def visualize_filter_bank(filter_bank):
    fig, axs = plt.subplots(4, 12)
    for i in range(0, 18):
        plt.subplot(4, 12, i + 1)
        plt.axis("off")
        plt.imshow(filter_bank[:, :, i], cmap="gray")
    for i in range(0, 18):
        plt.subplot(4, 12, (i + 18) + 1)
        plt.axis("off")
        plt.imshow(filter_bank[:, :, i + 18], cmap="gray")
    for i in range(0, 12):
        plt.subplot(4, 12, (i + 36) + 1)
        plt.axis("off")
        plt.imshow(filter_bank[:, :, i + 36], cmap="gray")

    plt.tight_layout()
    plt.show()


def visualize_ft_map(ft_map):
    fig, axs = plt.subplots(4, 12)
    for i in range(0, 18):
        plt.subplot(4, 12, i + 1)
        plt.axis("off")
        plt.imshow(ft_map[i, :, :], cmap="gray")
    for i in range(0, 18):
        plt.subplot(4, 12, (i + 18) + 1)
        plt.axis("off")
        plt.imshow(ft_map[i + 18, :, :], cmap="gray")
    for i in range(0, 12):
        plt.subplot(4, 12, (i + 36) + 1)
        plt.axis("off")
        plt.imshow(ft_map[i + 36, :, :], cmap="gray")

    plt.tight_layout()
    plt.show()


def main() -> None:
    # Load the images
    pat_id = "R0002"
    pat_ori = "0"
    IMG_DIR_PATH = "Niftisv2/" + pat_id + "/" + pat_ori
    images_path = sift.load_img_dir(IMG_DIR_PATH, img_type="nifti")
    images = sift.load_pre_post_imgs(images_path)

    OrigpreEVT = images[0]
    OrigpostEVT = images[1]

    # Resize images to fit segmentation shape
    pre_img = resize(OrigpreEVT, (OrigpreEVT.shape[0] // 2, OrigpreEVT.shape[1] // 2))
    post_img = resize(
        OrigpostEVT, (OrigpostEVT.shape[0] // 2, OrigpostEVT.shape[1] // 2)
    )

    LM_filter_banks = makeLMfilters()

    visualize_filter_bank(LM_filter_banks)

    filtered_imgs = create_ft_map(LM_filter_banks, pre_img)

    visualize_ft_map(filtered_imgs)


if __name__ == "__main__":
    main()
