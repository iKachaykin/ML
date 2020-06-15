import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def noise(im, p):
    inds_pixels = np.arange(0, im.size, 3)


class Image:

    def __init__(self, fn=None, im=None, resolution=None):
        self._fn = fn
        self._im = np.array(im) if im is not None else None
        self._resolution = tuple(list(resolution)) if resolution is not None else None

    def read_im(self, fn=None):
        if fn is not None:
            self._fn = fn
        self._im = colors.rgb_to_hsv(plt.imread(self._fn))
        self._resolution = self._im.shape
        self._im = self._im.ravel()

    def im_arr(self):
        return self._im.copy()

    def im_mp(self):
        return self._im.reshape(*self._resolution)

    def show_im(self, show_immediately=False, create_figure=False, subplot=None):
        if create_figure:
            plt.figure()
        elif subplot is not None:
            if isinstance(subplot, int):
                plt.subplot(subplot)
            elif isinstance(subplot, tuple):
                plt.subplot(*subplot)
        _im_mp = self.im_mp()
        _im_mp = np.where(_im_mp == -1, 0, _im_mp)
        _im_mp = colors.hsv_to_rgb(_im_mp)
        plt.imshow(_im_mp)
        plt.axis(False)
        if show_immediately:
            plt.show()

    def noise_random(self, p):
        inds_pixels = np.arange(0, self._im.size, self._resolution[-1])
        rand = np.random.rand(inds_pixels.size)
        im_noised = np.zeros_like(self._im)
        for i in range(self._resolution[-1]):
            im_noised[inds_pixels+i] = np.where(rand < p, -1, self._im[inds_pixels+i])
        return Image(im=im_noised, resolution=self._resolution)

    def noise_rect(self, i, j, height, width):
        im_noised = self.im_mp().copy()
        i_left, i_right, j_left, j_right = self.get_borders(i, j, height, width)
        im_noised[i_left:i_right, j_left:j_right, :] = -1
        return Image(im=im_noised.ravel(), resolution=self._resolution)

    def noise_random_rects(self, height, width, num=10):
        im_noised = self.im_mp().copy()
        for _ in range(num):
            i, j = np.random.randint(0, self._resolution[0], 1)[0], np.random.randint(0, self._resolution[1], 1)[0]
            i_left, i_right, j_left, j_right = self.get_borders(i, j, height, width)
            im_noised[i_left:i_right, j_left:j_right, :] = -1
        return Image(im=im_noised.ravel(), resolution=self._resolution)

    def get_patch(self, i, j, h):
        i_left, i_right, j_left, j_right = self.get_borders(i, j, h, h)
        im_patch = self.im_mp()[i_left:i_right, j_left:j_right]
        return Image(im=im_patch.ravel(), resolution=im_patch.shape)

    def get_borders(self, i, j, height, width):
        i_left = np.maximum(i - (height // 2 if height % 2 != 0 else height // 2 - 1), 0)
        i_right = np.minimum(i + height // 2 + 1, self._resolution[0])
        j_left = np.maximum(j - (width // 2 if width % 2 != 0 else width // 2 - 1), 0)
        j_right = np.minimum(j + width // 2 + 1, self._resolution[1])
        return i_left, i_right, j_left, j_right

    def get_center_height_width(self, i_left, i_right, j_left, j_right):
        height = i_right - i_left
        width = j_right - j_left
        i_center = i_left + (height // 2 if height % 2 != 0 else height // 2 - 1)
        j_center = j_left + (width // 2 if width % 2 != 0 else width // 2 - 1)
        if 0 <= i_center < self._resolution[0] and 0 <= j_center < self._resolution[1]:
            return i_center, j_center, height, width
        raise IndexError('Center was out of the image!')

    def get_patches(self, h, step=None):
        if step is None:
            step = h
        patches = []
        for i_left in range(0, self._resolution[0], step):
            for j_left in range(0, self._resolution[1], step):
                i, j, width, height = self.get_center_height_width(i_left, i_left+step, j_left, j_left+step)
                patches.append(self.get_patch(i, j, h))
        return patches

    def get_noised_patches(self, h, step=None):
        npatches = []
        for patch in self.get_patches(h, step):
            if -1 in patch.im_arr():
                npatches.append(patch)
        return npatches

    def get_dict(self, h, step=None):
        dpatches = []
        for patch in self.get_patches(h, step):
            if -1 not in patch.im_arr():
                dpatches.append(patch)
        return dpatches


if __name__ == '__main__':
    img = Image('Husky.png')
    img.read_im()
    img.show_im(create_figure=True)

    img_noised = img.noise_random_rects(10, 10, 10)
    img_noised.show_im(create_figure=True)

    img_patch = img.get_patch(399, 299, 200)
    img_patch.show_im(create_figure=True)

    plt.figure()
    count = 1
    for patch in img.get_patches(200):
        patch.show_im(subplot=(4, 3, count))
        count += 1

    plt.figure()
    count = 1
    for patch in img_noised.get_noised_patches(200):
        patch.show_im(subplot=(4, 3, count))
        count += 1

    plt.figure()
    count = 1
    for patch in img_noised.get_dict(200):
        patch.show_im(subplot=(4, 3, count))
        count += 1

    plt.show()
