import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
import math
from sklearn.preprocessing import MinMaxScaler


def get_kernel(k_size, mode, from_boundary, scale):

    assert k_size // 2 != 0

    one = torch.ones(1).float()
    kernel = torch.zeros(3, 1, k_size, k_size).float()
    c = k_size // 2

    if mode == 0:
        kernel[:, :, c, c] = one
    elif mode == 1:
        kernel[:, :, from_boundary, from_boundary: k_size - from_boundary] = one
        kernel[:, :, k_size - 1 - from_boundary, from_boundary: k_size - from_boundary] = one

        kernel[:, :, from_boundary: k_size - from_boundary, from_boundary] = one
        kernel[:, :, from_boundary: k_size - from_boundary, k_size - 1 - from_boundary] = one

        norms = math.pow((k_size - 2 * from_boundary), 2)

        if not torch.equal(kernel[0][0][c][c], one):
            norms = norms - 1.0

        kernel = scale * kernel / norms

    return kernel


if __name__ == "__main__":
    img_path = 'test_img.png'

    mode = 1
    from_boundary = 1
    k_size = 5
    norm = 20

    assert Path(img_path).exists()

    img = np.array(Image.open(img_path, mode='r').convert('RGB')).transpose(2, 0, 1)

    imgs = torch.from_numpy(img).float().unsqueeze(0)

    kernel = get_kernel(k_size, mode, from_boundary, norm)

    stride = 1
    padding = (kernel.shape[-1] - stride) // 2

    img_conv = F.conv2d(imgs, kernel, stride=1, padding=padding, groups=3)

    print(torch.equal(img_conv, imgs))
    print(kernel)
    print('Input Image shape:', img.shape)
    print('Conved Image Shape:', img_conv.shape)

    in_img = img_conv[0].numpy()
    if np.max(in_img) > 255.0:
        print('Max: ', np.max(in_img))
        scaler = MinMaxScaler()
        for i in range(in_img.shape[0]):

            in_img[i] = scaler.fit_transform(in_img[i]) * 255.0

    img_conv = Image.fromarray(in_img.transpose(1, 2, 0).astype(np.uint8))

    save_name = str(k_size) + 'k_' + str(mode) + 'm_' + str(from_boundary) + 'b_' + str(norm) + 'n_.png'
    img_conv.save(save_name)
    print('Image Name: ', save_name)
    pass
