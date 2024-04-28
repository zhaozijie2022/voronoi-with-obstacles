import multiprocessing
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from a_star import get_navi_path
from read_img import read_img, find_nearest_point, line_distance
from tqdm import tqdm

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from a_star import get_navi_path
from tqdm import tqdm
import multiprocessing


def process_pixel(args):
    pixel, data, points, n_points, img, diagonal = args
    known = diagonal  # 以对角线长度作为已知最短距离
    belong = 0
    sides = [[] for _ in range(n_points)]

    for j in range(n_points):
        if line_distance(data[pixel], points[j]) > known:
            continue

        if (data[pixel] == points[j]).all():
            known = 0
            belong = j
            break

        path = get_navi_path(
            obstacle_map=img,
            start=tuple([data[pixel][0], data[pixel][1]]),
            goal=tuple([points[j][0], points[j][1]]),
            known_shortest_distance=known
        )
        if path is not None and len(path) < known:
            known = len(path)
            belong = j

    sides[belong].append(data[pixel])
    return sides


def gen_voronoi_map(n_points, img):
    x = np.arange(0, img.shape[0], 1)
    y = np.arange(0, img.shape[1], 1)
    xx, yy = np.meshgrid(x, y)
    data = np.array([xx.flatten(), yy.flatten()]).T
    n_pixels = len(data)

    x_p = np.linspace(0, img.shape[0], int(np.sqrt(0.5 * n_points)))
    y_p = np.linspace(0, img.shape[1], 2 * int(np.sqrt(0.5 * n_points)))
    xx_p, yy_p = np.meshgrid(x_p, y_p)
    points = np.array([xx_p.flatten(), yy_p.flatten()]).T
    n_points = len(points)

    new_points = points.copy()
    belong = [0] * len(data)

    r = 0
    diagonal = np.linalg.norm([img.shape[0], img.shape[1]])
    while True:
        sides = [[] for _ in range(n_points)]

        pool = multiprocessing.Pool()
        args_list = [(i, data, points, n_points, img, diagonal) for i in range(n_pixels)]

        for result in tqdm(pool.imap_unordered(process_pixel, args_list), total=n_pixels):
            for j, side in enumerate(result):
                sides[j].extend(side)
        pool.close()
        pool.join()

        for j, side in enumerate(sides):
            new_point = find_nearest_point(points=side, obstacle_map=img)
            new_points[j] = new_point

        err = 0
        r += 1

        img_show = img.copy()
        for i, point in enumerate(new_points):
            img_show[int(point[0]), int(point[1])] = 0.5

        plt.imshow(img_show)
        plt.show()

        if r >= 40:
            print("voronoi points can't converge in %d steps" % r)
            return points
        for i in range(n_points):
            err += np.linalg.norm(points[i] - new_points[i])
        if err > 1e-2:
            points = new_points.copy()
        else:
            return points


if __name__ == "__main__":
    img = read_img("./map-downsample-origin.bmp")
    n_points = 2 * 5 ** 2
    points = gen_voronoi_map(n_points, img)
    plt.imshow(img, cmap='gray')