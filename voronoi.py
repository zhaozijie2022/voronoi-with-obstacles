import os
import copy
import multiprocessing
import pickle as pkl

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from navigation import get_navi_path, line_distance
from read_map import read_map


def init_voronoi(n_points, map):
    n_pixels = map.shape[0] * map.shape[1]
    x = np.arange(0, map.shape[0], 1)
    y = np.arange(0, map.shape[1], 1)
    xx, yy = np.meshgrid(x, y)
    pixels = np.array([xx.flatten(), yy.flatten()]).T

    x_p = np.linspace(5, map.shape[0] - 5, int(np.sqrt(0.5 * n_points)), dtype=np.int32)
    y_p = np.linspace(5, map.shape[1] - 5, 2 * int(np.sqrt(0.5 * n_points)), dtype=np.int32)
    xx_p, yy_p = np.meshgrid(x_p, y_p)
    points = np.array([xx_p.flatten(), yy_p.flatten()]).T

    # points = []
    # for i in range(n_points):
    #     point = (np.random.randint(0, map.shape[0]),
    #              np.random.randint(0, map.shape[1]))
    #     while (point in points) or (map[point[0], point[1]] == 0):
    #         point = (np.random.randint(0, map.shape[0]),
    #                  np.random.randint(0, map.shape[1]))
    #     points.append(point)
    # points = np.array(points)

    points = sort_points(points)

    density = np.zeros(len(pixels))
    for i in range(n_pixels):
        if map[pixels[i][0], pixels[i][1]] == 0:
            density[i] = 0.
        else:
            density[i] = 1.

    return pixels, points, density


def sort_points(points):
    sorted_idx = np.argsort(points[:, 1])
    return points[sorted_idx]


def voronoi_show(map, points=None, sides=None, edges=None, adjacency=None):
    colors = sns.color_palette("rainbow", len(points))
    img_show = np.array(cv2.cvtColor(map.astype(np.uint8), cv2.COLOR_GRAY2BGR))

    if points is not None:
        img_show[points[:, 0], points[:, 1], :] = [0, 0, 0]

    if sides is not None:
        for i, side in enumerate(sides):
            for pixel in side:
                img_show[int(pixel[0]), int(pixel[1]), :] = np.array(np.array(colors[i][:3]) * 255, dtype=np.uint8)

    if edges is not None:
        for edge in edges:
            plt.plot([edge[0][1], edge[1][1]], [edge[0][0], edge[1][0]], 'k-')
    elif adjacency is not None:
        for i in range(len(adjacency)):
            for j in range(i + 1, len(adjacency)):
                if adjacency[i, j] == 1:
                    plt.plot([points[i][1], points[j][1]], [points[i][0], points[j][0]], 'k-')

    plt.imshow(img_show)
    plt.show()


def voronoi_save(n_points, r, data_dict):
    px = "p%d" % n_points
    rx = "r%d" % r
    os.makedirs(os.path.join("results/", px, rx), exist_ok=True)
    for key, data in data_dict.items():
        pkl_name = os.path.join("results/", px, rx, key + ".pkl")
        pkl.dump(data, open(pkl_name, "wb"))


def pixel_find_nearest_points(args):
    i, pixel, points, n_points, map, diagonal = args
    # known = diagonal  # 以对角线长度作为已知最短距离
    known = 6 * (map.shape[0] + map.shape[0]) / np.sqrt(0.5 * n_points)
    belong = 0

    if map[pixel[0], pixel[1]] == 0:
        return None

    for j in range(n_points):
        if (pixel == points[j]).all():
            return i, j

        if line_distance(pixel, points[j]) > known:
            continue  # 剪枝

        path = get_navi_path(obstacle_map=map,
                             start=tuple([pixel[0], pixel[1]]),
                             goal=tuple([points[j][0], points[j][1]]), )
        _dist = len(path) if (path is not None) else diagonal

        # _dist = line_distance(pixel, points[j])

        if _dist < known:
            known = _dist
            belong = j

    return i, belong


def voronoi_partition(map, points, pixels, diagonal):
    n_pixels = len(pixels)
    n_points = len(points)
    sides = [[] for _ in range(n_points)]
    belongs = [0] * n_pixels

    pool = multiprocessing.Pool()
    args_list = [(i, pixels[i], points, n_points, map, diagonal) for i in range(n_pixels)]

    for result in tqdm(pool.imap_unordered(pixel_find_nearest_points, args_list), total=n_pixels):
        if result is not None:
            i, j = result
            sides[j].append(pixels[i])
            belongs[i] = j
    pool.close()
    pool.join()
    return sides, belongs


def update_points(new_points, belongs, pixels, density, ):
    tmp = np.zeros([len(new_points), 3])
    for i, pixel in enumerate(pixels):
        tmp[belongs[i]][0] += density[i] * pixel[0]
        tmp[belongs[i]][1] += density[i] * pixel[1]
        tmp[belongs[i]][2] += density[i]

    for j in range(len(new_points)):
        new_points[j][0] = tmp[j][0] / (tmp[j][2] + 1e-2)
        new_points[j][1] = tmp[j][1] / (tmp[j][2] + 1e-2)

    return new_points


def process_connect(args):
    map, points, i, judge_sides, known = args
    js = []
    for j in range(i + 1, n_points):
        if line_distance(points[i], points[j]) > known:
            continue
        path = get_navi_path(
            obstacle_map=map,
            start=tuple([points[i][0], points[i][1]]),
            goal=tuple([points[j][0], points[j][1]]),
        )
        if path is not None:
            judge_set = judge_sides[i] + judge_sides[j]
            if all([p in judge_set for p in path]):
                js.append(j)
    return i, js


def update_connectivity(map, points, sides):
    judge_sides = copy.deepcopy(sides)
    n_points = len(points)
    for i in range(n_points):
        judge_sides[i] = [tuple(p) for p in sides[i]]
    connectivity = np.zeros([n_points, n_points])
    known = 6 * (map.shape[0] + map.shape[0]) / np.sqrt(0.5 * n_points)

    pool = multiprocessing.Pool()
    args_list = [(map, points, i, judge_sides, known) for i in range(n_points)]
    for result in tqdm(pool.imap_unordered(process_connect, args_list), total=n_points):
        i, js = result
        for j in js:
            connectivity[i, j] = connectivity[j, i] = 1
    pool.close()
    pool.join()

    # for i in tqdm(range(n_points)):
    #     for j in range(i + 1, n_points):
    #         if np.linalg.norm(points[i] - points[j]) > 50:
    #             continue
    #         path = get_navi_path(
    #             obstacle_map=map,
    #             start=tuple([points[i][0], points[i][1]]),
    #             goal=tuple([points[j][0], points[j][1]]),
    #         )
    #         judge_set = judge_sides[i] + judge_sides[j]
    #         if all([p in judge_set for p in path]):
    #             connectivity[i, j] = 1
    #             connectivity[j, i] = 1

    return connectivity


def voronoi_map(n_points, map):
    pixels, points, density = init_voronoi(n_points, map)
    n_pixels, n_points = len(pixels), len(points)

    past_points = points.copy()
    belongs = [0] * len(pixels)  # 记录每个pixel属于哪个side
    sides = [[] for _ in range(n_points)]  # 记录每个side包含哪些pixel
    connectivity = np.zeros([n_points, n_points])  # 记录两个side是否连通

    max_r = 40
    diagonal = map.shape[0] + map.shape[1]
    data_dict = {"points": points, "belongs": belongs, "sides": sides, "connectivity": connectivity}
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("results/", "p%d" % n_points), exist_ok=True)
    for r in range(max_r):

        points = sort_points(points)

        sides, belongs = voronoi_partition(map, points, pixels, diagonal)
        data_dict["sides"], data_dict["belongs"] = sides, belongs

        points = update_points(points, belongs, pixels, density)
        data_dict["points"] = points

        connectivity = update_connectivity(map, points, sides)
        data_dict["connectivity"] = connectivity
        # print(connectivity)

        voronoi_show(map, points=points, sides=sides, adjacency=connectivity)
        voronoi_save(n_points, r, data_dict)

        err = np.sum(np.linalg.norm(past_points - points, axis=1))

        if err > 1.:
            past_points = points.copy()
        else:
            print("Converged!")
            return
    print("Not Converged Until Max Iteration!")
    return


if __name__ == "__main__":
    map = read_map("maps/map-downsample-origin.bmp")
    # 0表示障碍物, 1表示通行
    n_points = 2 * 4 ** 2
    voronoi_map(n_points, map)


