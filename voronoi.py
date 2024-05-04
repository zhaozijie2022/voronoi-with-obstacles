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

    density = np.zeros(len(pixels))
    for i in range(n_pixels):
        if map[pixels[i][0], pixels[i][1]] == 0:
            density[i] = 0.
        else:
            density[i] = 1.

    return pixels, points, density


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


def voronoi_save(p_name, data_dict):
    os.makedirs(os.path.join("results/", p_name), exist_ok=True)
    for key, data in data_dict.items():
        pkl_name = os.path.join("results/", p_name, key + "_" + p_name + ".pkl")
        pkl.dump(data, open(pkl_name, "wb"))


def pixel_find_nearest_points(args):
    i, pixel, points, n_points, map, diagonal = args
    known = diagonal  # 以对角线长度作为已知最短距离
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


def update_connectivity(map, points, sides):
    judge_sides = copy.deepcopy(sides)
    n_points = len(points)
    for i in range(n_points):
        judge_sides[i] = [tuple(p) for p in sides[i]]
    connectivity = np.zeros([n_points, n_points])
    for i in tqdm(range(n_points)):
        for j in range(i + 1, n_points):
            if np.linalg.norm(points[i] - points[j]) > 50:
                continue
            path = get_navi_path(
                obstacle_map=map,
                start=tuple([points[i][0], points[i][1]]),
                goal=tuple([points[j][0], points[j][1]]),
            )
            judge_set = judge_sides[i] + judge_sides[j]
            if all([p in judge_set for p in path]):
                connectivity[i, j] = 1
                connectivity[j, i] = 1

    return connectivity


def voronoi_map(n_points, map):
    pixels, points, density = init_voronoi(n_points, map)
    n_pixels, n_points = len(pixels), len(points)

    new_points = copy.deepcopy(points)
    belongs = [0] * len(pixels)  # 记录每个pixel属于哪个side
    sides = [[] for _ in range(n_points)]  # 记录每个side包含哪些pixel
    connectivity = np.zeros([n_points, n_points])  # 记录两个side是否连通

    max_r = 40
    diagonal = map.shape[0] + map.shape[1]
    data_dict = {"points": new_points, "belongs": belongs, "sides": sides, "connectivity": connectivity}
    os.makedirs("results", exist_ok=True)
    for r in range(max_r):

        sides, belongs = voronoi_partition(map, new_points, pixels, diagonal)
        data_dict["sides"], data_dict["belongs"] = sides, belongs

        new_points = update_points(new_points, belongs, pixels, density)
        data_dict["points"] = new_points

        connectivity = update_connectivity(map, new_points, sides)
        data_dict["connectivity"] = connectivity
        print(connectivity)

        voronoi_show(map, points=new_points, sides=sides, adjacency=connectivity)
        voronoi_save("p%d_r%d" % (n_points, r), data_dict)

        err = np.sum(np.linalg.norm(points - new_points, axis=1))

        if err > 5.:
            points = new_points.copy()
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

# def find_nearest_point(pixels, obstacle_map):
#     # 遍历side中的所有pixels, 寻找与所有点距离之和最近的点 作为质心, 即新的vertex
#     min_distance_sum = float('inf')  # 初始化最小距离和为正无穷大
#     nearest_point = None
#
#     for pixel in pixels:
#         if obstacle_map[int(pixel[0]), int(pixel[1])] == 0:
#             continue
#         distance_sum = 0
#
#         for other_pixel in pixels:
#             if obstacle_map[int(other_pixel[0]), int(other_pixel[1])] == 0:
#                 continue
#             if not (pixel == other_pixel).all():
#                 # distance_sum += len(get_navi_path(
#                 #     obstacle_map=obstacle_map,
#                 #     start=tuple([point[0], point[1]]),
#                 #     goal=tuple([other_point[0], other_point[1]])
#                 # ))
#                 distance_sum += line_distance(pixel, other_pixel)
#                 if distance_sum > min_distance_sum:
#                     break
#
#         if distance_sum < min_distance_sum:
#             min_distance_sum = distance_sum
#             nearest_point = pixel
#
#     return nearest_point
#
#
# def process_centroid(args):
#     pixels, obstacle_map = args
#     min_distance_sum = float('inf')  # 初始化最小距离和为正无穷大
#     nearest_point = None
#
#     for pixel in pixels:
#         if obstacle_map[int(pixel[0]), int(pixel[1])] == 0:
#             continue
#         distance_sum = 0
#
#         for other_pixel in pixels:
#             if obstacle_map[int(other_pixel[0]), int(other_pixel[1])] == 0:
#                 continue
#             if not (pixel == other_pixel).all():
#                 distance_sum += len(calculate_shortest_distance(
#                     obstacle_map=obstacle_map,
#                     start=tuple([pixel[0], pixel[1]]),
#                     goal=tuple([other_pixel[0], other_pixel[1]])
#                 ))
#                 # distance_sum += line_distance(pixel, other_pixel)
#
#                 if distance_sum > min_distance_sum:
#                     break
#
#         if distance_sum < min_distance_sum:
#             min_distance_sum = distance_sum
#             nearest_point = pixel
#
#     return nearest_point
