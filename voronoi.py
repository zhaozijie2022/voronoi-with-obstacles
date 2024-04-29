import multiprocessing
import cv2
import numpy as np
import matplotlib.pyplot as plt
from navigation import get_navi_path, line_distance
from read_img import read_img

from tqdm import tqdm
import multiprocessing
import seaborn as sns


def process_pixel(args):
    # 计算每个像素点到维诺顶点的距离, 并返回
    i, data, points, n_points, img, diagonal = args
    known = diagonal  # 以对角线长度作为已知最短距离
    belong = 0

    if img[data[i][0], data[i][1]] == 0:
        return None  # 如果该点是障碍物, 则跳过

    for j in range(n_points):
        if line_distance(data[i], points[j]) > known:
            continue  # 如果该点到维诺顶点的直线距离都大于已知最短距离, 则跳过, 不必计算导航距离

        if (data[i] == points[j]).all():
            belong = j
            break

        path = get_navi_path(
            obstacle_map=img,
            start=tuple([data[i][0], data[i][1]]),
            goal=tuple([points[j][0], points[j][1]]),
        )
        _dist = len(path)

        if _dist <= known:
            known = _dist
            belong = j

        # _dist = line_distance(data[pixel], points[j])
        # if _dist < known:
        #     known = _dist
        #     belong = j

    return i, belong


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


def voronoi_map(n_points, img):
    # region
    x = np.arange(0, img.shape[0], 1)
    y = np.arange(0, img.shape[1], 1)
    xx, yy = np.meshgrid(x, y)
    data = np.array([xx.flatten(), yy.flatten()]).T
    n_pixels = len(data)

    x_p = np.linspace(5, img.shape[0] - 5, int(np.sqrt(0.5 * n_points)), dtype=np.int32)
    y_p = np.linspace(5, img.shape[1] - 5, 2 * int(np.sqrt(0.5 * n_points)), dtype=np.int32)
    xx_p, yy_p = np.meshgrid(x_p, y_p)
    points = np.array([xx_p.flatten(), yy_p.flatten()]).T
    n_points = len(points)

    den = np.zeros(len(data))
    for i in range(n_pixels):
        if img[data[i][0], data[i][1]] == 0:
            den[i] = -5.
        else:
            den[i] = 1.
    belongs = [0] * len(data)

    new_points = points * 1
    # endregion

    r = 0
    diagonal = np.linalg.norm([img.shape[0], img.shape[1]])
    while True:

        img_show = img * 1
        np.save("r%d_p%d.npy" % (n_points, r), points)
        for i, point in enumerate(new_points):
            img_show[int(point[0]), int(point[1])] = 0

        plt.imshow(img_show, cmap="gray")
        plt.show()

        sides = [[] for _ in range(n_points)]

        pool = multiprocessing.Pool()
        args_list = [(i, data, points, n_points, img, diagonal) for i in range(n_pixels)]

        for result in tqdm(pool.imap_unordered(process_pixel, args_list), total=n_pixels):
            if result is not None:
                i, j = result
                sides[j].append(data[i])
                belongs[i] = j
        pool.close()
        pool.join()
        print([len(side) for side in sides])

        colors = sns.color_palette("rainbow", n_points)
        img_show = np.array(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR))
        for i, pixel in enumerate(data):
            img_show[int(pixel[0]), int(pixel[1]), :] = np.array(np.array(colors[belongs[i]][:3]) * 255, dtype=np.uint8)
        cv2.imshow("img_show", img_show)
        cv2.waitKey(0)

        # plt.show()

        # for j, side in tqdm(enumerate(sides)):
        #     new_point = find_nearest_point(pixels=side, obstacle_map=img)
        #     new_points[j] = new_point

        # pool2 = multiprocessing.Pool()
        # args_list2 = [(side, img) for side in sides]
        # for result in tqdm(pool2.imap_unordered(process_centroid, args_list2), total=n_points):
        #     for j, new_point in enumerate(result):
        #         new_points[j] = new_point
        # pool2.close()
        # pool2.join()

        tmp = np.zeros([n_points, 3])
        for i, pixel in enumerate(data):
            tmp[belongs[i]][0] += den[i] * pixel[0]
            tmp[belongs[i]][1] += den[i] * pixel[1]
            tmp[belongs[i]][2] += den[i]

        for j, point in enumerate(points):
            new_points[j][0] = tmp[j][0] / (tmp[j][2] + 1e-2)
            new_points[j][1] = tmp[j][1] / (tmp[j][2] + 1e-2)

        err = 0
        r += 1



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
    # img = read_img("./map-dilation.bmp")
    img = read_img("./map-downsample-origin.bmp")
    # 0表示障碍物, 1表示通行
    n_points = (2 * 2 ** 2)
    points = voronoi_map(n_points, img)
    plt.imshow(img, cmap='gray')