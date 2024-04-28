import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from a_star import get_navi_path
from tqdm import tqdm


def line_distance(p1, p2):
    return np.linalg.norm(p1 - p2)


def read_img(img_path="./map-downsample-origin.bmp"):
    # read in gray
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = np.array(img, dtype=np.float32)
    # img[img > 35] = 1.0
    return img


def find_nearest_point(points, obstacle_map):
    # 遍历side中的所有pixels, 寻找与所有点距离之和最近的点 作为质心, 即新的vertex
    min_distance_sum = float('inf')  # 初始化最小距离和为正无穷大
    nearest_point = None

    for point in points:
        distance_sum = 0

        for other_point in points:
            if point != other_point:
                distance_sum += get_navi_path(
                    obstacle_map=obstacle_map,
                    start=tuple([point[0], point[1]]),
                    goal=tuple([other_point[0], other_point[1]])
                )
                if distance_sum > min_distance_sum:
                    break

        if distance_sum < min_distance_sum:
            min_distance_sum = distance_sum
            nearest_point = point

    return nearest_point


def gen_voronoi_map(n_points, img):
    x = np.arange(0, img.shape[0], 1)
    y = np.arange(0, img.shape[1], 1)
    xx, yy = np.meshgrid(x, y)
    data = np.array([xx.flatten(), yy.flatten()]).T
    n_pixels = len(data)

    # den = np.zeros(len(data))
    # for i in range(len(data)):
    #     den[i] = img[int(data[i][0]), int(data[i][1])]

    # points = np.array([np.random.randint(0, img.shape[0], n_points), np.random.randint(0, img.shape[1], n_points)]).T
    x_p = np.linspace(0, img.shape[0], int(np.sqrt(0.5 * n_points)))
    y_p = np.linspace(0, img.shape[1], 2 * int(np.sqrt(0.5 * n_points)))
    xx_p, yy_p = np.meshgrid(x_p, y_p)
    points = np.array([xx_p.flatten(), yy_p.flatten()]).T
    n_points = len(points)

    new_points = points * 1
    belong = [0] * len(data)

    r = 0
    diagonal = np.linalg.norm([img.shape[0], img.shape[1]])
    while True:
        sides = [[] for _ in range(n_points)]

        # 对每一个像素点, 计算他们与哪个vertex最近
        for i in tqdm(range(n_pixels)):
            known = diagonal  # 以对角线长度作为已知最短距离
            for j in range(n_points):
                # dist.append(np.linalg.norm(pixel - points[j]))

                if line_distance(data[i], points[j]) > known:
                    continue  # 如果直线距离都已经大于known, 那么就不必计算导航距离了

                if (data[i] == points[j]).all():
                    known = 0
                    belong[i] = j
                    break

                path = get_navi_path(
                    obstacle_map=img,
                    start=tuple([data[i][0], data[i][1]]),
                    goal=tuple([points[j][0], points[j][1]]),
                    known_shortest_distance=known
                )
                if path is not None and (len(path) < known):
                    known = len(path)
                    belong[i] = j

            sides[belong[i]].append(data[i])

        # 对每一个sides, 计算他们的质心
        for j, side in tqdm(enumerate(sides)):
            new_point = find_nearest_point(points=side, obstacle_map=img)
            new_points[j] = new_point




        # tmp = np.zeros([n_points, 3])
        # for i, pixel in enumerate(data):
        #     tmp[belong[i]][0] += den[i] * pixel[0]
        #     tmp[belong[i]][1] += den[i] * pixel[1]
        #     tmp[belong[i]][2] += den[i]
        #
        # for j, point in enumerate(points):
        #     new_points[j][0] = tmp[j][0] / (tmp[j][2] + 1e-2)
        #     new_points[j][1] = tmp[j][1] / (tmp[j][2] + 1e-2)

        err = 0
        r += 1
        # colors = sns.color_palette("rainbow", n_points)
        # img_show = np.array(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR))
        # for i, pixel in enumerate(data):
        #     img_show[int(pixel[0]), int(pixel[1]), :] = np.array(np.array(colors[belong[i]][:3]) * 255, dtype=np.uint8)

        img_show = img * 1
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
            points = new_points * 1
            # new_points = points * 1
        else:
            return points


if __name__ == "__main__":
    img = read_img("./map-downsample-origin.bmp")
    n_points = 2 * 5 ** 2
    points = gen_voronoi_map(n_points, img)
    plt.imshow(img, cmap='gray')
