import ACO
import numpy as np

# 旅行商测试矩阵
# 	1	2	3	4	5	6
# 1	0	6	2	1	MAX	MAX
# 2	6	0	6	MAX	3	MAX
# 3	2	6	0	2	2	4
# 4	1	MAX	2	0	MAX	5
# 5	MAX	3	2	MAX	0	3
# 6	MAX	MAX	4	5	3	0


# 旅行商初始化函数，总共6个城市
def InitD():
    cityNum = 6
    coordinate = [(4, 7.5), (0, 5.5), (5, 4), (10, 5.5), (1.5, 1), (11, 0)]
    MAX_INT = 1e8
    point = np.array([[0, 6, 2, 1, MAX_INT, MAX_INT], [6, 0, 6, MAX_INT, 3, MAX_INT], [2, 6, 0, 2, 2, 4],
                      [1, MAX_INT, 2, 0, MAX_INT, 5], [MAX_INT, 3, 2, MAX_INT, 0, 3], [MAX_INT, MAX_INT, 4, 5, 3, 0]])

    return cityNum, coordinate, point


def ACOTest():
    cityNum, coordinate, point = InitD()
    ACO.antColonyOptimization(cityNum, coordinate, point, setting={
        "iter_max": 300,
        "ifOptimanation": False,
        "threshold": 6,
        "skipNum": 20
    })


if __name__ == '__main__':
    ACOTest()
