import random
import time

import numpy as np

import utils  # 自定义工具函数包

# 蚂蚁数为城市数的2 / 3
num_ant = 4    # 蚂蚁数量
alpha = 2  # 信息素影响因子，选择[1, 5]比较合适
beta = 1  # 期望影响因子，选择[1, 5]比较合适
info = 0.3  # 信息素的挥发率，选择0.3比较合适
Q = 5  # 信息素增加强度系数


inf = 1e8  # 定义无穷大值
def cal_newpath(dis_mat, path_new, cityNum):
    """
  计算所有路径对应的距离
  :param dis_mat: 城市距离矩阵 ndarray
  :param path_new: 路径矩阵 ndarray
  :param cityNum: 城市数量 int
  :return: 动态规划最优路径 list
  """
    dis_list = []
    for each in path_new:
        dis = 0
        for j in range(cityNum - 1):
            dis = dis_mat[each[j]][each[j + 1]] + dis
        dis = dis_mat[each[cityNum - 1]][each[0]] + dis  # 回家
        dis_list.append(dis)
    return dis_list


def getDisAndPath(point, cityNum, setting):
    """
  计算所有路径对应的距离
  :param point: 城市距离矩阵 ndarray
  :param cityNum: 城市数量 int
  :param setting: 函数相关配置 obj, 见antColonyOptimization的setting
  :return: 最短路径值 double 最短路径 list
  """

    dis_mat = np.array(point)  # 转为矩阵
    # 期望矩阵
    e_mat_init = 1.0 / (dis_mat + np.diag([10000] * cityNum))  # 加对角阵是因为除数不能是0
    diag = np.diag([1.0 / 10000] * cityNum)
    e_mat = e_mat_init - diag  # 还是把对角元素变成0
    pheromone_mat = np.ones((cityNum, cityNum))  # 初始化每条边的信息素浓度，全1矩阵
    path_mat = np.zeros((num_ant, cityNum)).astype(int)  # 初始化每只蚂蚁路径，都从0城市出发
    count_iter = 0
    # 设置了一个统计连续多次没有产生更优解的计数器counter，如果当前迭代产生的解与上一次迭代产生的解相同，counter的值加1，当counter的值大于某一阈值threshold时，减少迭代次数skipNum次，同时counter清零。
    counter = 0  # 蚁群迭代次数简单优化方案的计数器
    ifOptimanation = setting["ifOptimanation"]
    threshold = setting["threshold"]
    iter_max = setting["iter_max"]
    skipNum = setting["skipNum"]
    pre_min_path = 0
    while count_iter < iter_max:
        for ant in range(num_ant):
            visit = 0  # 都从0城市出发
            unvisit_list = list(range(1, cityNum))  # 未访问的城市
            for j in range(1, cityNum):
                # 轮盘法选择下一个城市
                trans_list = []
                tran_sum = 0
                trans = 0
                for k in range(len(unvisit_list)):
                    trans += np.power(pheromone_mat[visit][unvisit_list[k]], alpha) * np.power(
                        e_mat[visit][unvisit_list[k]], beta)
                    trans_list.append(trans)
                    tran_sum = trans
                rand = random.uniform(0, tran_sum)  # 产生随机数
                for t in range(len(trans_list)):
                    if rand <= trans_list[t]:
                        visit_next = unvisit_list[t]
                        break
                    else:
                        continue
                path_mat[ant, j] = visit_next  # 填路径矩阵
                unvisit_list.remove(visit_next)  # 更新
                visit = visit_next  # 更新
        # 所有蚂蚁的路径表填满之后，算每只蚂蚁的总距离
        dis_allant_list = cal_newpath(dis_mat, path_mat, cityNum)
        # 每次迭代更新最短距离和最短路径
        if count_iter == 0:
            dis_new = min(dis_allant_list)
            path_new = path_mat[dis_allant_list.index(dis_new)].copy()
        else:
            if min(dis_allant_list) < dis_new:
                dis_new = min(dis_allant_list)
                path_new = path_mat[dis_allant_list.index(dis_new)].copy()
        # 蚁群算法迭代次数的简单优化
        if ifOptimanation == True:
            if round(pre_min_path, 2) == round(dis_new, 2):
                counter += 1
                if counter >= threshold:
                    iter_max -= skipNum
                    counter = 0
            pre_min_path = dis_new
        # 更新信息素矩阵
        pheromone_change = np.zeros((cityNum, cityNum))
        for i in range(num_ant):
            for j in range(cityNum - 1):
                pheromone_change[path_mat[i, j]][path_mat[i, j + 1]] += Q / dis_mat[path_mat[i, j]][path_mat[i, j + 1]]
            pheromone_change[path_mat[i, cityNum - 1]][path_mat[i, 0]] += Q / dis_mat[path_mat[i, cityNum - 1]][
                path_mat[i, 0]]
        pheromone_mat = (1 - info) * pheromone_mat + pheromone_change
        count_iter += 1  # 迭代计数+1，进入下一次
    return dis_new, path_new.tolist(), iter_max


def antColonyOptimization(cityNum, coordinate, point, setting):
    """
  蚁群算法
  :param cityNum: 城市数量 int
  :param coordinate: 城市坐标 list
  :param point: 城市距离矩阵 ndarray
  :param setting: 函数相关配置 obj
  :return: 最小距离 double, 运行时间 double, 迭代次数 int

  setting相关配置:
    iter_max: 最大迭代次数 int
    ifOptimanation: 是否使用简单优化后的方案 bool
    threshold: 阈值 int
    skipNum: 达到阈值后跳过的迭代次数 int
  示例:
    setting = {
      "iter_max": 500,
      "ifOptimanation": True,
      "threshold": 6,
      "skipNum": 20
    }
  """
    start = time.perf_counter()  # 程序开始时间
    # skipNum次数为1 5 10 15
    dis, path, iterNum = getDisAndPath(point, cityNum, setting)
    end = time.perf_counter()  # 程序结束时间
    utils.printTable(path, 7, end - start, cityNum, round(dis, 2))  # 打印表格
    utils.drawNetwork(coordinate, point, path, inf)
    return round(dis, 2), end - start, iterNum




