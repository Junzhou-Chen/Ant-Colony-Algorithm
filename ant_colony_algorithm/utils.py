import matplotlib.pyplot as plt
import networkx as nx
from prettytable import PrettyTable


# 打印旅行商问题的运行结果表格

def createTable(table_obj):
    """
  打印数据表格
  :param table_obj: 表格对象 obj
  :return: none
  参数示例:
  result_obj = {
    "header": ["TSP参数", "运行结果"],
    "body": [
      ["城市数量", cityNum],
      ["最短路程", distance],
      ["运行时间", time_str],
      ["最小路径", path_str]
    ],
    # name的值要和header一致, l: 左对齐 c: 居中 r: 右对齐
    "align": [
      { "name": "TSP参数", "method": "l" },
      { "name": "运行结果", "method": "l" }
    ],
    "setting": {
      "border": True, # 默认True
      "header": True, # 默认True
      "padding_width": 5 # 空白宽度
    }
  }
  """
    pt = PrettyTable()
    for key in table_obj:
        # 打印表头
        if key == "header":
            pt.field_names = table_obj[key]
        # 打印表格数据
        elif key == "body":
            for i in range(len(table_obj[key])):
                pt.add_row(table_obj[key][i])
        # 表格参数的对齐方式
        elif key == "align":
            for i in range(len(table_obj[key])): pt.align[table_obj[key][i]["name"]] = table_obj[key][i]["method"]
        # 表格其他设置
        elif key == "setting":
            for key1 in table_obj[key]:
                if key1 == "border":
                    pt.border = table_obj[key][key1]
                elif key1 == "hearder":
                    pt.header = table_obj[key][key1]
                elif key1 == "padding_width":
                    pt.padding_width = table_obj[key][key1]
            # for key1 in table_obj[key]: pt[key1] = table_obj[key][key1]
    print(pt)


# survive
def timeFormat(number):
    """
  时间格式保持两位
  :param number: 数字 int
  :return: 两位的数字字符 str
  """
    if number < 10:
        return "0" + str(number)
    else:
        return str(number)


def calcTime(time):
    """
  将毫秒根据数值大小转为合适的单位
  :param time: 数字 double
  :return: 时间字符串 str
  """
    count = 0
    while time < 1:
        if count == 3:
            break
        else:
            count += 1
        time *= 1000
    if count == 0:
        hour = int(time // 3600)
        minute = int(time % 3600 // 60)
        second = time % 60
        if hour > 0: return timeFormat(hour) + "时" + timeFormat(minute) + "分" + timeFormat(int(second)) + "秒"
        if minute > 0: return timeFormat(minute) + "分" + timeFormat(int(second)) + "秒"
        if second > 0: return str(round(time, 3)) + "秒"
    elif count == 1:
        return str(round(time, 3)) + "毫秒"
    elif count == 2:
        return str(round(time, 3)) + "微秒"
    elif count == 3:
        return str(round(time, 3)) + "纳秒"


# survive
def pathToString(path, everyRowNum):
    """
  将最优路径列表转为字符串
    :param everyRowNum:
  :param path: 最优路径列表 list
  :param: everyRowNum 每行打印的路径数,除去头尾 int
  :return: 路径字符串 str
  """
    min_path_str = ""
    for i in range(len(path)):
        min_path_str += str(path[i] + 1) + ("\n--> " if i != 0 and i % everyRowNum == 0 else " --> ")
    min_path_str += "1"  # 单独输出起点编号
    return min_path_str


# 打印表格
def printTable(path, everyRowNum, runTime, cityNum, distance):
    """
  将最优路径列表转为字符串
  :param: path: 最优路径列表 list
  :param: everyRowNum 每行打印的路径数,除去头尾 int
  :param: runTime 程序运行时间 double
  :param: cityNum 城市数量 int
  :param: distance 最优距离 double
  :return: none
  """
    path_str = pathToString(path, everyRowNum)
    time_str = calcTime(runTime)  # 程序耗时
    # 打印的表格对象
    result_obj = {
        "header": ["TSP参数", "运行结果"],
        "body": [
            ["城市数量", cityNum],
            ["最短路程", distance],  # 最小值就在第一行最后一个
            ["运行时间", time_str],  # 计算程序执行时间
            ["最小路径", path_str]  # 输出路径
        ],
        "align": [
            {"name": "参数", "method": "l"},
            {"name": "运行结果", "method": "l"}
        ],
    }
    createTable(result_obj)  # 打印结果


###########################################################################
# 画图函数

def isPath(path, i, j):
    """
  判断边是否为最小路径
  :param path: 最优路径列表 list
  :param: i / j 路径的下标 int
  :return: 布尔值
  """
    idx = path.index(i)
    pre_idx = idx - 1 if idx - 1 >= 0 else len(path) - 1
    next_idx = idx + 1 if idx + 1 < len(path) else 0
    if j == path[pre_idx] or j == path[next_idx]:
        return True
    return False


def drawNetwork(coordinate, point, path, inf, *args):
    """
  画出网络图
  :param coordinate: 城市坐标 list
  :param: point 城市距离矩阵 ndarray
  :param: path 最优路径 list
  :param: inf 无穷大值 double
  :return: none
  """

    G_min = nx.Graph()  # 最短路径解
    G = nx.Graph()  # 城市路径图
    edges = []
    for i in range(len(coordinate)):
        m = i + 1
        G_min.add_node(m, pos=coordinate[i])  # 添加节点
        G.add_node(m, pos=coordinate[i])
        for j in range(i + 1, len(coordinate)):
            if point[i][j] != inf:
                if isPath(path, i, j):
                    G_min.add_edge(i + 1, j + 1, weight=int(point[i][j]), color='r')
                G.add_edge(i + 1, j + 1, weight=int(point[i][j]))
    tmp_edges = nx.get_edge_attributes(G_min, 'color')
    for key in tmp_edges:
        edges.append(tmp_edges[key])
    pos = pos_min = nx.get_node_attributes(G_min, 'pos')
    labels = nx.get_edge_attributes(G_min, 'weight')
    label = nx.get_edge_attributes(G, 'weight')
    # 城市所有路径
    plt.subplot(121)
    plt.title("TSP City Network")
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_color='y')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=label)  # 画路径长度
    # 最短路径解
    plt.subplot(122)
    plt.title("Solution Of Ant Colony Algorithm")
    nx.draw(G_min, pos_min, with_labels=True, font_weight='bold', node_color='g', edge_color=edges)
    nx.draw_networkx_edge_labels(G_min, pos_min, edge_labels=labels)  # 画路径长度
    plt.show()




