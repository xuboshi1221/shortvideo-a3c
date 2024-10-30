#这段 Python 代码定义了一个名为 load_trace 的函数，其目的是从指定的文件夹中加载网络跟踪数据文件，并解析这些文件中的时间戳和带宽数据
#0代表4g的trace，1代表wifi的trace。合成01后的文件，第一列是4g的带宽，第二列是wifi的带宽
import os

COOKED_TRACE_FOLDER = './data/network_traces/multipath/'  #保存网络轨迹的目录
BW_ADJUST_PARA = 1 #定义一个全局变量 BW_ADJUST_PARA，用于调整带宽数据的参数。在这个例子中，它被设置为 1，意味着不会对带宽数据进行调整。你可以根据需要修改这个值来进行缩放。

#定义一个函数 load_trace，它接受一个参数 cooked_trace_folder，该参数指定了包含跟踪数据文件的文件夹路径。默认值是之前定义的 COOKED_TRACE_FOLDER。
def load_trace(cooked_trace_folder=COOKED_TRACE_FOLDER):
    cooked_files = os.listdir(cooked_trace_folder)  #使用 os.listdir 函数列出指定文件夹中的所有文件名。
    cooked_files.sort(key=lambda x: int(x))     #将文件名列表 cooked_files 进行排序。这里假设文件名是数字，使用 lambda 函数将文件名转换为整数进行排序。
    #0代表4g的trace，1代表wifi的trace

    all_cooked_time = []   #初始化三个空列表，用于存储所有文件的时间戳和带宽数据。
    all_cooked_4gbw = []
    all_cooked_wifibw = []
    for cooked_file in cooked_files:   #用for循环遍历每个文件名
        file_path = cooked_trace_folder + cooked_file   #构建每个文件的完整路径。
        cooked_time = []  #为当前文件初始化三个空列表，用于存储时间戳和带宽数据。
        cooked_4gbw = []
        cooked_wifibw = []
        with open(file_path, 'r') as f:  #打开文件，'rb' 模式表示以二进制读取模式打开。
            for line in f:    #读取文件的每一行
                parse = line.split(',')  #split() 函数按逗号分割一行数据
                cooked_time.append(float(parse[0]))    #分割后的第一个元素（时间戳）转换为浮点数并添加到 cooked_time 列表中
                cooked_4gbw.append(float(parse[1])*BW_ADJUST_PARA)   #第二个元素（带宽）转换为浮点数、乘以 BW_ADJUST_PARA 并添加到 cooked_4gbw 列表中。
                cooked_wifibw.append(float(parse[2])*BW_ADJUST_PARA)   #第三个元素（带宽）转换为浮点数、乘以 BW_ADJUST_PARA 并添加到 cooked_wifibw 列表中。
        # all_cooked_time.append(cooked_time) #将当前文件的时间戳和带宽数据添加到相应的总列表中。all_cooked_time[0]是一个轨迹中的所有带宽数据。
        # all_cooked_4gbw.append(cooked_4gbw)
        # all_cooked_wifibw.append(cooked_wifibw)

    return cooked_time, cooked_4gbw, cooked_wifibw
