# Simulate the user watch pattern#这段代码定义了一个名为 Retention 的类，用于模拟用户观看视频的行为模式，特别是用户在观看过程中的保留率和流失率。这个类通过随机数生成器来模拟用户可能在何时停止观看视频。
import numpy as np
import math
import random
VIDEO_CHUNCK_LEN = 1000.0  # 视频块的长度，单位为毫秒
# RANDOM_SEED = 42
# np.random.seed(RANDOM_SEED)
# random.seed(RANDOM_SEED)

class Retention:  #已经读取过用户留存率文件了，这个函数根据文件的两列数据生成用户可能在哪个时间点离开。
    def __init__(self, user_time, user_retent_rate, seeds): #初始化方法，用于创建用户保留率模型，模拟用户观看时长。seeds接受的是两个元素的数组

        assert len(user_time) == len(user_retent_rate)  # 确保用户的时间和保留率数据长度一致
        self.user_time = user_time  # 将传入的用户观看时间和保留率数据进行存储
        self.user_retent_rate = user_retent_rate  
        # print(self.user_time)
        # print(self.user_retent_rate)
        video_time_len = self.user_time[-2]  # 计算视频时长，17秒的视频有0-18行，第0行100%留存率，第18行0%留存率，所以-2下标才是真实的视频时长。

        self.user_churn_rate = 1.0 - np.array(user_retent_rate).astype('float64')  # 用户的流失率：1-保留率，转化为浮点数数组。churn_rate流失率
        self.prop = np.diff(self.user_churn_rate).ravel() # 计算流失率的差分，用于概率分布

        np.random.seed(seeds[0])  # 使用第一个随机种子初始化numpy随机数生成器,下句使用np.random.choice表示在哪一秒停止观看。
        #print(seeds[0])
        interval = np.random.choice(self.user_time[:-1], p=self.prop)  # ms # 根据流失率差分分布，选择用户在某个时间点的观看时长间隔。# 选择停止观看的时间点，单位为毫秒
        if interval == self.user_time[-2]:  #  如果用户观看到了倒数第二个时间点，意味着用户观看到视频末尾。0-18行数据，其实视频只有17秒
            self.sample_playback_duration = interval  # 用户观看到最后一个块的结束
        else:  # uniform distribute over the second   # 否则在选择的时间点和下一秒之间进行均匀分布采样，模拟随机的观看时长
            random.seed(seeds[1])  # 使用第二个随机种子初始化Python的随机数生成器，下句使用random.uniform具体到某一毫秒停止观看。
            #print(seeds[1])
            self.sample_playback_duration = int(random.uniform(interval, interval+1000)) # 具体观看到某一毫秒。

    def get_ret_duration(self):  # ms   # 获取用户在该视频下离开的时间，单位为毫秒
        return self.sample_playback_duration  

    def get_watch_chunk_cnt(self):   # 获取用户观看的块数，如果在12421ms离开，用户观看了12个块
        return math.floor(self.sample_playback_duration / VIDEO_CHUNCK_LEN)  
