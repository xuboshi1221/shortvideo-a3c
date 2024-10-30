# multi-video play #这段代码定义了一个名为 Player 的类，它模拟了一个视频播放器的行为，包括视频的下载、播放以及与用户观看行为相关的一些属性。
import numpy as np
import math
import os

#VIDEO_BIT_RATE = [750,1200,1850]  # Kbps 视频比特率级别，用于计算平滑度。比如这个块级别是0，下个块级别是1，则平滑度惩罚是1200-750=450
#BITRATE_LEVELS = 3  # 视频的码率等级（3种，0,1,2）
VIDEO_CHUNCK_LEN = 1000.0    # 视频块长度，单位为毫秒，每个块1000毫秒
MILLISECONDS_IN_SECOND = 1000.0   # 将秒转换为毫秒的常量,一秒等于1000毫秒
VIDEO_SIZE_FILE = 'data/short_video_size/' # 存储视频尺寸文件的目录
VIDEO_SIZE_SCALE = 1.0  # chunk size   # 用于缩放视频块大小
USER_RET = './data/user_ret/'  # 存储用户保留率文件的目录
DISTINCT_VIDEO_NUM = 7   # 视频种类数量，7类视频，每类下面有一个视频。

class Player:  # 视频播放器类定义，代表一个视频
    # initialize each new video and player buffer
    def __init__(self, video_num):  #读取一个视频所有块比特率信息，
        self.video_size = {}  # 初始化视频块字典，用于存储不同码率下的视频块,比如{0:[块1大小,块2大小,...]，1：[块1大小,块2大小,...]，2：[块1大小,块2大小,...]}，键是码率级别，值是块大小列表
        videos = []  # 获取所有可用的视频目录
        for root, dirs, files in os.walk(VIDEO_SIZE_FILE): #第一个值为目录路径，第二个值为列表包含了root目录下的所有子目录的名字，第三个值为列表包含了root目录下的所有非目录文件的名字
            videos = dirs    # 获取所有视频文件夹名称，videos[0]=1_tj
            videos.sort()    # 按字母顺序排序文件夹
            break
        video_name = videos[video_num % DISTINCT_VIDEO_NUM]    # 根据视频编号选择一个视频文件夹，还没到具体的比特率文件
        bitrate =2
        self.video_size[bitrate] = []    # 初始化每个码率的块大小列表，video_sieze字典的键为2，值为空列表。
        file_name = VIDEO_SIZE_FILE + video_name + '/video_size_' + str(bitrate)  # 文件名拼接，把目录和比特率文件连接起来,由于前面设置了bitrate=2,所以只读取最高比特率文件。
        with open(file_name) as f:   # 打开对应码率的视频块大小文件
            for line in f:  # 按行读取每个块的大小
                self.video_size[bitrate].append(int(line.split()[0])/VIDEO_SIZE_SCALE) # 将当前行分割成一个列表，然后取第一个元素，再转化为整数再缩放。将大小存入码率为2的列表    
        # 获取视频的总块数，再计算视频总时长，单位为毫秒
        self.chunk_num = len(self.video_size[2])
        self.video_len = self.chunk_num * VIDEO_CHUNCK_LEN  # ms       
        # 视频块下载相关的计数器
        self.video_chunk_counter = 0   # 已下载的视频块计数器
        self.video_chunk_remain = self.chunk_num - self.video_chunk_counter  # 剩余未下载的视频块数量     
        self.download_chunk_bitrate = []        # 存储已下载块的码率
        # 播放相关的属性 
        self.video_play_counter = 0     # 已播放的视频块计数器   
        self.play_timeline = 0.0        # 当前播放的时间线，用于记录播放进度，ms
        self.buffer_size = 0  # ms      # 视频缓冲区大小，单位为毫秒       
        self.preload_size = 0 # B       # 预加载的视频大小，记录已下载但未播放的视频数据，单位为字节。
        # 加载用户观看时长和保留率信息
        self.user_time = []  # 新建空列表保存用户观看时间
        self.user_retent_rate = []  # 新建空列表保存用户保留率
        with open(USER_RET + video_name) as file:  # 打开用户观看信息文件
            for line in file:
                self.user_time.append(float(line.split()[0]) * MILLISECONDS_IN_SECOND)  # 用户观看时长（秒转换为毫秒）
                self.user_retent_rate.append(line.split()[1])  # 用户保留率

    def get_user_model(self):  # 获取用户观看模型（时间和保留率）
        return self.user_time, self.user_retent_rate   
    def get_video_len(self):  # 获取视频总时长
        return self.video_len
    def get_chunk_sum(self):   # 获取视频的总块数
        return self.chunk_num   
    def get_preload_size(self): # 获取预加载的总大小，B
        return self.preload_size   
    def get_video_size(self, quality): # 获取当前选择下载块的大小
        try:
            video_chunk_size = self.video_size[quality][self.video_chunk_counter]  # 获取video_size字典里键为2的列表数据，列表下标为video_chunk_counter视频块大小
        except IndexError:  # 如果请求的块超出范围，抛出异常
            raise Exception("You're downloading chunk ["+str(self.video_chunk_counter)+"] is out of range. "+ "\n   % Hint: The valid chunk id is from 0 to " + str(self.chunk_num-1) + " %")
        return video_chunk_size
    def record_download_bitrate(self, bit_rate):   # 记录下载块的码率
        self.download_chunk_bitrate.append(bit_rate)  # 将码率添加到下载记录中
        self.preload_size += self.video_size[bit_rate][self.video_chunk_counter]  # 更新预加载大小（字节数据层面，不是时间）
    def get_video_quality(self, chunk_id):    # 获取指定块所选择下载的码率
        if chunk_id >= len(self.download_chunk_bitrate):   # 如果请求的块尚未下载
            return -1  # means no video downloaded    # 返回 -1 表示未下载
        return self.download_chunk_bitrate[chunk_id]   # 返回该块的码率
    def get_downloaded_bitrate(self):   # 获取已下载块的码率列表
        return self.download_chunk_bitrate 
    
    # def get_undownloaded_video_size(self, P):  #获取未来P个未下载的块大小
    #     chunk_playing = self.get_chunk_counter()  # 获取当前下载块的计数器
    #     future_videosize = []  # 存储未来的视频块大小 
    #     size_in_level = []  # 存储每个码率下的块大小
    #     for k in range(P):   # 遍历未来P个块
    #         size_in_level.append(self.video_size[2][int(chunk_playing+k)])   # video_size是个字典，键为2，值为列表
    #     future_videosize.append(size_in_level)    # 将结果添加到列表中
    #     return future_videosize  # 返回未来视频块大小列表

    # def get_future_video_size(self, P): # 获取未来P个未播放的块大小
    #     interval = 1 # 默认间隔为1
    #     chunk_playing = self.get_play_chunk()   # 获取当前播放块的索引
    #     if chunk_playing % 1 == 0:  # Check whether it is an integer    # 检查是否为整数
    #         interval = 0   # 如果是整数，则间隔为0
    #     future_videosize = []   # 存储未来的视频块大小     
    #     size_in_level = []    # 存储每个码率下的块大小
    #     for k in range(P):     # 遍历未来P个块
    #         size_in_level.append(self.video_size[2][int(chunk_playing + interval + k)])  # video_size是个字典，键为2，值为列表
    #     future_videosize.append(size_in_level)  # 将结果添加到列表中
    #     return future_videosize  # 返回未来视频块大小列表

    def get_play_chunk(self): # 获取当前播放块的索引
        return self.play_timeline / VIDEO_CHUNCK_LEN   
    def get_chunk_counter(self):    # 获取已下载块的索引
        return self.video_chunk_counter
    def get_remain_video_num(self):  # 获取剩余未下载块的数量
        self.video_chunk_remain = self.chunk_num - self.video_chunk_counter 
        return self.video_chunk_remain     
    def get_buffer_size(self): # 获取当前缓冲区大小，ms
        return self.buffer_size  
    
    # 计算因用户滑动退出导致的带宽浪费 
    def bandwidth_waste(self, user_ret):    
        download_len = len(self.download_chunk_bitrate)  # 获取已下载的块数
        waste_start_chunk = math.ceil(user_ret.get_ret_duration() / VIDEO_CHUNCK_LEN)  # 计算浪费开始的块索引。ceil上取整，floor下取整
        sum_waste_each_video = 0   # 初始化浪费的总大小
        for i in range(waste_start_chunk, download_len):  # 遍历从浪费开始块到下载结束块
            # download_bitrate = self.download_chunk_bitrate[i]   # 获取该块的下载码率
            download_size = self.video_size[2][i]  # 获取该码率下该块的数据大小
            sum_waste_each_video += download_size  # 累加浪费的总大小
        return sum_waste_each_video  # 返回浪费的总带宽,B    
    # 下载视频，缓冲区增加，单位为毫秒
    def video_download(self, download_len):  # ms
        self.buffer_size += download_len  # 将下载时间添加到缓冲区中,ms
        self.video_chunk_counter += 1   # 下载块计数器增加
        end_of_video = False  # 初始化视频结束标志为False
        if self.video_chunk_counter >= self.chunk_num:  # 如果所有块都已下载
            end_of_video = True   # 标记视频结束
        return end_of_video   # 返回是否下载完视频
    # 播放视频，返回播放后的时间线和缓冲区大小
    def video_play(self, play_time):  # ms
        buffer = self.buffer_size - play_time # 播放时间从缓冲区中扣除，得到剩余缓冲区(ms)
        self.play_timeline += np.minimum(self.buffer_size, play_time)   # 更新播放时间线：播放时间不能超过缓冲区大小，举例缓冲区只有0.2s，但是play_time为0.5
        self.buffer_size = np.maximum(self.buffer_size - play_time, 0.0) # 更新真实播放后的缓冲区，最小值为0
        return self.play_timeline, buffer  # 返回当前播放时间线和剩余缓冲区大小,这个bufer有助于帮助判断是否卡顿，播放视频后真实的剩余缓冲区大小还是看buffer_size
    


