# input: download_video_id, bitrate, sleep_time
# output: info needed by schedule algorithm
# buffer: ms
#这段代码是一个模拟视频播放环境的Python脚本，它涉及到用户行为、网络条件、视频播放等多个方面。

import numpy as np
import math
from numpy.lib.utils import _split_line
from .video_player import Player              #模拟视频播放器的行为，播放时间线等
from .user_module import Retention             #模拟用户保留率行为，生成用户退出的时间点。
from .network_module import Network            #模拟网络带宽和传输延迟，用于下载块。

USER_FILE = 'logs/sample_user/user.txt'   #定义一个全局变量USER_FILE，用于存储用户数据。
# user_file = open(USER_FILE, 'wb')   # a追加写入模式，w覆盖写入模式,wb二进制覆盖写入。
LOG_FILE = 'logs/log.txt'                 #定义一个全局变量LOG_FILE，用于记录日志。
# log_file = open(LOG_FILE, 'w')  # a追加写入模式，w覆盖写入模式。
NEW = 0   #定义两个常量，用于表示添加新视频和删除视频的操作。
DEL = 1
PLAYER_NUM = 5    #定义一个常量，表示播放器的数量。此处代表算上当前播放的视频最多预加载五个视频

class Environment:  #定义一个名为Environment的类，用于模拟视频播放环境。
    def __init__(self, user_sample_id, all_cooked_time, all_cooked_4gbw,all_cooked_wifibw,paths, video_num, seeds): #定义Environment类的初始化函数。
        # global USER_FILE
        # USER_FILE = 'logs/sample_user/user_'+str(user_sample_id)+'.txt'   #根据用户样本ID修改用户文件路径。
        # self.user_file = open(USER_FILE, 'wb')   #以二进制写模式打开用户文件,覆盖写入。
        # global LOG_FILE
        # self.log_file = open(LOG_FILE, 'w')  #a追加写入模式，w覆盖写入模式。   
        print("初始化完成")

    def reset(self, user_sample_id, all_cooked_time, all_cooked_4gbw,all_cooked_wifibw,paths, video_num, seeds):  #我添加的reset函数。
        global USER_FILE
        USER_FILE = 'logs/sample_user/user_'+str(user_sample_id)+'.txt'   #根据用户样本ID修改用户文件路径。
        self.user_file = open(USER_FILE, 'wb')   #以二进制写模式打开用户文件,覆盖写入。
        global LOG_FILE
        self.log_file = open(LOG_FILE, 'w')  #a追加写入模式，w覆盖写入模式。

        self.wasted_bytes = 0   #浪费的数据量。
        self.players = []  #初始化一个空列表，用于存储Player对象。
        self.seeds = seeds  #将传入的种子值赋给实例变量seeds。
        self.user_models = []  # 初始化一个空列表，记录当前视频的用户动作(retention类)即用户滑动时间，与players同步更新。
        self.video_num = video_num   #将传入的视频数量赋给实例变量video_num。7个视频
        self.video_cnt = 0   #初始化视频计数器。
        self.play_video_id = 0  #初始化正在播放的视频ID,0-6。
        self.network = Network(all_cooked_time, all_cooked_4gbw, all_cooked_wifibw, paths)  #创建一个Network对象，自动调用init函数，用于模拟网络条件,添加了wifi路径和选路的变量。
        self.timeline = 0.0   #初始化时间线变量。
        self.total_watched_len = 0.0     #初始化观看总时长
        self.total_downloaded_len = 0.0   #初始化已下载总时长。
        self.all_videos_done=0
        self.swipe=0
        for p in range(PLAYER_NUM):   #循环创建指定数量的Player对象，p依次取0-4
            self.players.append(Player(p))   #将创建的Player对象添加到players列表中。自动调用init函数初始化播放状态和缓冲区，以及视频块大小等。可以代表一个视频的所有属性
            user_time, user_retent_rate = self.players[-1].get_user_model()  #根据Player对象获取用户播放信息（观看时间和保留率）。
            self.user_models.append(Retention(user_time, user_retent_rate, seeds[self.video_cnt])) #根据观看时间和保留率创建一个Retention对象并添加到user_models列表中。
            self.user_file.write((str(self.user_models[-1].get_ret_duration()) + '\n').encode())   #使用Retention类的函数获取用户停留时长并写入用户文件。
            self.user_file.flush()  #刷新用户文件，确保写入。     
            self.total_watched_len += self.user_models[-1].get_ret_duration()  # 更新总观看时长（若干个视频的总观看时长）。
            self.video_cnt += 1   #视频计数器加1           
        self.start_video_id = 0   #开始视频id 0-6
        self.end_video_id = PLAYER_NUM - 1      #结束视频id 0-6
        #下面这几个status环境状态才是reset的关键
        delay=0
        video_size=0
        self.end_of_video=0
        first_step=1   
        result_array = np.array([delay,video_size,self.end_of_video,self.play_video_id,len(self.players),first_step,
                                 self.players[0].get_buffer_size()/1000,self.players[1].get_buffer_size()/1000,self.players[2].get_buffer_size()/1000,self.players[3].get_buffer_size()/1000,self.players[4].get_buffer_size()/1000,
                                 self.players[0].get_remain_video_num(),self.players[1].get_remain_video_num(),self.players[2].get_remain_video_num(),self.players[3].get_remain_video_num(),self.players[4].get_remain_video_num()], dtype=np.float32)
        return result_array
    
    def player_op(self, operation): #定义一个函数，用于添加或删除视频。
        if operation == NEW:   #如果操作是添加新视频。
            print('--------------ADD new video--------------')
            if self.video_cnt >= self.video_num:  # 如果视频计数器大于等于视频总数，无法添加新视频，直接返回。
                return
            self.players.append(Player(self.video_cnt))   #添加一个新的Player对象,Player[5]其实是添加的是第六个视频      
            user_time, user_retent_rate = self.players[-1].get_user_model()   #获取最后一个新添加的短视频的用户模型。           
            self.user_models.append(Retention(user_time, user_retent_rate, self.seeds[self.video_cnt]))   #调用Retention类根据时间和保留率生成具体观看时间。
            self.user_file.write((str(self.user_models[-1].get_ret_duration()) + '\n').encode())    #将新的用户保留时长写入用户文件。
            self.user_file.flush()  #刷新用户文件，确保写入。 
            self.total_watched_len += self.user_models[-1].get_ret_duration()  # 更新总观看时长。
            self.video_cnt += 1     #更新视频计数器。  
            self.end_video_id += 1        #更新末尾视频ID，start_video_id和end_video_id用视频编号0-6代表1-7视频  
        else:
            print('--------------DEL old video--------------')
            self.players.remove(self.players[0])   #移除第一个Player对象。后面对象的下标会自动前提，原先下标为1的对象自动把下标变为0
            self.user_models.remove(self.user_models[0])  #移除第一个Retention对象，后面对象的下标会自动前提，原先下标为1的对象自动把下标变为0.
    
    def play_videos(self, action_time,download_video_id):  #定义一个函数，用于播放视频,action_time也就是step函数中的delay，下载花了多长时间与此同时也就播放了多少时间。
        wasted_bw = 0
        buffer = 0
        # Continues to play if all the following conditions are satisfied:
        # 1) there's still action_time len# 2) the last video hasn't caused rebuf# 3) the video queue is not empty (will break inside the loop if its already empty)
        while buffer >= 0 and action_time > 0:  #当缓冲区不为负且还有剩余播放时间时，继续播放。
            timeline_before_play = self.players[self.play_video_id].play_timeline       #获取当前视频的播放时间线。 players列表下标总是0-4
            video_remain_time = self.user_models[self.play_video_id].get_ret_duration() - timeline_before_play   #计算当前视频的剩余可播放时间。
            max_play_time = min(action_time, video_remain_time) #计算最大可播放时间：17秒的视频，已知用户只会播放到12.3秒，此时播放到了10秒，剩余2.3秒，下载一个块花了0.5秒，此时max_play_time=0.5
            timeline_after_play, buffer = self.players[self.play_video_id].video_play(max_play_time)   #播放视频并更新播放后的时间线和缓冲区大小。
            actual_play_time = timeline_after_play - timeline_before_play   #计算实际播放时间，考虑到可能缓冲区不够而没有播放完毕。
            action_time -= actual_play_time  #更新剩余播放时间。
            if actual_play_time == video_remain_time:   #如果实际播放时间等于视频的剩余可播放时间，代表此视频播放完毕。
                self.end_of_video = 1   #当前视频因用户滑动退出。
                # 输出视频编号和总长度，以及用户观看长度和下载长度，
                print("\nUser stopped watching Video ", self.play_video_id, "( ", self.players[self.play_video_id].get_video_len(), " ms ) :")
                print("User watched for ", self.user_models[self.play_video_id].get_ret_duration(), " ms, you downloaded ", self.players[self.play_video_id].get_chunk_counter()*1000, " ms.")   
                # 输出观看块的比特率和浪费带宽
                # video_qualities = []   #定义视频质量列表，用于保存该视频每个块选择的比特率
                # bitrate_cnt = min(math.ceil(self.players[self.play_video_id].get_play_chunk()), self.players[self.play_video_id].get_chunk_sum()) #用户播放了多少个块
                # for i in range(1, bitrate_cnt):  #遍历下载的视频块。
                #     video_qualities.append(self.players[self.play_video_id].get_video_quality(i-1))  #添加视频质量到列表。             
                # video_qualities.append(self.players[self.play_video_id].get_video_quality(bitrate_cnt-1))
                #print("Your watched bitrates are: ", video_qualities)  #打印下载的视频质量。我感觉这打印的是用户观看的视频块的比特率而不是下载块的。
                #wasted_bw += self.players[0].bandwidth_waste(self.user_models[self.play_video_id]) #计算浪费的带宽，只有当用户退出一个视频才会计算整个视频的浪费带宽。
                #print("Your wasted_bw in this video: ", wasted_bw) 
                # 移动到下一个视频的头部进行播放
            #     self.player_op(DEL)       #删除当前视频。
            #     self.start_video_id += 1      #更新起始视频ID。start_video_id和end_video_id用视频编号0-6代表1-7视频
            #     self.player_op(NEW)       #添加新视频。
                self.play_video_id += 1   #更新播放视频ID。      
            if self.play_video_id >= self.video_num:   #如果播放完所有5个视频。
                self.all_videos_done=1   #用户结束观看
                print("played out!")  #打印播放完毕信息。
                break
        if buffer < 0:  # 如果缓冲区为负，代表卡顿了。设置buff为负的action_time,注意此时的action_time在上面已经减去实际播放时间了，所以代表实际的卡顿时长。
            buffer = (-1) * action_time  # rebuf time is the remain action time(cause the player will stuck for this time too) 
        return buffer, wasted_bw
              
    def step(self, download_video_id, paths):  #原先的函数名是buffer_management，包含三个形参：下载的视频，比特率和睡眠时间
        buffer = 0  #当前缓冲区的时间长度。
        rebuf = 0   #重新缓冲的时间。
        delay = 0    #下载视频时的延迟。
        video_size = 0    #上个下载块的大小。 
        done=0      
        print("Download Video ", download_video_id, " chunk (", self.players[download_video_id].get_chunk_counter()+1, " / ",
              self.players[download_video_id].get_chunk_sum(), ") and path is",paths," , play_video_id is ",self.play_video_id,file=self.log_file)   
                  
        video_size = self.players[download_video_id].get_video_size(2)  #根据比特率计算视频块大小，不用具体选择下载那个块,Player类会自动处理下载未下载的块。
        self.players[download_video_id].record_download_bitrate(2)   #记录下载的视频比特率。
        delay = math.floor(self.network.network_simu(video_size,paths)+0.5)  # ms 调用network_simu函数模拟网络下载延迟，加0.5再用floor函数代表对原数据四舍五入到最接近的毫秒整数。          
        buffer, wasted = self.play_videos(delay,download_video_id)  #调用play_videos函数模拟视频播放。 把delay传入action_time,意味着在下载视频块期间，播放器也会播放相应时间长度的视频内容。
        self.total_downloaded_len += 1000  # 更新总下载长度，加上1000毫秒。

        if self.players[download_video_id].get_chunk_counter()+1 == self.players[download_video_id].get_chunk_sum(): 
            self.end_of_video = 1   #当前视频因下载完毕而结束              
        else:  #否则用户还在正常下载和播放
            self.end_of_video = self.players[download_video_id-self.start_video_id].video_download(1000) #当前视频因所有的块下载完毕而结束。
        if self.play_video_id == self.video_num:  # if user leaves  用户看完最后一个视频。
            self.all_videos_done=1   

        self.wasted_bytes += wasted   #更新总浪费的数据量。 
        first_step=0     
        if buffer < 0:  #如果缓冲区小于0，表示发生了重新缓冲。
            rebuf = abs(buffer)  #设置重新缓冲的时间为缓冲区的绝对值。
        reward=rebuf   #奖励函数
        done=self.all_videos_done  #所有视频结束标志  
        delay=delay/1000   #s
        video_size=video_size/1000000   #MB
        s_prime = np.array([delay,video_size,self.end_of_video,self.play_video_id,len(self.players), first_step,
                                 self.players[0].get_buffer_size()/1000,self.players[1].get_buffer_size()/1000,self.players[2].get_buffer_size()/1000,self.players[3].get_buffer_size()/1000,self.players[4].get_buffer_size()/1000,
                                 self.players[0].get_remain_video_num()-1,self.players[1].get_remain_video_num()-1,self.players[2].get_remain_video_num()-1,self.players[3].get_remain_video_num()-1,self.players[4].get_remain_video_num()-1], dtype=np.float32)
        return s_prime,reward,done
          
    def close(self):     #我添加的reset函数。
        # 关闭用户文件，用户看了多少毫秒
        if self.user_file:
            self.user_file.close()
        # 关闭日志文件，用户对于某个块选了什么比特率，平滑度损失，播放到多少毫秒
        if self.log_file:
            self.log_file.close()
        # 清空播放器
        self.players = []      
        # 打印结束信息
        print("Environment closed.")