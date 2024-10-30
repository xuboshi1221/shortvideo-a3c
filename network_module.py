# shared by all players #这段代码定义了一个 Network 类，用于模拟网络环境中视频块的下载过程。
MILLISECONDS_IN_SECOND = 1000.0  # 每秒的毫秒数，用于将时间单位从秒转换为毫秒
PACKET_PAYLOAD_PORTION = 0.95   # 数据包中有效载荷的比例，考虑到协议头等开销
LINK0_RTT = 80  # millisec   # 链路往返时延 (Round Trip Time)，单位为毫秒
LINK1_RTT = 60  # 
PACKET_SIZE = 1500  # bytes  # 数据包大小，单位为字节
B_IN_MB = 1000000.0   # 1MB等于1000000字节，用于将带宽从Mbps转换为B/s
BITS_IN_BYTE = 8.0  # 1字节等于8比特

class Network:   
    def __init__(self, cooked_time, cooked_4gbw, cooked_wifibw, paths):  # 初始化函数，传入经过的时间序列和4g带宽，wifi带宽，以及选择的路径0代表4g,1代表wifi
        assert len(cooked_time) == len(cooked_4gbw)  # 检查时间序列和带宽序列长度是否一致
        assert len(cooked_time) == len(cooked_wifibw)
        self.cooked_time = cooked_time  # 存储经过的时间序列
        self.cooked_4gbw = cooked_4gbw    # 存储4g带宽序列
        self.cooked_wifibw = cooked_wifibw    # 存储wifi带宽序列
        self.paths=paths   #默认使用4g下载
        self.mahimahi_ptr = 1   # 初始化指针(pointer)，用于遍历带宽数据
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]  # 记录上一次模拟网络数据传输时的时间点

    # 计算下载某个视频块所需的时间
    def network_simu(self,video_chunk_size,paths):  
        delay = 0.0  # in s   # 初始化延迟时间，单位为秒
        video_chunk_counter_sent = 0  # in bytes   已发送的视频块数据大小，单位为字节
       
        while True:  # download video chunk over mahimahi  # 开始模拟下载过程。
            throughput4g = self.cooked_4gbw[self.mahimahi_ptr] * B_IN_MB / BITS_IN_BYTE       # B/s  #计算当前时刻的带宽，原先的单位是Mbps，转换为B/s (字节每秒)
            throughputwifi = self.cooked_wifibw[self.mahimahi_ptr] * B_IN_MB / BITS_IN_BYTE   # B/s  #计算当前时刻的带宽，原先的单位是Mbps，转换为B/s (字节每秒)
            duration = self.cooked_time[self.mahimahi_ptr] - self.last_mahimahi_time    # s   # 计算当前时间片段的持续时间，单位为秒,应该都是0.5秒
            # 计算在该持续时间内可以发送的有效数据量，考虑数据包的有效载荷比例
            if paths==0:
                packet_payload = throughput4g * duration * PACKET_PAYLOAD_PORTION  # B
            else:
                packet_payload = throughputwifi * duration * PACKET_PAYLOAD_PORTION  # B
            # 检查是否已经足够发送整个视频块
            if video_chunk_counter_sent + packet_payload > video_chunk_size:  # B  # 如果在未来0.5秒内可以发送完视频
                if paths==0:
                    fractional_time = (video_chunk_size - video_chunk_counter_sent) / throughput4g / PACKET_PAYLOAD_PORTION  # 计算一下具体多少时间发送完毕,s
                else:
                    fractional_time = (video_chunk_size - video_chunk_counter_sent) / throughputwifi / PACKET_PAYLOAD_PORTION  # 计算一下具体多少时间发送完毕
                delay += fractional_time    # s  # 累加发送部分的时间 (秒)
                self.last_mahimahi_time += fractional_time  # s  #  更新最后的时间戳
                break   # 完成该视频块的下载，退出循环
            video_chunk_counter_sent += packet_payload  # B    # 如果视频块还没有发送完，则更新已发送的数据量和时间。
            delay += duration  # s   
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]  # 更新最后的时间戳
            self.mahimahi_ptr += 1   # 移动指针到下一个时间点
            if self.mahimahi_ptr >= len(self.cooked_4gbw): # 如果指针超过了带宽字段的长度，网络带宽trace太短，不足以观看完毕视频，重新循环利用trace
                self.mahimahi_ptr = 1  # 回到开头
                self.last_mahimahi_time = 0   # 时间重置为0

        delay *= MILLISECONDS_IN_SECOND  # 将延迟从秒转换为毫秒
        if paths==0:    # 再加上往返时延 (RTT)
            delay += LINK0_RTT
        else:
            delay += LINK1_RTT

        return delay  # ms 返回一个延迟值，单位为毫秒，代表下载完这个视频块需要多长时间。