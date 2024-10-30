import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import time
import numpy as np
from simulator import controller5video
from simulator import short_video_load_trace
 
# 超参数
n_train_processes = 1   #训练进程的数量。
learning_rate = 0.00001  #学习率。
update_interval = 5     #更新间隔，表示每执行多少次环境步骤后更新一次网络。
print_interval = 20     #打印间隔，定义了每多少回合打印一次平均得分。
gamma = 0.98             #折扣因子，用于计算回报。
max_train_ep = 10     #训练时的最大回合数。
max_test_ep = 200      #测试时的最大回合数。
RAND_RANGE = 1000    #计算选块选路概率用到

#播放器超参数
RANDOM_SEED = 42  # the random seed for user retention  定义种子，种子的值决定了随机数生成器的初始状态
np.random.seed(RANDOM_SEED)     #设置随机数生成器，这个操作确保了随机数生成器在每次运行时都从相同的初始状态开始
seeds = np.random.randint(100, size=(7, 2))    #生成7行2列随机数，由于随机数生成器已经被设置了种子，所以每次执行这个操作时，生成的随机数数组都将是相同的。
ALL_VIDEO_NUM = 5  #视频数量
paths=0       #0为4g,1为wifi。默认为0，因为4g随时随地都可以用，wifi不行。

#定义ActorCritic类，这是一个演员-评论家网络，用于同时学习策略（动作概率）和价值函数。
class ActorCritic(nn.Module):
    def __init__(self): #初始化网络结构。先有输入层，然后进入策略网络，最后softmax激活
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(16, 256)   #输入层：16个状态特征，256个神经元  
        self.fc_pi_chunks = nn.Linear(256, 5) #策略块网络输出：选择5个块中的一个
        self.fc_pi_paths = nn.Linear(256, 2) #策略路网络输出：选择2条路中的一个
        self.fc_v = nn.Linear(256, 1)  #价值网络输出：状态值

    def pi_chunks(self, x, softmax_dim=0):   #定义策略网络，输出动作概率。
        x = F.relu(self.fc1(x))    # 接收状态向量x,经过全连接层提取，然后用激活函数ReLU输出
        x = self.fc_pi_chunks(x)          # 把激活后的状态向量x转移进输出层得到动作概率分布
        prob_chunks = F.softmax(x, dim=softmax_dim)  # 使用Softmax函数生成动作概率
        return prob_chunks
    
    def pi_paths(self, x, softmax_dim=0):   #定义策略网络，输出动作概率。
        x = F.relu(self.fc1(x))    # 调用全连接层fc1和激活函数ReLU
        x = self.fc_pi_paths(x)          # 调用输出层，得到动作概率分布
        prob_paths = F.softmax(x, dim=softmax_dim)  ## 使用Softmax函数生成动作概率
        return prob_paths

    def v(self, x):              #定义价值网络，输出状态的价值估计。
        x = F.relu(self.fc1(x))   # 调用全连接层fc1和激活函数ReLU
        v = self.fc_v(x)      # 把激活后的状态向量x转移进价值输出层得到v
        return v

def train(global_model, rank):   #定义train函数，用于训练模型,每5轮更新一次全局网络。
    local_model = ActorCritic()  # 定义局部模型
    local_model.load_state_dict(global_model.state_dict()) # 同步全局模型的参数
    optimizer = optim.Adam(global_model.parameters(), lr=learning_rate)   #定义Adam优化器。
    env = controller5video.Environment(0,all_cooked_time,all_cooked_4gbw,all_cooked_wifibw,paths,ALL_VIDEO_NUM,seeds)   #创建环境并初始化环境。

    for n_epi in range(max_train_ep):  #训练指定回合数，循环max_train_ep次，每次循环代表一个训练回合,即在该trace下跑完七个视频。
        n_epi=n_epi+1  #我不想从0开始，我想从1开始当做训练的第一轮。
        done = False
        s = env.reset(0,all_cooked_time,all_cooked_4gbw,all_cooked_wifibw,paths,ALL_VIDEO_NUM,seeds)  #重置环境，获取初始状态。
        while not done:  #只要5个视频没有看完就一直循环
            s_lst, a1_lst,a2_lst, r_lst = [], [], [], []     # 用于存储轨迹，status,action1,action2,reward
            for t in range(update_interval):  #也就是说每执行五次选块选路的操作更新一次网络权重或者七个视频播放完毕。
                print("状态输入:", s)
                prob_chunks = local_model.pi_chunks(torch.from_numpy(s).float())   # 选块动作概率分布
                prob_paths = local_model.pi_paths(torch.from_numpy(s).float())     # 选路动作概率分布  
                # temperature = 1.0  # 调高温度系数以鼓励更多的探索 
                # prob_chunks = F.softmax(local_model.pi_chunks(torch.from_numpy(s).float()) / temperature, dim=0)
                # prob_paths = F.softmax(local_model.pi_paths(torch.from_numpy(s).float()) / temperature, dim=0)
                for i in range(-5,0):                   
                    if s[i]==0:
                        prob_chunks[i] = 0.0
                for num in range(0,5):
                        if num < s[3]:
                            prob_chunks[num]= 0.0
                if prob_chunks.sum().item() > 0:
                    prob_chunks = prob_chunks / prob_chunks.sum()
                else:
                    print("所有块的概率都被置为0,结束当前循环")
                    done= True
                    break
                print("修正后的概率分布：",prob_chunks,prob_paths)
                # m1 = Categorical(prob_chunks)   #创建一个Categorical分布对象m，它根据prob_chunks中的概率分布来定义。 突出5个候选值
                # a1 = m1.sample().item()     # 从分布m中采样一个动作。sample()方法返回一个采样的动作索引，.item()方法将这个索引转换成一个Python数值。动作空间0-4
                # m2 = Categorical(prob_paths)  #创建分类分布对象，突出2个候选值
                # a2 = m2.sample().item()     # 采样动作 。动作空间0-1
                # a1 = prob_chunks.argmax().item()
                # a2 = prob_paths.argmax().item()
                # a1_cumsum = np.cumsum(prob_chunks)
                # a1 = (a1_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
                # a2_cumsum = np.cumsum(prob_paths)
                # a2 = (a2_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()    
                # 创建掩码，只选择非零概率的项
                mask = prob_chunks > 0
                filtered_probs = prob_chunks[mask]
                # 创建新的 `Categorical` 对象进行采样
                m1 = Categorical(filtered_probs)
                a1_sampled_index = m1.sample().item()
                # 根据掩码映射到原始索引
                a1 = torch.nonzero(mask)[a1_sampled_index].item()
                # 采样路径动作
                m2 = Categorical(prob_paths)
                a2 = m2.sample().item()  # 从 prob_paths 分布中采样   

                print("采样的动作",a1,a2)
                if s[5]==1:
                    a1=0
                    a2=0
                    s_prime, r, done = env.step(a1,a2)
                    print("无论第一步选择什么,自动变更为0,0操作")  #选择最前面的视频，用4g加载。
                else:
                    s_prime, r, done = env.step(a1,a2)  #将采样的动作a1应用于环境，环境根据这个动作进行状态转移，并返回新的状态。r代表rebuff，越小越好。 
                print(f"Episode {n_epi}, Step {t}, Done值是: {done}")
                s_lst.append(s)   #将当前的状态 s 添加到状态列表 s_lst 中
                a1_lst.append([a1])   # 记录动作，创建了一个包含动作 a1 的列表，并将其添加到动作列表 a1_lst 中。使用列表存储动作是为了保持动作的维度，这在后续处理时（例如，计算损失函数）是必要的。
                a2_lst.append([a2])   
                r_lst.append(r/100.0)   # 将获得的奖励 r 除以 100 并添加到奖励列表 r_lst 中
                s = s_prime    # 这行代码将环境返回的新状态 s_prime 赋值给当前状态变量 s。在每个时间步之后，当前状态需要更新为新状态，以便在下一个决策中使用。
                if done:      # break跳出每五步更新一次的for循环，同时done为真代表不再进入播放全部视频的while大循环
                    break

            s_final = torch.tensor(s_prime, dtype=torch.float) # 将环境返回的新状态 s_prime 转换为 PyTorch 张量 s_final
            R = 0.0 if done else local_model.v(s_final).item() # 如果done为True，表示当前回合结束，那么未来的回报为0。done为False，表示当前回合未结束，使用.item() 从单元素张量中提取 Python 数值，然后会使用价值函数local_model.v 来估计可能得到的回报。
            #r_lst 提供了即时奖励的原始数据，而R是基于这些数据和折扣因子gamma计算得到的累积回报估计，用于指导价值函数的学习。
            td_target_lst = []           # 初始化一个空列表 td_target_lst，用于存储每个时间步的TD目标值（用于估计状态价值函数或动作价值函数。TD目标值结合了当前状态的价值估计和未来奖励的预期）。
            for reward in r_lst[::-1]:   #从后向前遍历奖励列表 r_lst，使用奖励来更新R的估计值。r_lst[::-1] 表示反向迭代奖励列表。
                R = gamma * R + reward   #在每次迭代中，R 被更新为先前的 R 乘以折扣因子 gamma 加上当前迭代的奖励
                td_target_lst.append([R])    # 反向计算TD目标值
            td_target_lst.reverse() #将TD目标值列表td_target_lst 反转。因为在上一步中，列表是从后向前填充的，所以需要反转列表以恢复原始的时间顺序，以便与状态列表 s_lst 和动作列表 a1_lst、a2_lst 对齐。

            # 将状态，动作和TD目标值列表转换为张量,计算损失，反向传播，同步全局参数。
            #_batch,a2_batch, td_target = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a1_lst),torch.tensor(a2_lst), torch.tensor(td_target_lst)
            # 将状态，动作和TD目标值列表转换为numpy数组，然后再转换为张量
            s_batch = np.array(s_lst).astype(np.float32)
            a1_batch = np.array(a1_lst)
            a2_batch = np.array(a2_lst)
            td_target = np.array(td_target_lst)
            # 将numpy数组转换为张量
            s_batch = torch.from_numpy(s_batch)
            a1_batch = torch.from_numpy(a1_batch)
            a2_batch = torch.from_numpy(a2_batch)
            td_target = torch.from_numpy(td_target)

            print("td_target shape:", td_target.shape)
            print("v(s_batch) shape:", local_model.v(s_batch).shape)
            advantage = td_target - local_model.v(s_batch)   # 这行代码计算优势函数（Advantage Function），它衡量采取某个动作相比于平均情况能带来多少额外的回报。
            #td_target是TD目标值，代表未来的期望回报。local_model.v(s_batch) 是价值网络对状态 s_batch 的价值估计。优势函数用于指导策略更新，告诉模型在特定状态下采取特定动作相比于其他动作有多好。
            pi_chunks = local_model.pi_chunks(s_batch, softmax_dim=1)   # 重新计算策略，softmax_dim=1 指定了应用 softmax 函数的维度。
            pi_paths = local_model.pi_paths(s_batch, softmax_dim=1)   # 重新计算策略
            #这两行代码使用 gather 方法从概率分布中选择实际执行的动作 a1_batch 和 a2_batch 对应的概率。
            #gather 方法根据索引从输入张量中抽取值，这里用于获取实际采取动作的概率，这些概率将用于后续的策略梯度计算。
            pi_chunks_a1 = pi_chunks.gather(1, a1_batch)  # 选择实际执行的动作概率，是策略网络计算出的针对每个状态实际执行的选块动作 a1 的概率，用于后续计算损失。
            pi_paths_a2 = pi_paths.gather(1, a2_batch)    # 选择实际执行的动作概率，是策略网络计算出的针对每个状态实际执行的选路动作 a2 的概率，用于后续计算损失。
            #这两行代码定义了 损失函数（loss function），用于更新策略网络和价值网络。由两部分组成：一部分是用于策略网络的策略损失，另一部分是用于价值网络的价值损失。
            #通过调整动作的概率来优化策略网络，最终目标是最大化累计回报。后半部分损失的目的是通过最小化状态的价值估计和TD目标值之间的差距，从而优化价值网络。
            loss1 = -torch.log(pi_chunks_a1) * (-advantage).detach() + F.smooth_l1_loss(local_model.v(s_batch), td_target.detach())      # 损失函数
            loss2 = -torch.log(pi_paths_a2) * (-advantage).detach() + F.smooth_l1_loss(local_model.v(s_batch), td_target.detach())      # 损失函数

            optimizer.zero_grad()  #在每次迭代开始时，需要清零（重置）模型参数的梯度。这是因为在PyTorch中，梯度是累加的
            loss1.mean().backward()     # 反向传播 
            loss2.mean().backward()     # 反向传播
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)   #梯度裁剪
            # 同步全局参数，通过遍历局部模型和全局模型的参数，将局部模型的梯度复制到全局模型的对应参数中。
            for name, param in local_model.named_parameters():   #检查梯度是否包含nan空或者infinity无穷大
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"Gradient NaN or Inf in {name}")
            for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
                global_param._grad = local_param.grad
            optimizer.step()     # 更新全局模型参数
            local_model.load_state_dict(global_model.state_dict())   # 更新局部模型以便在下一次迭代时将使用最新的参数进行前向传播和损失计算。

    env.close()   # 关闭环境
    print("Training process {} reached maximum episode.".format(rank))


def test(global_model):  # 用测试进程的test函数每隔20轮打印一次得分 
    env = controller5video.Environment(0,all_cooked_time,all_cooked_4gbw,all_cooked_wifibw,paths,ALL_VIDEO_NUM,seeds)   #创建环境并初始化环境。
    score = 0.0    #用于记录智能体在测试过程中的累积得分。
    
    for n_epi in range(max_test_ep):    # 最大测试回合数,range不包括括号内的值，n_epi的第一次循环取0
        n_epi=n_epi+1  #我不想从0开始，我想从1开始当做训练的第一轮。
        done = False  #done 用于标记测试是否结束。初始值为 False，表示测试尚未结束。
        s = env.reset(0,all_cooked_time,all_cooked_4gbw,all_cooked_wifibw,paths,ALL_VIDEO_NUM,seeds)  #重置环境，获取初始状态。

        while not done:
            prob_chunks = global_model.pi_chunks(torch.from_numpy(s).float())  # 使用全局模型计算选择块的动作概率。
            prob_paths = global_model.pi_paths(torch.from_numpy(s).float())    # 使用全局模型计算选择路径的动作概率。
            a1 = Categorical(prob_chunks).sample().item()    # 从块选择的概率分布中采样一个动作。
            a2 = Categorical(prob_paths).sample().item()
            s_prime, r, done= env.step(a1,a2)    # 执行动作   
            s = s_prime  #将环境返回的新状态 s_prime 赋值给当前状态变量 s
            score += r    # 累加得分

        if n_epi % print_interval == 0 and n_epi != 0:   # 每隔20回合输出平均得分
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0   # 重置得分。
            time.sleep(1)   # 延时 1 秒。
    env.close()
    print("Testing process {} reached maximum episode.".format(rank))

#当一个Python文件被运行时，Python解释器会自动创建一些特殊的变量，name 就是其中之一。如果这个文件是作为主程序直接运行的，那么变量 name 的值会被设置为字符串main。
#如果这个文件是被其他Python文件导入的，那么name的值会被设置为该文件的模块名。
if __name__ == '__main__':   #确保代码块只在脚本直接运行时执行，而不是在导入时执行。
    cooked_trace_folder = 'data/network_traces/multipath/'
    global all_cooked_time, all_cooked_4gbw, all_cooked_wifibw
    all_cooked_time, all_cooked_4gbw, all_cooked_wifibw = short_video_load_trace.load_trace(cooked_trace_folder)

    global_model = ActorCritic()   # 定义全局模型
    global_model.share_memory()    # 将模型的内存共享给多个进程。

    processes = []  #用于存储创建的进程对象。
    # for rank in range(n_train_processes + 1):  # +1 for test process  # 启动n个训练进程和1个测试进程
    #     if rank == 0:
    #         p = mp.Process(target=test, args=(global_model,))          # 测试进程   
    #     else: 
    #         p = mp.Process(target=train, args=(global_model, rank,))   # 训练进程
    #     p.start()    # 启动进程
    #     processes.append(p) #将进程对象添加到列表中。
    for rank in range(n_train_processes):  #  启动n个训练进程
        p = mp.Process(target=train, args=(global_model, rank,))   # 训练进程,用两个训练进行调试，测试进程未启用。
        p.start()    # 启动进程
        processes.append(p) #将进程对象添加到列表中。
    for p in processes:
        p.join()  # join是一个进程同步方法，用于让主进程等待每个子进程执行完毕。主进程会等所有训练和测试进程完成任务后，才会继续执行接下来的代码。

    torch.save(global_model.state_dict(), './model/a3c_model.pth')
    print("model is saved to model directory")
    
