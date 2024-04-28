'''
Soft Actor-Critic version 2
using target Q instead of V net: 2 Q net, 2 target Q net, 1 policy net
add alpha loss compared with version 1
paper: https://arxiv.org/pdf/1812.05905.pdf
要点：1.输出随机策略, 泛化性强（和之前PPO的思路一样, 其实是正确的）
2.基于最大熵, 综合评价奖励值与动作多样性（-log p(a\s)表示熵函数, 单个动作概率越大, 熵函数得分越低；为了避免智能体薅熵函数分数（熵均贫化）, 
可以动态改变熵函数系数（温度系数）, 先大后小, 实现初期注重探索后期注重贪婪
3.【重点】随机策略同样是网络生成方差+均值, 和“以为PPO能做的”一样, 但是必须用重参数化, 即不直接从策略分布中取样, 而是从标准正态分布
N(0,1)中取样i, 与网络生成的方差+均值mu,sigma得到实际动作a=mu+sigma*i, 这样保留了参数本身, 才能利用链式法则求出loss相对参数本身的梯度；
如果直接取样a, a与参数mu和sigma没有关系, 根本没法求相对mu和sigma的梯度；之前隐隐觉得之前的PPO算法中间隔了个正态分布所以求梯度这一步存在问题其实是对的...
4.目前SAC实现的算法（openAI和作者本人的）都用了正态分布替代多模Q函数, 如果想用多模Q函数需要用网络实现SVGD取样方法拟合多模Q函数（也是发明人在原论文中用的方法(Soft Q-Learning不是SAC））
'''

import datetime
import numpy as np
import itertools
import torch.nn as nn
from sac import SAC
#from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
import env 
import time
import matplotlib.pyplot as plt
import numpy as np
from Orbit_Dynamics.CW_Prop import Free

# 字典形式存储全部参数
args={'policy':"Gaussian", # Policy Type: Gaussian | Deterministic (default: Gaussian)
        'eval':True, # Evaluates a policy a policy every 10 episode (default: True)
        'gamma':0.99, # discount factor for reward (default: 0.99)
        'tau':0.1, # target smoothing coefficient(τ) (default: 0.005) 参数tau定义了目标网络软更新时的平滑系数, 
                     # 它控制了新目标值在更新目标网络时与旧目标值的融合程度。
                     # 较小的tau值会导致目标网络变化较慢, 从而增加了训练的稳定性, 但也可能降低学习速度。
        'lr':0.0003, # learning rate (default: 0.0003)
        'alpha':0.2, # Temperature parameter α determines the relative importance of the entropy\term against the reward (default: 0.2)
        'automatic_entropy_tuning':False, # Automaically adjust α (default: False)
        'batch_size':512, # batch size (default: 256)
        'num_steps':1000, # maximum number of steps (default: 1000000)
        'hidden_sizes':[1024,512,512,256], # 隐藏层大小, 带有激活函数的隐藏层层数等于这一列表大小
        'updates_per_step':1, # model updates per simulator step (default: 1) 每步对参数更新的次数
        'start_steps':1000, # Steps sampling random actions (default: 10000) 在开始训练之前完全随机地进行动作以收集数据
        'target_update_interval':10, # Value target update per no. of updates per step (default: 1) 目标网络更新的间隔
        'replay_size':10000000, # size of replay buffer (default: 10000000)
        'cuda':True, # run on CUDA (default: False)
        'LOAD PARA': False, #是否读取参数
        'task':'Train', # 测试或训练或画图, Train,Test,Plot
        'activation':nn.ReLU, #激活函数类型
        'plot_type':'2D-2line', #'3D-1line'为三维图, 一条曲线；'2D-2line'为二维图, 两条曲线
        'plot_title':'reward-steps.png',
        'seed':114514, #网络初始化的时候用的随机数种子  
        'max_epoch':50000,
        'logs':True} #是否留存训练参数供tensorboard分析 
                    
# Environment
# env = NormalizedActions(gym.make(args.env_name))
if args['logs']==True:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('./runs/')

sma = 7171e3
thrust = 1e-2
ref_p = 2e3
T_f = 2000
discreT = 50
Cd = np.eye(6)
R = np.eye(3)*1e6
Q11 = np.eye(3)*1e1
Q22 = np.eye(3)*2
S11 = np.eye(3)
S22 = np.eye(3)
horizon = 5

num_ff = 3 # 编队卫星数量
'''分配每个星的参考对象'''
ref_sat_distro = np.array([[1,2],[0,2],[0,1]])

env_f = env.Rel_trans(sma, thrust, ref_p, T_f, discreT,Cd,R,Q11,Q22,S11,S22, horizon)
# Agent
agent = SAC(env_f.observation_space.shape[0], env_f.action_space, args) #discrete不能用shape, 要用n提取维度数量
#Tensorboard
'''创建一个SummaryWriter对象, 用于将训练过程中的日志信息写入到TensorBoard中进行可视化。
   SummaryWriter()这是创建SummaryWriter对象的语句。SummaryWriter是TensorBoard的一个API, 用于将日志信息写入到TensorBoard中。
   format括号里内容是一个字符串格式化的表达式, 用于生成一个唯一的日志目录路径。{}是占位符, format()方法会将占位符替换为对应的值
   datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")：这是获取当前日期和时间, 并将其格式化为字符串。
   strftime()方法用于将日期时间对象转换为指定格式的字符串。在这里, 日期时间格式为"%Y-%m-%d_%H-%M-%S", 表示年-月-日_小时-分钟-秒。
   "autotune" if args.automatic_entropy_tuning else ""：这是一个条件表达式, 用于根据args.automatic_entropy_tuning的值来决定是否插入"autotune"到日志目录路径中。
   'runs/{}_SAC_{}_{}_{}'是一个字符串模板, 其中包含了四个占位符 {}。当使用 format() 方法时, 传入的参数会按顺序替换占位符, 生成一个新的字符串。'''
#writer = SummaryWriter('runs/{}_SAC_{}_{}.txt'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),args['policy'], "autotune" if args['automatic_entropy_tuning'] else ""))
'''
显示图像：用cmd（不是vscode的终端） cd到具体存放日志的文件夹（runs）, 然后
    tensorboard --logdir=./
或者直接在import的地方点那个启动会话
如果还是不行的话用
    netstat -ano | findstr "6006" 
在cmd里查一下6006端口有没有占用, 用taskkill全杀了之后再tensorboard一下
    taskkill /F /PID 26120
'''

# Memory
memory = ReplayMemory(args['replay_size'])

def multi_step(agent,env_f,memory, travel, Multi_Agent_state, Multi_dumm_state, Multi_target_state_ini, dep_param):
    sc_m = 1e-3       
    # 传入的常数 dep_param = [horizon, num_ff ,ref_sat_distro, heading_ex, dT_ex, T_f_ex]  
    horizon = dep_param[0]
    num_ff = dep_param[1]
    ref_sat_distro = dep_param[2]
    heading_ex = dep_param[3]
    dT_ex = dep_param[4]
    T_f_ex = dep_param[5]


    s_len = 6*horizon # 智能体数据间的间隔, 意思就是每个智能体有多少步
    r_len = 6         # 时间相邻两个状态的数据间隔
    Sat_inject_flag = np.zeros((3,1))  # 三个智能体的入轨状态
    Sat_done_flag = np.zeros((3,1))  # 三个智能体的完成状态
    '''提前开辟存储空间,为了在所有智能体执行完动作后, 统一统计动作reward导致的不能在3号智能体还没算完的时候
    你就把memorypush进去了, 你得等3个全部算完, 你就必须得把next——state啥的这些玩意存下来'''
    data_saver4push = np.empty((num_ff, 7), dtype=object)

    '''临时变量, 存放每个agent在每个step的参考星的观测光锥'''
    # 当前的轨道序列 每个智能体, 参考包括自己在内的3个智能体位置进行决策
    multi_agent_seq = np.zeros((3*6*horizon,))
    # 无智能体干预的轨道序列
    # multi_dummy_seq = np.zeros((3*6*horizon,)) # 0425 这玩意没有意义，预测用不上，规划动作用不上，输出奖励用不上
    # 目标轨道的序列
    multi_target_seq = np.zeros((3*6*horizon,)) 
    next_Multi_Agent_state = np.zeros((num_ff,6))
    next_Multi_target_state = np.zeros((num_ff,6))
    next_Multi_dumm_state = np.zeros((num_ff,6)) 

    # 自己把自己坑了，输入是一维，那从此定个规则吧，有大写字母是矩阵，小写是一维 0472241
    input_multi_dumm_state = np.zeros((3*6,))

    next_Multi_Agent_pos = np.zeros((num_ff,3))
    next_Multi_dummy_pos = np.zeros((num_ff,3))

    for sat_index in range(num_ff):
        # 下面整理输入
        # 读取agent的参考, 例如, 1号agent要参考6,3,那么序列的顺序就是1,6,3
        local_ref = np.array([sat_index, *ref_sat_distro[sat_index,:]])

        for sat_i, ref_i in zip(range(3), local_ref):
            for ri in range(0,horizon):
                # 预测序列是按这样布置的, 先把第一个智能体的1-5步放完, 再把第二个智能体的1-5步放完, 再放第三个智能体的1-5步
                #   sat1 t1-t5, sat2 t1-t5, sat3 t1-t5
                t_r = (ri+1)*dT_ex
                # 这种就是按照 先排完一个智能体的全部数据, 再排另外一个智能体
                index_start = sat_i*s_len + ri*r_len  
                index_end = index_start + 6
                multi_agent_seq[index_start:index_end] = Free(t_r,0,Multi_Agent_state[ref_i,:], 1)
                # target 的外推原点一直是公元0年 所以加一个 dT_ex
                multi_target_seq[index_start:index_end] = Free(dT_ex + t_r,0, Multi_target_state_ini[ref_i,:], 1)
           
            '''我去，这个dumm是 来自每个真实智能体 在上一步输出的 next_Multi_dumm_state 构成的，然后
             在input里取出自己想要的即可，这样分立地都实现了MPC外推，我真太牛逼了 '''
            input_multi_dumm_state[sat_i*6:sat_i*6+6] = Multi_dumm_state[ref_i,:]

        ''' 1e3是为了 放缩, 让状态在01之间
        顺便说一下, 区分state_env_input和state, 一个是个智能体用的, 一个是整个场景包括7个智能体用的
        #   下面这个是输入给agent的 所以 要放缩 
        '''

        # 这个地方 dumm的形状不对 不是一维....
        state_env_input = np.concatenate((multi_agent_seq*sc_m, multi_target_seq*sc_m, heading_ex, np.array([travel])))
        action = agent.select_action(state_env_input)  # 开始输出actor网络动作
        # 注意，只有state_env里的东西是除以了1000的，其他变量照常输出
        next_state_env, next_dumm_state, next_agent_pos, next_dummy_pos, J_f_agent,J_f_dummy,reward, dV = env_f.step(action, state_env_input,input_multi_dumm_state) # Step
        '''涉及到太多运算了, 把状态都提出来算了，next_state_env里是放缩过的,   没放缩:next_agent_pos, next_dummy_pos  '''
        '''
        # Ignore the "done" signal if it comes from hitting the time horizon.
        由于智能体只学一步, 所以memory只记录一步的
         即使状态里80%的东西都被我扔掉了, 不作为实际的运动, 他们也是有用的, 在网络里用于校准Q网络
        由于这里算出来的奖励没有用, 因为参考星的位置都是估计出来的, 不是真实的, 得所有星都算完一遍之后, 再评估
        传入放缩过变量
        ''' 
        data_saver4push[sat_index,:] = [action, state_env_input, next_state_env, J_f_agent, J_f_dummy, reward, dV]
        ''' 传递给下一个时刻的状态(while级别的下一个循环）, 有且仅有 第n个卫星的 一个 step时间之后 的状态 1*6'''
        '''读出next_state_en的状态时候，需要反放缩'''
        '''与内部157行左右区分！，这里state只读出 1个星（真实星）的状态 6个 , 不同星是用行来区分的\需要反放缩'''
        next_Multi_Agent_state[sat_index,:] = next_state_env[0:6]/sc_m
        next_Multi_target_state[sat_index,:] = next_state_env[3*horizon*6:3*horizon*6 + 6]/sc_m
        '''只有第一个智能体的dumm状态是需要的  不需要放缩'''
        next_Multi_dumm_state[sat_index,:] = next_dumm_state[0:6]
        '''仅用于计算面积奖励 读出pos的时候 不需要放缩
            只有第一个智能体和智障体 的pos状态是需要的'''
        next_Multi_Agent_pos[sat_index,:] = next_agent_pos[0,:]
        next_Multi_dummy_pos[sat_index,:] = next_dummy_pos[0,:]

    # 哪个智能体的时间都tm一样，随便抓一个就行 TRAVEL的定义域是[0,1]
    travel += dT_ex/T_f_ex

    # 开始计算奖励
    ''' 1、先计算heading_reward_all 有点尴尬的是, 我这个heading area只计算三个星的....后面4,5,6星得改改 '''
    heading_reward_all = env_f.calculate_area(next_Multi_Agent_pos) - env_f.calculate_area(next_Multi_dummy_pos)

    # 再依次push数据
    isInject = 0
    isDone = 0
    reward_all = 0
    for sat_index in range(num_ff):
        '''[action, state_env_input, next_state_env, J_f_agent, J_f_dummy, reward, dV]'''
        J_indv_agent = data_saver4push[sat_index,3]
        J_indv_dummy = data_saver4push[sat_index,4]
        '''内部奖励有没有意义呢, 应该还是有的'''
        reward_pseudo = data_saver4push[sat_index,5]
        dV_indv = data_saver4push[sat_index,6]
        '''2、计算入轨奖励, 由于智能体分别动作, 所以done与否其实只该评价一个智能体''' 
        exc_t_puni = 0.0
        exc_t_rewa = 1.0
        if np.linalg.norm(J_indv_dummy)<1:
            exc_t_puni = 1.0
        if np.linalg.norm(J_indv_dummy)<1e-1:
            exc_t_rewa = 0.0
        # 仿照内部
        reward_push = exc_t_rewa * (1 - travel**(1/2)) *heading_reward_all - exc_t_puni*1e-3*travel**(1/5)*(
            dV_indv + 3*(np.linalg.norm(J_indv_agent)-np.linalg.norm(J_indv_dummy))) + reward_pseudo*0.2
        
        if np.linalg.norm(J_indv_agent)-np.linalg.norm(J_indv_dummy)>100 and np.linalg.norm(J_indv_dummy)<1:
            reward_push = -1e3

        # 收敛极限
        Conv_tol = 5
        done_push = False
        if travel >= 1.0 or np.linalg.norm(J_indv_agent) <= Conv_tol:
            done_push = True

            if Sat_done_flag[sat_index,0] == 0: 
                    '''完成动作的智能体不再重复记录到达'''
                    isDone += 1
                    Sat_done_flag[sat_index,0] = 1
            
            if np.linalg.norm(J_indv_agent) > Conv_tol:
                isInject += 0 #用来区分越界和到达目标
                reward_push = 0
            elif np.linalg.norm(J_indv_agent) <= Conv_tol and travel <= 1.0:
                isInject += 1
                if Sat_inject_flag[sat_index,0] == 0: 
                    '''到达目标轨道的智能体不再重复获得奖励'''
                    reward_push += 10000
                    Sat_inject_flag[sat_index,0] = 1
        '''[action, state_env_input, next_state_env, J_f_agent, J_f_dummy, reward, dV]'''
        action_push = data_saver4push[sat_index,0]
        state_push = data_saver4push[sat_index,1]
        next_state_push = data_saver4push[sat_index,2]
        memory.push(state_push, action_push, reward_push, next_state_push, done_push) # Append transition to memory
        reward_all += reward_push
    # 完成一次外推
    All_Inject = False
    Alldone = False
    if isInject >= 3:
        All_Inject = True
    if isDone >= 3:
        Alldone = True
    return travel, next_Multi_Agent_state, next_Multi_dumm_state, All_Inject, Alldone, reward_all/num_ff     

if args['task']=='Train':
    # Training Loop
    updates = 0
    best_avg_reward=0
    total_numsteps=0
    steps_list=[]
    episode_reward_list=[]
    avg_reward_list=[]

    if args['LOAD PARA']==True:
        agent.load_checkpoint("sofarsogood.pt")
        best_avg_reward = 40

    for i_episode in itertools.count(1): #itertools.count(1)用于创建一个无限迭代器。它会生成一个连续的整数序列, 从1开始, 每次递增1。
        success=False
        episode_reward = 0
        Alldone=False
        episode_steps = 0

        '''start from here'''
        agent_A, dummy_A, target_A, heading_ex, dT_ex, T_f_ex = env_f.reset(ini_fin_in = [0.2,0.4,0.3,0.8],ini_phi = 0.0, seed = None)
        # reset之后得到的全是归一化的变量
        dep_param = np.array([horizon, num_ff ,ref_sat_distro, heading_ex, dT_ex, T_f_ex], dtype=object)
        # 这个只是得到了7颗卫星的初始状态, 没有光锥状态, 切记勿搞混（7是示例, 因为物理中没有7做常数
        '''每一轮的外推就是用这个状态'''
        # 智能体的单个状态，用于迭代记录
        Multi_Agent_state = np.zeros((num_ff,6))
        Multi_target_state_ini = np.zeros((num_ff,6))
        for sat_index in range(num_ff):            
            phase_diff = 2*np.pi/num_ff
            pseu_t = phase_diff*sat_index
            Multi_Agent_state[sat_index,:] = Free(pseu_t,0,agent_A,1)
            # 一直保持初始
            Multi_target_state_ini[sat_index,:] = Free(pseu_t,0,target_A,1)
        #     
        Multi_dumm_state = Multi_Agent_state
        # 进入外推
        # 外推嵌套关系：1、时间（2、挨个agent(3、agent参考的3个agent 挨个录入（4、每个被参考的agent计算horizon光锥）))
        travel = 0 
        '''开始外推, 每轮只用更新Multi_Agent_state, 因为 Multi_target_seq 永远从初始时刻 Multi_target_state_ini 推得'''
        while True: 
            # new 和 next在这里代表的变量含义相同，区分内部外部
            travel, new_Multi_Agent_state, new_Multi_dumm_state, All_Inject, Alldone,reward_ave_all = multi_step(agent,env_f,memory, travel, Multi_Agent_state, Multi_dumm_state, Multi_target_state_ini, dep_param)
            Multi_Agent_state = new_Multi_Agent_state
            Multi_dumm_state = new_Multi_dumm_state
            
            ''''end here'''
            episode_steps += 1
            episode_reward += reward_ave_all # 没用gamma是因为在 sac 里求q的时候用了
            total_numsteps+=1
            if len(memory) > args['batch_size']:
                # Number of updates per step in environment 每次交互之后可以进行多次训练...
                for i in range(args['updates_per_step']):
                    # Update parameters of all the networks
                    # print("i",i)
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args['batch_size'], updates)
                    if args['logs']==True:
                        writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                        writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                        writer.add_scalar('loss/policy', policy_loss, updates)
                        writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                        writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1
            if Alldone:
                #env.plot(i_episode,episode_steps)
                steps_list.append(i_episode)
                episode_reward_list.append(episode_reward)
                if len(episode_reward_list)>=500:
                    avg_reward_list.append(sum(episode_reward_list[-500:])/500)
                else:
                    avg_reward_list.append(sum(episode_reward_list)/len(episode_reward_list))
                if All_Inject >= num_ff:
                    success=True
                break

        if args['logs']==True:
            writer.add_scalar('reward/train', episode_reward, i_episode)
        #writer.add_scalar('reward/train', episode_reward, i_episode)
        print("Episode: {}, episode steps: {}, reward: {}, success: {}".format(i_episode, episode_steps, round(episode_reward, 4), success))
        # round(episode_reward,2) 对episode_reward进行四舍五入, 并保留两位小数

        if i_episode % 100 == 0 and args['eval'] is True: #评价上一个训练过程
            avg_reward = 0.
            episodes = 10
            if args['LOAD PARA']==True:
                episodes = 50
            done_num=0
            for _  in range(episodes):
                '''start'''
                # state = env_f.reset()
                # episode_reward = 0
                # Alldone = False
                # steps = 0
                # while not Alldone:
                #     action = agent.select_action(state, evaluate=False) #evaluate为True时为确定性网络, 直接输出mean
                #     next_state_env, reward, Alldone, All_Inject = env_f.step(action)

                agent_A, dummy_A, target_A, heading_ex, dT_ex, T_f_ex = env_f.reset(ini_fin_in = None,ini_phi = None,seed = None)
                # reset之后得到的全是归一化的变量
                dep_param = np.array([horizon, num_ff ,ref_sat_distro, T_f_ex, dT_ex], dtype=object)
                # 这个只是得到了7颗卫星的初始状态, 没有光锥状态, 切记勿搞混（7是示例, 因为物理中没有7做常数
                '''每一轮的外推就是用这个状态'''
                # 智能体的单个状态，用于迭代记录
                Multi_Agent_state = np.zeros((num_ff,6))
                Multi_target_state_ini = np.zeros((num_ff,6))
                for sat_index in range(0, num_ff):            
                    phase_diff = 2*np.pi/num_ff
                    pseu_t = phase_diff*sat_index
                    Multi_Agent_state[sat_index,:] = Free(pseu_t,0,agent_A,1)
                    # 一直保持初始
                    Multi_target_state_ini[sat_index,:] = Free(pseu_t,0,target_A,1)    
                Multi_dumm_state = Multi_Agent_state
                # 进入外推
                # 外推嵌套关系：1、时间（2、挨个agent(3、agent参考的3个agent 挨个录入（4、每个被参考的agent计算horizon光锥）))
                travel = 0 
                '''开始外推, 每轮只用更新Multi_Agent_state, 因为 Multi_target_seq 永远从初始时刻 Multi_target_state_ini 推得'''
                test_steps = 0
                while True: 
                    # new 和 next在这里代表的变量含义相同，区分内部外部
                    travel, new_Multi_Agent_state, new_Multi_dumm_state, All_Inject, Alldone, reward_ave_all= multi_step(agent,env_f,memory, travel, Multi_Agent_state, Multi_dumm_state, Multi_target_state_ini, dep_param)
                    Multi_Agent_state = new_Multi_Agent_state
                    Multi_dumm_state = new_Multi_dumm_state
                    '''end'''
                    episode_reward += reward_ave_all
                    test_steps +=1
                    if All_Inject:
                        done_num+=1
                    if Alldone:
                        print(test_steps)
                        break
                
                avg_reward += episode_reward

            avg_reward /= episodes
            #writer.add_scalar('avg_reward/test', avg_reward, i_episode)
            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {},完成数：{}".format(episodes, round(avg_reward, 4),done_num))
            print("----------------------------------------")
            if avg_reward>best_avg_reward:
                best_avg_reward=avg_reward
                agent.save_checkpoint('sofarsogood.pt')
            if done_num==episodes and avg_reward_list[-1]>300:
                agent.save_checkpoint("tri_agent.pt")
                env_f.plot(args, steps_list, episode_reward_list, avg_reward_list)
                break

        if i_episode==args['max_epoch']:
            print("训练失败, {}次仍未完成训练".format(args['max_epoch']))
            env_f.plot(args, steps_list, episode_reward_list, avg_reward_list)
            if args['logs']==True:
                writer.close()
            break

if args['task']=='Test':
    agent.load_checkpoint("tri_agent.pt")
    avg_reward = 0
    episodes = 100
    done_num=0
    for i  in range(episodes):
        state = env_f.reset()
        episode_reward = 0
        Alldone = False
        steps=0
        while not Alldone:
            action = agent.select_action(state, evaluate=False) #evaluate为True时为确定性网络, 直接输出mean
            next_state_env, reward, Alldone, All_Inject = env_f.step(action)
            episode_reward += reward
            state = next_state_env
            steps+=1
            if All_Inject:
                done_num+=1
            if Alldone:
                break
        avg_reward += episode_reward
        # print("EP_reward:",episode_reward)
    avg_reward /= episodes
    #writer.add_scalar('avg_reward/test', avg_reward, i_episode)
    print("----------------------------------------")
    print("Test Episodes: {}, Avg. Reward: {},完成数：{}".format(episodes, round(avg_reward, 4),done_num))
    print("----------------------------------------")

if args['task']=='Plot':
    agent.load_checkpoint('tri_agent.pt')
    done_num=0
    # Sat1_array,Sat2_array,Sat3_array = np.zeros((T_f/discreT,3))
    Satpos_array = np.zeros((3,np.ceil(T_f/discreT).astype(int),3))
    '''初末轨道'''
    in_fin_orbi = np.array([0.2,0.15,0.25,0.35])*np.pi
    '''相位'''
    ini_phi = 0.1*np.pi
    state = env_f.reset(in_fin_orbi, ini_phi)
    Alldone = False
    steps=0
    heading_reward_all = 0
    while not Alldone:
        #evaluate为True时为确定性网络, 直接输出mean
        action = agent.select_action(state, evaluate=True) 
        next_state_env, Alldone, All_Inject, state_list,hw = env_f.plotstep(action)
        print(steps)
        state = next_state_env
        heading_reward_all += hw
        steps+=1
        if Alldone:
            break
    print(heading_reward_all)
    tri_state = np.array(state_list)
    for i in range(tri_state.shape[0]):
        for sat_j in range(3):
            Satpos_array[sat_j,i,:] = tri_state[i,sat_j*6:sat_j*6+3]
        # Sat2_array.append(plot_data[i][6:9]/1000)
        # Sat3_array.append(plot_data[i][12:15]/1000)
    '''plot the ini and final orbit'''
    ini_orb_state = env_f.orbit_i0
    fin_orb_state = env_f.orbit_f0
    t_period = np.arange(0, 2*np.pi, 0.05)
    Xi_orbit = np.zeros((6, len(t_period)))
    Xf_orbit = np.zeros((6, len(t_period)))
    for j in range(0, len(t_period)):
        Xi_orbit[:, j] = Free(t_period[j], 0, ini_orb_state, 1)
        Xf_orbit[:, j] = Free(t_period[j], 0, fin_orb_state, 1)
    fig = plt.figure()
    ax = fig.gca(projection='3d') 
    ax.plot(Xi_orbit[0, :], Xi_orbit[1, :], Xi_orbit[2, :], 'k')
    ax.plot(Xf_orbit[0, :], Xf_orbit[1, :], Xf_orbit[2, :], 'r')
    for sat_j in range(3):    
        ax.plot(Satpos_array[sat_j,0:steps,0],Satpos_array[sat_j,0:steps,1],Satpos_array[sat_j,0:steps,2],'b',linewidth=1) #画三维图
    ax.plot(0,0,0,'r*') #画一个位于原点的星形

    plt.show()
    # (args,Sat1_array[:][0],Sat1_list[:][1],Sat1_list[:][2])
    # (args,Sat2_list[:][0],Sat2_list[:][1],Sat2_list[:][2])
    # (args,Sat3_list[:][0],Sat3_list[:][1],Sat3_list[:][2])
    #writer.add_scalar('avg_reward/test', avg_reward, i_episode)
    print("----------------------------------------")
    print("完成：{}".format(Alldone))
    print("----------------------------------------")