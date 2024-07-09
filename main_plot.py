'''
Soft Actor-Critic version 2
using target Q instead of V net: 2 Q net, 2 target Q net, 1 policy net
add alpha loss compared with version 1
paper: https://arxiv.org/pdf/1812.05905.pdf
要点：1.输出随机策略，泛化性强（和之前PPO的思路一样，其实是正确的）
2.基于最大熵，综合评价奖励值与动作多样性（-log p(a\s)表示熵函数，单个动作概率越大，熵函数得分越低；为了避免智能体薅熵函数分数（熵均贫化），
可以动态改变熵函数系数（温度系数），先大后小，实现初期注重探索后期注重贪婪
3.【重点】随机策略同样是网络生成方差+均值，和“以为PPO能做的”一样，但是必须用重参数化，即不直接从策略分布中取样，而是从标准正态分布
N(0,1)中取样i，与网络生成的方差+均值mu,sigma得到实际动作a=mu+sigma*i，这样保留了参数本身，才能利用链式法则求出loss相对参数本身的梯度；
如果直接取样a，a与参数mu和sigma没有关系，根本没法求相对mu和sigma的梯度；之前隐隐觉得之前的PPO算法中间隔了个正态分布所以求梯度这一步存在问题其实是对的...
4.目前SAC实现的算法（openAI和作者本人的）都用了正态分布替代多模Q函数，如果想用多模Q函数需要用网络实现SVGD取样方法拟合多模Q函数（也是发明人在原论文中用的方法(Soft Q-Learning不是SAC））
'''

import datetime
import numpy as np
import itertools
import torch.nn as nn
from masac import MASAC
#from torch.utils.tensorboard import SummaryWriter
from replay_mem_multi import ReplayMemory
import multi_env 
import time
import matplotlib.pyplot as plt
import numpy as np
from Orbit_Dynamics.CW_Prop import Free

# 字典形式存储全部参数
args={'policy':"Gaussian", # Policy Type: Gaussian | Deterministic (default: Gaussian)
        'eval':True, # Evaluates a policy a policy every 10 episode (default: True)
        'gamma':0.99, # discount factor for reward (default: 0.99)
        'tau':0.1, # target smoothing coefficient(τ) (default: 0.005) 参数tau定义了目标网络软更新时的平滑系数，
                     # 它控制了新目标值在更新目标网络时与旧目标值的融合程度。
                     # 较小的tau值会导致目标网络变化较慢，从而增加了训练的稳定性，但也可能降低学习速度。
        'lr':0.0003, # learning rate (default: 0.0003)
        'alpha':0.2, # Temperature parameter α determines the relative importance of the entropy\term against the reward (default: 0.2)
        'automatic_entropy_tuning':False, # Automaically adjust α (default: False)
        'batch_size':512, # batch size (default: 256)
        'num_steps':1000, # maximum number of steps (default: 1000000)
        'hidden_sizes':[1024,512,256], # 隐藏层大小，带有激活函数的隐藏层层数等于这一列表大小
        'updates_per_step':1, # model updates per simulator step (default: 1) 每步对参数更新的次数
        'start_steps':1000, # Steps sampling random actions (default: 10000) 在开始训练之前完全随机地进行动作以收集数据
        'target_update_interval':10, # Value target update per no. of updates per step (default: 1) 目标网络更新的间隔
        'replay_size':10000000, # size of replay buffer (default: 10000000)
        'cuda':True, # run on CUDA (default: False)
        'LOAD PARA': False, #是否读取参数
        'task':'Train',# 测试或训练或画图，Train,Test,Plot
        'logs':True, 
        'activation':nn.ReLU, #激活函数类型
        'plot_type':'2D-2line', #'3D-1line'为三维图，一条曲线；'2D-2line'为二维图，两条曲线
        'plot_title':'reward-steps.png',
        # 'seed':None, #网络初始化的时候用的随机数种子  
        'obgen_seed': 114514, # 轨道初始化种子
        'max_epoch':50000} #是否留存训练参数供tensorboard分析 


sma = 7171e3
thrust = 5e-3
ref_p = 2e3
T_f = 2000
discreT = 50
Cd = np.eye(6)
R = np.eye(3)*1e6
Q11 = np.eye(3)*1e1
Q22 = np.eye(3)*2
S11 = np.eye(3)
S22 = np.eye(3)
num_ff = 3


# ''' 测试场景 '''
# scn = 'l2h_ni'
# in_fin_orbi = np.array([0.2,0.15,0.3,0.35])*np.pi
# ini_phi = 0.2*np.pi
# ''''''
# scn = 'l2h_shun'
# in_fin_orbi = np.array([0.3,0.15,0.2,0.35])*np.pi
# ini_phi = 0*np.pi
# ''''''
scn = 'h2l_ni'
# in_fin_orbi = np.array([0.2, 0.4, 0.25, 0.2])*np.pi
in_fin_orbi = np.array([0.25, 0.35, 0.3, 0.21])*np.pi # 原来的
ini_phi = 0.05*np.pi
# ''''''
# scn = 'h2l_shun'
# in_fin_orbi = np.array([0.3,0.35,0.25,0.15])*np.pi
# ini_phi = 0.05*np.pi


env_f = multi_env.Rel_trans(sma, thrust, ref_p, T_f, discreT,Cd,R,Q11,Q22,S11,S22,pre_horizon = 5)

# 智能体和缓存区
agent_sac = [None for _ in range(num_ff)]
memory_n = [None for _ in range(num_ff)]
for agt_idx in range(num_ff):
    agent_sac[agt_idx] = MASAC(env_f.observation_space.shape[0], env_f.action_space, args, num_ff) #discrete不能用shape，要用n提取维度数量
    memory_n[agt_idx] = ReplayMemory(args['replay_size'])


for agt_idx in range(num_ff):
    agent_sac[agt_idx].load_checkpoint('model/tri_converged_{}.pt'.format(agt_idx))

Satpos_array = np.zeros((3,np.ceil(T_f/discreT+1).astype(int),3))
obs_n = env_f.reset(in_fin_orbi, ini_phi)
agent_reward = [0.0 for _ in range(3)] # individual agent reward
state_list = [obs_n[1]]
step_in_ep = 0
single_done_num = [0 for _ in range(num_ff)]
dumm_area_list = []
agt_area_list = []
action_amp_list = []
while True:
    action_n = [None for _ in range(num_ff)]
    for agt_idx in range(num_ff):
        action_n[agt_idx] = agent_sac[agt_idx].select_action(obs_n[agt_idx], evaluate = True)  # 开始输出actor网络动作
    
    action_amp_list.append(sum(action_n))

    new_obs_n, h_rew_n, done_n, isInject_n= env_f.plotstep(action_n) # Step
    dumm_area_list.append(env_f.dum_area*1e-6/(env_f.c_dis**2))
    agt_area_list.append(env_f.agt_area*1e-6/(env_f.c_dis**2))
    for i, rew in enumerate(h_rew_n): 
        agent_reward[i] += rew
        single_done_num[i] += isInject_n[i]

    step_in_ep += 1
    obs_n = new_obs_n
    state_list.append(obs_n[1])
    terminal = (step_in_ep >= T_f/discreT )
    
    if all(isInject_n) or terminal:
        multireward = ', '.join(f"{round(rwd, 2)}" for rwd in agent_reward)
        print(f'step:{step_in_ep}, reward:[{multireward}]')
        if all(isInject_n):
            success=True
        break

print(agent_reward[i])
tri_state = np.array(state_list)*1e3
for i in range(tri_state.shape[0]):
    for sat_j in range(num_ff):
        Satpos_array[sat_j,i,:] = tri_state[i,sat_j*6:sat_j*6+3]

'''plot the ini and final orbit'''
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
ax = plt.axes(projection='3d')

import seaborn as sns
sns.set_style('darkgrid')

# 设置字体和字号
font = {'family': 'serif',
        'serif': 'Times New Roman',
        'weight': 'normal',
        'size': 15,
        }
plt.rc('font', **font)

'''反放缩'''
Xi_orbit = Xi_orbit * (1e-3/env_f.c_dis)
Xf_orbit = Xf_orbit * (1e-3/env_f.c_dis)
Satpos_array = Satpos_array * (1e-3/env_f.c_dis)
'''正式画图'''
plt.plot(Xi_orbit[0, :], Xi_orbit[1, :], Xi_orbit[2, :], 'k')
plt.plot(Xf_orbit[0, :], Xf_orbit[1, :], Xf_orbit[2, :], 'r')

for sat_j in range(num_ff):    
    ax.plot(Satpos_array[sat_j,0:step_in_ep,0],Satpos_array[sat_j,0:step_in_ep,1],Satpos_array[sat_j,0:step_in_ep,2],color=[0,1/(num_ff-1)*sat_j,1],linewidth=1,
            label=f"agnet{sat_j}") #画三维图
plt.plot(0,0,0,'r*') #画一个位于原点的星形

'''
# def set_axes_equal(ax):
#     """
#     Make the axes of a 3D plot have equal scale so that spheres appear as spheres,
#     cubes as cubes, etc.. This is one possible solution to Matplotlib's ax.set_aspect('equal')
#     and ax.axis('equal') not working for 3D.
#     """
#     x_limits = ax.get_xlim3d()
#     y_limits = ax.get_ylim3d()
#     z_limits = ax.get_zlim3d()

#     x_range = abs(x_limits[1] - x_limits[0])
#     y_range = abs(y_limits[1] - y_limits[0])
#     z_range = abs(z_limits[1] - z_limits[0])

#     x_middle = np.mean(x_limits)
#     y_middle = np.mean(y_limits)
#     z_middle = np.mean(z_limits)

#     # The plot bounding box is a sphere in the sense of the infinity norm, hence
#     # setting the range to a 3-tuple will result in a cube.
#     plot_radius = 0.5 * max([x_range, y_range, z_range])

#     ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
#     ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
#     ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
# ax.view_init(elev=45, azim=45)  # 45度仰角，45度方位角
# set_axes_equal(ax)'''

ax.set_xlabel('x/km')
ax.set_ylabel('y/km')
ax.set_zlabel('z/km')

plt.title(f'MASAC, sma = {round(sma/1e3,1)}km, ref = {ref_p/1e3} km')
plt.legend(loc='upper right', bbox_to_anchor=(1.12, 1.03))
plt.tight_layout()



# 重开一个窗口绘制2D图
fig2, ax2 = plt.subplots()  # 创建一个新的figure实例
# dumm_area_list = dumm_area_list
# agt_area_list = agt_area_list
# 绘制2D图，横轴为step_in_ep，纵轴为dumm_area_list
heading_x = [jj * discreT for jj in range(step_in_ep)]
ax2.plot(heading_x, dumm_area_list, label='MPC Area',color = 'C0')
ax2.plot(heading_x, agt_area_list, label='Agent Area',color = 'C1')
ax2.set_xlabel('Time[s]')
ax2.set_ylabel('Area[km^2]')
ax2.set_title('Heading Area')
ax2.legend(loc='lower right')

fig3, ax3 = plt.subplots()  # 创建一个新的figure实例
ax3.plot(heading_x, dumm_area_list, label='Agent Area',color = 'C1')
ax3.plot(heading_x, agt_area_list, label='MPC Area',color = 'C0')
ax3.set_xlabel('Time[s]')
ax3.set_ylabel('Area[km^2]')
ax3.set_title('Heading Area')
ax3.legend(loc='lower right')


print(f'MASAC-{scn}-sma{sma/1e3}-thrust{thrust}-ref_p{ref_p/1e3}-step{step_in_ep}')
print(f'MASAC-{scn}-sma{sma/1e3}-thrust{thrust}-ref_p{ref_p/1e3}-step{step_in_ep}-heading')



plt.show()


print("----------------------------------------")
print("完成：{}".format(success))
print("----------------------------------------")