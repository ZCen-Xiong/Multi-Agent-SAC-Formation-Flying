import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
import os
import math
from Orbit_Dynamics.CW_Prop import Free, orbitgen
from control_dep import discrete_state ,MPC_dep, quadprog
'''一个无重力环境下的三维测试用环境'''

class Rel_trans:
    def __init__(self, sma, thrust, ref_p, T_f, discreT,Cd,R,Q11,Q22,S11,S22,pre_horizon):
        # 输入单位为m和s 半长轴 推力加速度 相对轨道振幅
        #状态空间设置为无限连续状态空间,虽然不知道相比设成离散空间有什么影响
        '''用于 状态到 智能体输入的缩放映射，因为输出太大的话，智能体会直接脑死亡'''
        self.scaling = 1e-3 
        self.sma = sma # semi major axis
        self.mu = 398600.5e9
        # normalizing
        self.c_time = np.sqrt(self.mu / (self.sma ** 3))
        # self.nT_real = np.sqrt(self.mu / (self.sma ** 3))   # real angular velocity
        self.c_dis = self.mu**(1/3)/self.sma
        # normalized
        self.nT = 1 
        self.T_f_norm = T_f*self.c_time
        self.dT_norm = discreT*self.c_time
        self.thrust_norm = thrust*self.c_dis/(self.c_time**2)
        self.p_norm = ref_p*self.c_dis
        
        # prediction horizon is xxx steps
        self.horizon = pre_horizon
        # action 包括a和b两个方向,3个智能体,每个智能体预测h步
        self.action_space=spaces.Box(low = -0.1 , high = 0.1, shape=(2*3*self.horizon,), dtype=np.float32) 
        # 实际观测状态是当前状态+预测的序列  3个智能体 初始和结束状态 6个参数 +3(空间指向) +1 时间历程
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(3*6*2*self.horizon+3+1,),dtype=np.float32)

        self.isInject=0

        self.Ad_norm, self.Bd_norm = discrete_state(self.dT_norm,self.nT)
        self.Cd_norm = Cd # Cd是读出矩阵,保持不变
        # Q矩阵（状态的费用）
        Q11 = Q11/ (self.c_dis**2)
        Q22 = Q22 / (self.c_dis**2 / (self.c_time**2))
        self.Q_norm = np.block([[Q11, np.zeros((3, 3))], [np.zeros((3, 3)), Q22]])
        # S 结束状态的费用
        S11 = np.eye(3) /  (self.c_dis**2)
        S22 = np.eye(3) / (self.c_dis**2 / (self.c_time**2))
        self.S_norm = np.block([[S11, np.zeros((3, 3))], [np.zeros((3, 3)), S22]])
        # R矩阵（控制的费用）
        self.R_norm = R / (self.c_dis**2 / (self.c_time**4))

        ''' 生成MPC控制器'''
        self.MPC = MPC_dep(6,6, self.horizon, 
                           self.Ad_norm, self.Bd_norm, self.Cd_norm, 
                           self.R_norm, self.Q_norm, self.S_norm)
        self.Q_bar = self.MPC.Qbar_func()
        self.R_bar = self.MPC.Rbar_func()
        # U对状态影响 AB联合
        self.F_norm = self.MPC.Fbar_func()
        #  A联合
        self.M_norm = self.MPC.Mbar_func()
        Hbar = self.MPC.H_func()
        self.Hbar_half = Hbar/2 + Hbar.T/2
        self.f = self.MPC.f_func()

    def reset(self, ini_fin_in = None,ini_phi = None, seed = None): #,prop_t
        super().__init__()

        '''固定随机数种子'''
        if seed is not None:
            np.random.seed(seed)

        self.t_scn = 0 # 场景初始时间
        self.isInject=0 # done无法用于区分是否到达目标,所以加入另一个标志位
        self.done=False # done是q函数计算所必要的
        if ini_fin_in is None:
            ''' initial orbit 以下为随机生成两组差距36度的轨道'''
            self.azi_0 = np.random.uniform(0, 2*np.pi)
            # 只朝z正已经覆盖了全部目标空间了,因为智能体姿态可以正负转,主要害怕的是-0到+0的奇异
            self.ele_0 = np.random.uniform(0.05*np.pi, 0.5*np.pi)
            # final orbit 轨道差为0.2pi和0.1pi
            self.azi_f = self.azi_0 + np.random.uniform(-0.1*np.pi, 0.1*np.pi)
            self.ele_f = self.ele_0 + np.random.uniform(-0.1*np.pi, 0.1*np.pi)
            '''固定实验 '''
            if self.ele_f > 0.5*np.pi or self.ele_f < 0.1*np.pi:
                self.ele_f = self.ele_0 - (self.ele_f - self.ele_0)
        else:
            self.azi_0 = ini_fin_in[0]
            self.ele_0 = ini_fin_in[1]
            self.azi_f = ini_fin_in[2]
            self.ele_f = ini_fin_in[3]

        # Check and adjust self.ele_f if it exceeds 0.5*pi

        # real amplitude =  20 * sma /(mu)**(1/3)
        self.orbit_i0 = orbitgen(self.nT , self.azi_0, self.ele_0, self.p_norm)
        self.orbit_f0 = orbitgen(self.nT , self.azi_f, self.ele_f, self.p_norm)
        # 生成初始轨道的随机相位
        if ini_phi is None:
            phase_d = np.random.uniform(0.0,2*np.pi)
        else:
            phase_d = ini_phi
        # 求出轨道状态
        Xi_0 = Free(phase_d,0,self.orbit_i0,self.nT)
        Xf_0 = Free(phase_d,0,self.orbit_f0,self.nT)
        theta_i = np.arctan2(Xi_0[1], Xi_0[0])
        theta_f = np.arctan2(Xf_0[1], Xf_0[0])
        theta_diff = theta_f - theta_i  # 顺时针下,f滞后于i的相位
        f_phase_adjust = 0
        if np.abs(theta_diff) > np.pi:  # 说明超出范围了,因为我在生成轨道的是都只会差一点角度
            real_phd = 2*np.pi - np.abs(theta_diff)
            if theta_f < 0:
                f_phase_adjust = 2*real_phd
        elif theta_diff > 0:
            f_phase_adjust = 2*theta_diff
        self.Xf_adj = Free(f_phase_adjust,0,Xf_0,self.nT)
        # 尤其需要注意,sat_state是 t时刻的[智能体1,智能体2,智能体3]
        # 而后面的 agent_seq 是 [智能体1（1-5时刻）,智能体2（1-5时刻）,智能体3（1-5时刻）]
        agent_state = Xi_0
        dumm_state =  agent_state
        target_state = self.Xf_adj
                
        def an2vec(elevation_rad,azimuth_rad):
            # 计算方向矢量
            x = np.cos(elevation_rad) * np.cos(azimuth_rad)
            y = np.cos(elevation_rad) * np.sin(azimuth_rad)
            z = np.sin(elevation_rad)
            # 输出方向矢量
            direction_vector = np.array([x, y, z])
            return direction_vector

        self.heading = an2vec(self.ele_f,self.azi_f)

        # reset  不再作为step的前序,完全解耦
        return agent_state, dumm_state, target_state, self.heading, self.dT_norm, self.T_f_norm
    
    '''计算目标天区与三颗星的方向构成面积的奖励'''
    def calculate_area(self, pos):
        link1 = pos[1,:] - pos[0,:]
        link2 = pos[2,:] - pos[0,:]
        formation_product = np.cross(link1, link2)
        proj_area = abs(0.5 * np.dot(formation_product, self.heading))
        return proj_area
    
    def step(self,Sat_action, state_input, dumm_state): 
        '''
        这个step被大改了, 为了简便期间，尽量使用了原STEP的变量和结构
        简便起见,每次step仅处理一个智能体的运动,这与网络内部无关,仅是出于编程需要,这样扩展网络的话,仅需要在外部修改
        action是一个向量系数 3*2*self.horizon 个 3个智能体 每个智能体2个维度
        前一半的系数3*horizon是alpha,后一半的系数3*horizon是beta     
        state_input : 智能体序列    0:3*6*horizon                18*h  被放缩
                    目标序列,       3*6*horizon: 2*3*6*horizon   18*h  被放缩
                    指向,      -4:-1
                    时间历程   -1  

        dumm_state  智障状态(不是序列       18                  不用放缩   智障状态不需要作为智能体动作的参考
        '''  
        ''' 把长数组分别拆到对应的里面, 另外需要注意state_input是被放缩过，所以反放缩一下！！！！'''
        '''与外部172行左右区分！，这里seq是3个星状态序列，3*6*5，状态是三个星状态 3*6'''
        agent_seq = state_input[0:3*6*self.horizon]/self.scaling
        target_seq = state_input[3*6*self.horizon:2*3*6*self.horizon]/self.scaling
        # 历程
        travel = state_input[-1]
        def seq2state(seq,horizon):
            # 提取三个切片
            slice1 = seq[0:6]
            slice2 = seq[6*horizon:6*horizon+6]
            slice3 = seq[2*6*horizon:2*6*horizon+6]
            # 将三个切片合并成一个 1x18 的数组
            return np.concatenate((slice1, slice2, slice3))
        '''提取出这个step中的初始状态'''
        
        agent_state = seq2state(agent_seq, self.horizon)
        # dumm_state = agent_state      # 注释掉的话，是全程MPC，不注释的话，是局部MPC，和157行只能有一个活着
        '''注意，这里拿到是外推过的target'''
        target_state = seq2state(target_seq, self.horizon)

        # 沿着法向指向的矢量
        normal_track = target_seq - agent_seq  # 第一个智能体h*6个状态 第二个智能体 h*6个状态 第三个智能体 h*6个状态
        alpha = np.repeat(Sat_action[0:3*self.horizon],6)
        # 沿着径向指向的矢量 由目标状态指向原点
        radius_track = - target_seq    # 3 * horizon * 6 第一个智能体h*6个状态 第二个智能体 h*6个状态 第三个智能体 h*6个状态
        beta = np.repeat(Sat_action[3*self.horizon:],6)  # 3 * horizon
        # 未修正的路径航点  alpha偏置,beta偏置 + 当前预测点
        raw_point = np.dot(alpha,normal_track) + np.dot(beta,radius_track)

        # 使用连续的
        target_weight = travel
        # 修正后的路径航点 3*(h*6)
        pred_point = raw_point * (1-target_weight) + target_seq

        # MPC序列 
        lb = -np.ones([3*self.horizon,1])*self.thrust_norm
        ub = np.ones([3*self.horizon,1])*self.thrust_norm

        s_len = 6*self.horizon # 智能体数据间的间隔,意思就是每个智能体有多少 步 * 状态
        r_len = 6              # 时间相邻两个状态的数据间隔
        # 采用了dV计算,直接不用了
        dV = 0.0
        
        next_agent_state = np.zeros([3*6,]) # 18个状态,就是最正统的智能体在T_SCN控制后的实际状态
        next_agent_pos = np.zeros([3,3])    # 仅用于计算三颗星的方向与目标天区
         # 仅用于计算是否入轨
        next_agent_seq = np.zeros([3*s_len,]) # 从t_scn开始的外推

        next_dumm_state = np.zeros([3*6,]) # 不使用强化学习的MPC规划
        next_dumm_pos = np.zeros([3,3])
         # 仅用于计算是否入轨 所以暂时用不上
        next_dumm_seq = np.zeros([3*s_len,])
        
        for sat_i in range(3):
            # 从target集合中,取出单个智能体的预测序列 
            # 取出第一个智能体的 6*horizon 目标状态
            s_target_seq = pred_point[s_len*sat_i:s_len*(sat_i+1)] 
            # 取出第一个智能体的 6*1当前状态
            s_sat_state = agent_state[sat_i*6:sat_i*6+6]
            s_dumm_state = dumm_state[sat_i*6:sat_i*6+6]
            # 一次项系数 first_order
            fo = np.hstack((s_sat_state, s_target_seq)).T @ self.f
            fo_dumm = np.hstack((s_dumm_state, target_seq[s_len*sat_i:s_len*(sat_i+1)])).T @ self.f
            # 计算出第一个智能体的 3*horizon 个机动控制
            Acons_norm = self.MPC.Acons_func() @ self.F_norm
            Bcons_norm = self.MPC.Dy_func(-np.ones((6,1))*1e2, np.ones((6,1))*1e2) - self.MPC.Acons_func()@ self.M_norm @ s_sat_state.reshape(6, 1)
            s_agent_act_seq = quadprog(self.Hbar_half, fo.T, Acons_norm, Bcons_norm,  None, None,lb, ub)  
            # 计算出没有智能体参与的 3*horizon 个机动控制
            s_dumm_act_seq = quadprog(self.Hbar_half, fo_dumm.T, Acons_norm, Bcons_norm,  None, None,lb, ub).flatten()  


            s_agent_act_seq = s_agent_act_seq.flatten() # 3*horizon
            current_action = s_agent_act_seq[0:3]
            dV = dV + np.linalg.norm(current_action)
            if sat_i == 0:
                First_dV = np.linalg.norm(current_action)
            # 计算出下一个时刻的状态
            next_agent_state[6*sat_i:6*(sat_i+1)] = self.Ad_norm @ s_sat_state + self.Bd_norm @ current_action
            next_dumm_state[6*sat_i:6*(sat_i+1)] = self.Ad_norm @ s_dumm_state + self.Bd_norm @ s_dumm_act_seq[0:3]
            # 仅用于计算三颗星的方向与目标天区
            next_agent_pos[sat_i,:] = next_agent_state[6*sat_i:6*sat_i+3]
            next_dumm_pos[sat_i,:] = next_dumm_state[6*sat_i:6*sat_i+3]
            # 仅用于计算是否入轨
            next_agent_seq[s_len*sat_i:s_len*(sat_i+1)] = self.M_norm @ agent_state[6*sat_i:6*(sat_i+1)] + self.F_norm @ s_agent_act_seq
            next_dumm_seq[s_len*sat_i:s_len*(sat_i+1)] = self.M_norm @ dumm_state[6*sat_i:6*(sat_i+1)] + self.F_norm @ s_dumm_act_seq

        # 0425 被挪到外面 /0426 算了，不挪了，作为一个内部指标的参考，表征对于参考运动的推测，

        heading_reward = self.calculate_area(next_agent_pos) - self.calculate_area(next_dumm_pos)
        J_agent = target_seq - next_agent_seq
        J_dumm = target_seq - next_dumm_seq

        '''计算第一个（真实存在）智能体的损失函数,将会被输出出去'''
        sat_i = 0 # 真实星在step的内部编号永远是第一个
        J_f_agent = target_seq[s_len*sat_i:s_len*(sat_i+1)] - next_agent_seq[s_len*sat_i:s_len*(sat_i+1)]
        J_f_dumm = target_seq[s_len*sat_i:s_len*(sat_i+1)] - next_dumm_seq[s_len*sat_i:s_len*(sat_i+1)]
        
        '''0425 被挪到外面'''
            # 慢于MPC则出现控制惩罚
        exc_t_puni = 0.0
        exc_t_rewa = 1.0
        if np.linalg.norm(J_dumm)<1:
            exc_t_puni = 1.0
        if np.linalg.norm(J_dumm)<1e-1:
            exc_t_rewa = 0.0

        
        # 计算一个无面积奖励的奖励，实际上就是只有控制惩罚
        reward = (exc_t_rewa * (1 - travel**(1/2)) * heading_reward - exc_t_puni*travel**(1/5)*(
            dV + 3*(np.linalg.norm(J_agent)-np.linalg.norm(J_dumm))))*1e-4
        
        if np.linalg.norm(J_agent)-np.linalg.norm(J_dumm)>100 and np.linalg.norm(J_dumm)<1:
            reward = -1e1

            # 收敛极限
            # Conv_tol = 10
            # if travel >= 1 or np.linalg.norm(J_agent) <= Conv_tol:
            #     self.done = True
            #     if np.linalg.norm(J_agent) > Conv_tol:
            #         self.isInject = 0 #用来区分越界和到达目标
            #         reward = 0
            #     elif np.linalg.norm(J_agent) <= Conv_tol and self.t_scn <= self.T_f_norm:
            #         self.isInject=1
            #         reward = 10000
               
        # 因为需要为下一个时刻
        for sat_i in range(3):
            for ri in range(self.horizon):
                # 预测序列是按这样布置的,先把第一个智能体的1-5步放完,再把第二个智能体的1-5步放完,再放第三个智能体的1-5步
                #   sat1 t1-t5, sat2 t1-t5, sat3 t1-t5
                t_r = (ri+1)*self.dT_norm
                # 这种就是按照 先排完一个智能体的全部数据,再排另外一个智能体
                index_start = sat_i*s_len + ri*r_len  # 0 
                index_end = index_start + 6
                agent_seq[index_start:index_end] = Free(t_r - self.dT_norm, 0, next_agent_state[sat_i*6:sat_i*6+6],self.nT)
                # 又是你的问题.... 174行注意，拿到的是外推过的target！不需要再外推了
                target_seq[index_start:index_end] = Free(t_r,0,target_state[sat_i*6:sat_i*6+6],self.nT)
        
        '''放在这里的原因是：例如这个函数输入是第一步，那么234到235行已经计算出第2步了，
            对于agent，并不要第二步作为输入, 动作是3-7步，而在267行已经+1了，所以时间外推不能放前面'''
        self.t_scn += self.dT_norm
        '''为了给下一次agent输入，这里的第一个状态是被放缩了的，请注意'''
        return np.concatenate((agent_seq*self.scaling, target_seq*self.scaling ,self.heading, np.array([travel]))), next_dumm_state, next_agent_pos, next_dumm_pos, J_f_agent,J_f_dumm,reward,First_dV
    
    def plot(self, args, data_x, data_y, data_z=None):
        if data_z!=None and args['plot_type']=="2D-2line":
            fig = plt.figure()
            ax = fig.gca() #Axes对象是图形的绘图区域,可以在其上进行各种绘图操作。通过gca()函数可以获取当前图形的Axes对象,如果该对象不存在,则会创建一个新的。
            plt.plot(data_x,data_y,'b',linewidth=0.5)
            plt.plot(data_x,data_z,'g',linewidth=1)
            ax.set_xlabel('x', fontsize=15)
            ax.set_ylabel('y', fontsize=15)
            ax.set_xlim(np.min(data_x),np.max(data_x))
            ax.set_ylim(np.min([np.min(data_y),np.min(data_z)]),np.max([np.max(data_y),np.max(data_z)]))
            if not os.path.exists('logs'):
                os.makedirs('logs')
            plt.savefig(args['plot_title'])# 'logs/{}epoch-{}steps.png'.format(epoch,steps)


def multi_step(agent,env_f,memory, travel, Multi_Agent_state, Multi_dumm_state, Multi_target_state_ini, dep_param):
    sc_m = 1e-3       
    # 传入的常数 dep_param = [horizon, num_ff ,ref_sat_distro, heading_ex, dT_ex, T_f_ex]  
    horizon = dep_param[0]
    num_ff = dep_param[1]
    ref_sat_distro = dep_param[2]
    heading_ex = dep_param[3]
    dT_ex = dep_param[4]
    T_f_ex = dep_param[5]
    T_scn = travel*T_f_ex

    s_len = 6*horizon # 智能体数据间的间隔, 意思就是每个智能体有多少步
    r_len = 6         # 时间相邻两个状态的数据间隔
    Sat_inject_flag = np.zeros((num_ff,1))  # 三个智能体的入轨状态
    Sat_done_flag = np.zeros((num_ff,1))  # 三个智能体的完成状态
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
        # local_ref = np.array([sat_index, *ref_sat_distro[sat_index,:]])
        local_ref = ref_sat_distro[sat_index,:]
        for sat_i, ref_i in zip(range(3), local_ref):
            for ri in range(0,horizon):
                # 预测序列是按这样布置的, 先把第一个智能体的1-5步放完, 再把第二个智能体的1-5步放完, 再放第三个智能体的1-5步
                #   sat1 t1-t5, sat2 t1-t5, sat3 t1-t5
                t_r = (ri+1)*dT_ex
                # 这种就是按照 先排完一个智能体的全部数据, 再排另外一个智能体
                index_start = sat_i*s_len + ri*r_len  
                index_end = index_start + 6
                multi_agent_seq[index_start:index_end] = Free(t_r - dT_ex,0,Multi_Agent_state[ref_i,:], 1)
                # target 的外推原点一直是公元0年 所以加一个 T_scn
                multi_target_seq[index_start:index_end] = Free(T_scn + t_r,0, Multi_target_state_ini[ref_i,:], 1)
        
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
        # 永远只有第一个航天器的状态在变，第二第三个好像一直不动
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

    # TRAVEL的定义域是[0,1]
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
        reward_push = exc_t_rewa * (1 - travel**(1/2)) *heading_reward_all*1e-4 - exc_t_puni*1e-4*travel**(1/5)*(
            dV_indv + 3*(np.linalg.norm(J_indv_agent)-np.linalg.norm(J_indv_dummy))) + reward_pseudo*0.1
        
        if np.linalg.norm(J_indv_agent)-np.linalg.norm(J_indv_dummy)>100 and np.linalg.norm(J_indv_dummy)<1:
            reward_push = -1e1

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
                    reward_push += 100
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