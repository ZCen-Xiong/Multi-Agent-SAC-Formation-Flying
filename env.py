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
        target_weight = travel**(1/5)
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
        sat_i = 1 # 真实星在step的内部编号永远是1
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
        reward = exc_t_rewa * (1 - travel**(1/2)) * heading_reward - exc_t_puni*1e-3*travel**(1/5)*(
            dV + 3*(np.linalg.norm(J_agent)-np.linalg.norm(J_dumm)))
        
        if np.linalg.norm(J_agent)-np.linalg.norm(J_dumm)>100 and np.linalg.norm(J_dumm)<1:
            reward = -1e3

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
                agent_seq[index_start:index_end] = Free(t_r,0,next_agent_state[sat_i*6:sat_i*6+6],self.nT)
                target_seq[index_start:index_end] = Free(self.t_scn + t_r,0,target_state[sat_i*6:sat_i*6+6],self.nT)
        
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
