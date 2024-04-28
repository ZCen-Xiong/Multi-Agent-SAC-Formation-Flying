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

'''首先准备好所有agent的外推'''
def multi_step(agent,env_f,memory,Multi_Agent_state,Multi_target_state_ini, dep_param):
            
            # 传入的常数 dep_param = np.array([horizon, num_ff ,ref_sat_distro, T_f_ex, dT_ex]   
            horizon = dep_param[0]
            num_ff = dep_param[1]
            ref_sat_distro = dep_param[2]
            T_f_ex = dep_param[3]
            dT_ex = dep_param[4]
            heading_ex = dep_param[5]

            s_len = 6*horizon # 智能体数据间的间隔, 意思就是每个智能体有多少步
            r_len = 6         # 时间相邻两个状态的数据间隔
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
            next_Multi_Agent_state = np.zeros(num_ff,6)
            next_Multi_target_state = np.zeros(num_ff,6)
            next_Multi_Agent_pos = np.zeros(num_ff,3)
            next_Multi_dummy_pos = np.zeros(num_ff,3)

            for sat_index in range(num_ff):
                # 下面整理输入
                # 读取agent的参考, 例如, 1号agent要参考6,3,那么序列的顺序就是1,6,3
                local_ref = np.array([sat_index, ref_sat_distro[sat_index,:]])

                for sat_i in local_ref:
                    for ri in range(0,horizon):
                        # 预测序列是按这样布置的, 先把第一个智能体的1-5步放完, 再把第二个智能体的1-5步放完, 再放第三个智能体的1-5步
                        #   sat1 t1-t5, sat2 t1-t5, sat3 t1-t5
                        t_r = (ri+1)*T_f_ex
                        # 这种就是按照 先排完一个智能体的全部数据, 再排另外一个智能体
                        index_start = sat_i*s_len + ri*r_len  
                        index_end = index_start + 6
                        multi_agent_seq[index_start:index_end] = Free(t_r,0,Multi_Agent_state[sat_i,:], 1)
                        # target 的外推原点一直是公元0年 所以加一个 dT_ex
                        multi_target_seq[index_start:index_end] = Free(dT_ex + t_r,0, Multi_target_state_ini[sat_i,:], 1)
                
                ''' 1e3是为了 放缩, 让状态在01之间
                顺便说一下, 区分state_env_input和state, 一个是个智能体用的, 一个是整个场景包括7个智能体用的'''
                state_env_input = np.concatenate((multi_agent_seq/1e3,multi_target_seq/1e3, heading_ex, np.array([travel])))
                action = agent.select_action(state_env_input)  # 开始输出actor网络动作
                
                next_state_env, next_agent_pos, next_dummy_pos, J_f_agent,J_f_dummy,reward, dV = env_f.step(action, state_env_input) # Step
                '''涉及到太多运算了, 把状态都提出来算了'''
                '''
                # Ignore the "done" signal if it comes from hitting the time horizon.
                由于智能体只学一步, 所以memory只记录一步的
                多说一句, 即使状态里80%的东西都被我扔掉了, 不作为实际的运动, 他们也是有用的, 在网络里用于校准Q网络
                另外, 由于这里算出来的奖励没有用, 因为参考星的位置都是估计出来的, 不是真实的, 得所有星都算完一遍之后, 再评估
                ''' 
                data_saver4push[sat_index,:] = [action, state_env_input, next_state_env, J_f_agent, J_f_dummy, reward, dV]
                ''' 传递给下一个时刻的状态(while级别的下一个循环）, 有且仅有 第n个卫星的 一个 step时间之后 的状态 1*6'''
                next_Multi_Agent_state[sat_index,:] = next_state_env[0:6]
                next_Multi_target_state[sat_index,:] = next_state_env[num_ff*horizon*6:num_ff*horizon*6 + 6]
                '''仅用于计算面积奖励'''
                next_Multi_Agent_pos[sat_index,:] = next_agent_pos
                next_Multi_dummy_pos[sat_index,:] = next_dummy_pos


            travel = next_state_env[-1] 

            # 开始计算奖励
            ''' 1、先计算heading_reward_all 有点尴尬的是, 我这个heading area只计算三个星的....后面4,5,6星得改改 '''
            heading_reward_all = env_f.calculate_area(next_Multi_Agent_pos) - env_f.calculate_area(next_Multi_dummy_pos)

            # 再依次push数据
            isInject = 0
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
                    if np.linalg.norm(J_indv_agent) > Conv_tol:
                        isInject += 0 #用来区分越界和到达目标
                        reward_push = 0
                    elif np.linalg.norm(J_indv_agent) <= Conv_tol and travel <= 1.0:
                        isInject += 1
                        if Sat_done_flag[sat_index,0] is 0: 
                            '''到达目标轨道的智能体不再重复获得奖励'''
                            reward_push += 10000
                            Sat_done_flag[sat_index,0] = 1

                '''[action, state_env_input, next_state_env, J_f_agent, J_f_dummy, reward, dV]'''
                action_push = data_saver4push[sat_index,0]
                state_push = data_saver4push[sat_index,1]
                next_state_push = data_saver4push[sat_index,2]
                memory.push(state_push, action_push, reward_push, next_state_push, done_push) # Append transition to memory
            # 完成一次外推
            return next_Multi_Agent_state      