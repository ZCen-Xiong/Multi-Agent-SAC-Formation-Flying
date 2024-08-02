import datetime
import numpy as np
import itertools
import torch.nn as nn
import random
from sac_multi import SAC
#from torch.utils.tensorboard import SummaryWriter
from replay_mem_multi import ReplayMemory
import env 
import time
import matplotlib.pyplot as plt
import numpy as np
from Orbit_Dynamics.CW_Prop import Free
# merge ceshi
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None) #没有充满时先用none创造出self.position
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity #超出记忆池容量后从第一个开始改写以维持容量不超标

    def sample(self, batch_size,seed):
        random.seed(seed)
        sequence = [i for i in range(1, 11)]
        batch = random.sample(sequence, batch_size)
        print(batch)  # 输出: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

aa = ReplayMemory(100)
for i in range(10):
    aa.sample(4,114)