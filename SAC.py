import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import gymnasium as gym
import math
import time
from torch.distributions import Normal
import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import collections
from lib.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from lib.dummy_vec_env import DummyVecEnv
from lib.trpo_step import trpo_step
from typing import Union,Dict


file_name = 'SAC'
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

new_directory1 = f'model/{file_name}'
new_directory2 = f'log/{file_name}'

model = os.path.join(current_directory, new_directory1).replace('\\', '/')
log = os.path.join(current_directory, new_directory2).replace('\\', '/')



LR = 3e-4
LR1 = 3e-4
LR2 = 3e-4
GAMMA = 0.99
GAE=0
MODEL_PATH = model
SAVE_PATH_PREFIX = log
TEST = False
SEED=0
MINIBATCHSIE=512
ROLLOUT=8
NUMACTOR=8
EPISODES =int(1e5)            # 训练的更新步数，真实更新步数：EPISODES*NUMACTOR*ROLLOUT
GRAD=0.5
MAX=int(1e5*NUMACTOR*ROLLOUT)
MIN=int(25e2*NUMACTOR*ROLLOUT)
mean_lambda=1e-3
std_lambda=1e-3
z_lambda=0.0
TAU=1e-2
if TEST:
    EPISODES=1
    NUMACTOR=2
    ROLLOUT=5


class TimeLimit(gym.Wrapper):
    """
    限制最大步数
    """
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        #toomany unpack的意思是，你放少了，还有属性没有接受
        observation, reward, done, truncated,info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done,truncated,info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low = self.action_space.low
        high = self.action_space.high
        
        
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action

    def _reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high
        
        return 2 * (action - low) / (high - low) - 1
    
    def action(self, action):
        return self._reverse_action(action)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if TEST:
    env=NormalizedActions(gym.make("HalfCheetah-v4",render_mode="human"))
else:
    env=NormalizedActions(gym.make("HalfCheetah-v4"))


env = TimeLimit(env, max_episode_steps=env.spec.max_episode_steps)

env.action_space.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
torch.use_deterministic_algorithms(True)


state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] 
max_action = float(env.action_space.high[0])

os.makedirs(f"{SAVE_PATH_PREFIX}", exist_ok=True)
os.makedirs(f"{MODEL_PATH}", exist_ok=True)



class Actor(nn.Module):
    """
    分别是连续动作的确定性和随机性策略
    这里是另一种对连续动作的处理，和td3类似的确定性动作，直接采用均值
    通过网络学习到mean和logstd，然后在用这个高斯分布采样
    
    """
    def __init__(self, state_dim=state_dim, action_dim=action_dim,max_action=max_action,
        init_w=3e-3,log_std_min=-20,log_std_max=2):
        super(Actor, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        
        self.mean_linear = nn.Linear(256, action_dim)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(256, action_dim)        
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)
        self.max_action=max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        
        mean    = self.mean_linear(a)
        log_std = self.log_std_linear(a)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob, z, mean, log_std
        
    
    def get_action(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        z      = normal.sample()
        action = torch.tanh(z)
        
        return action


class SQ(nn.Module):
    def __init__(self, state_dim=state_dim, action_dim=action_dim, init_w=3e-3):
        super(SQ, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        
        self.l3.weight.data.uniform_(-init_w, init_w)
        self.l3.bias.data.uniform_(-init_w, init_w)
        
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)
        

    def forward(self, state, action):
        q = F.relu(self.l1(torch.cat([state, action], 1)))
        q = F.relu(self.l2(q))
        return self.l3(q)
    
    def Q(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1,q2
    
    
class Value(nn.Module):
    def __init__(self,state_dim=state_dim,init_w=3e-3):
        super(Value, self).__init__()
        self.l1= nn.Linear(state_dim,256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        
        self.l3.weight.data.uniform_(-init_w, init_w)
        self.l3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self,state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, device):
        self.observations = torch.zeros(num_steps + 1, num_processes, obs_shape).to(device)
        self.rewards = torch.zeros(num_steps, num_processes, 1).to(device)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1).to(device)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1).to(device)
        self.actions = torch.zeros(num_steps, num_processes, action_dim).to(device, torch.float32)
        self.masks = torch.ones(num_steps + 1, num_processes, 1).to(device)
        self.exp = torch.zeros(num_steps, num_processes, 1).to(device)
        self.gae_tau=GAE

        self.num_steps = num_steps
        self.step = 0

    def insert(self, current_obs, action,reward, mask):
        self.observations[self.step + 1].copy_(current_obs)
        self.actions[self.step].copy_(action)
        # self.value_preds[self.step].copy_(value_pred)
        self.rewards[self.step].copy_(reward)
        self.masks[self.step + 1].copy_(mask)
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self,gamma):
        # self.value_preds[-1] = next_value
        # gae=0
        
        # 多步的rewards
        temp=0
        nstep=torch.zeros(NUMACTOR,1).to(device)
        old=self.rewards
        for i in reversed(range(self.rewards.size(0))):
            temp=temp+self.rewards[i]
            nstep+=1
            self.exp[i]=nstep
            self.rewards[i]=temp
            temp*=self.masks[i]
            l = torch.nonzero(temp.view(-1) == 0).squeeze().tolist()
            if  len(l)!=0:
                nstep[l]=0
                temp[l]+=old[i][l]
                self.rewards[i][l]=temp
                nstep+=1
                self.exp[i][l]=nstep
            temp=temp*gamma
        
        # temp=0
        # o=torch.zeros(NUMACTOR,1).to(device)
        # old=self.rewards
        # for i in range(self.rewards.size(0)):
        #     temp=temp+self.rewards[i]*(gamma**o)
        #     o+=1
        #     self.rewards[i]=temp
        #     temp*=self.masks[i]
        #     l = torch.nonzero(temp.view(-1) == 0).squeeze().tolist()
        #     if  len(l)!=0:
        #         o[l]=0
        #         temp[l]+=old[i][l]
        #         self.rewards[i][l]=temp
        #         o+=1
        
        
        
        # for step in reversed(range(self.rewards.size(0))):
        #     delta= self.rewards[step] + self.value_preds[step + 1] * gamma * self.masks[step + 1] - self.value_preds[step]
        #     gae=delta+GAMMA*self.gae_tau*self.masks[step+1]*gae
        #     self.returns[step]=gae+self.value_preds[step]
        
        # advantages = self.returns[:-1] - self.value_preds[:-1]
        # advantages = (advantages - advantages.mean()) / (
        #     advantages.std() + 1e-5)
        
        state=self.observations[:-1].view(-1,*self.observations.size()[2:])
        action=self.actions.view(-1, self.actions.size(-1))
        # ret=self.returns[:-1].view(-1, 1)
        next_state=self.observations[1:].view(-1,*self.observations.size()[2:])
        reward=self.rewards.view(-1,1)
        mask=self.masks[1:].view(-1,1)
        exp=self.exp.view(-1,1)
        
        # adv_targ = advantages.view(-1, 1)
        
        return (state.cpu().data.numpy(),action.cpu().data.numpy(),
                next_state.cpu().data.numpy(),mask.cpu().data.numpy(),reward.cpu().data.numpy(),exp.cpu().data.numpy())
        
            
class ReplayBuffer(object):
    def __init__(self,max=MAX):
        self.mem=[]
        self.memlen=0
        self.max=max
        self.pos=0

    
    def push(self,data):
        if len(self.mem)<self.max:
            self.mem.append(data)
        else:
            self.mem[int(self.pos)]=(data)
        self.pos = (self.pos + 1) % self.max
        
    def sample(self, batch_size):
        # ind=random.sample(range(0,len(self.mem)*NUMACTOR*ROLLOUT), batch_size)
        ind=np.random.randint(0, (len(self.mem)*NUMACTOR*ROLLOUT), size=batch_size)
        state=[]
        action=[]
        # ret=[]
        next_state=[]
        mask=[]
        reward=[]
        exp=[]

        for i in range(batch_size):
            two = ind[i] % (NUMACTOR * ROLLOUT)
            one = ind[i] // (NUMACTOR * ROLLOUT)
            s, a,n,m,R,e=self.mem[one]
            state.append(s[two])
            action.append(a[two])
            # ret.append(r[two])
            next_state.append(n[two])
            mask.append(m[two])
            reward.append(R[two])
            exp.append(e[two])
            
        return (torch.tensor(np.array(state)).to(device), torch.tensor(np.array(action)).to(device), 
                torch.tensor(np.array(next_state)).to(device),
                torch.tensor(np.array(mask)).to(device),torch.tensor(np.array(reward)).to(device),
                torch.tensor(np.array(exp)).to(device))
    
    def __len__(self):
        return len(self.mem)
        

class SAC():
    def __init__(self):
        super(SAC,self).__init__()
        self.ACTOR=Actor().to(device)
        self.Value=Value().to(device)
        self.Value_target=Value().to(device)
        self.Q=SQ().to(device)
        for target_param, param in zip(self.Value_target.parameters(), self.Value.parameters()):
            target_param.data.copy_(param.data)
        self.value_cri=nn.MSELoss()
        self.soft_q_cri=nn.MSELoss()
        self.v_o=torch.optim.Adam(self.Value.parameters(),lr=LR)
        self.s_o=torch.optim.Adam(self.Q.parameters(),lr=LR1)
        self.A_o=torch.optim.Adam(self.ACTOR.parameters(),lr=LR2)
        self.rollout = ROLLOUT
        self.num_agents=NUMACTOR
        self.gama=GAMMA
        self.meanlamda=mean_lambda
        self.stdlamda=std_lambda
        self.zlamda=z_lambda
        self.tau=TAU
        self.rollouts = RolloutStorage(self.rollout, self.num_agents,
            state_dim, action_dim, device)
        self.mem=ReplayBuffer()
        
        if TEST:
            self.ACTOR.eval()
            self.Value.eval()
            self.Value_target.eval()
            self.Q.eval()
        else:
            self.ACTOR.train()
            self.Value.train()
            self.Value_target.train()
            self.Q.train()
        
    @torch.no_grad()
    def get_action(self, s):
        action=self.ACTOR.get_action(s)
        return action
    
    @torch.no_grad()
    def get_values(self, s):
        values = self.Value_target(s)
        return values
    
    
    def compute_loss(self, sample):
        state, action,next_state,mask,reward,exp=sample
        
        q_value=self.Q(state,action)
        value=self.Value(state)
        new_action, log_prob, z, mean, log_std = self.ACTOR.evaluate(state)
        
        
        
        
        target_value = self.Value_target(next_state)
        # next_q_value = reward + mask * GAMMA * target_value
        next_q_value = reward + mask * (GAMMA**exp) * target_value
        q_loss=self.soft_q_cri(q_value,next_q_value.detach())
        
        expected_new_q_value=self.Q(state,new_action)
        #熵在这里
        next_value = expected_new_q_value - log_prob
        value_loss = self.value_cri(value, next_value.detach())
        
        log_prob_target = expected_new_q_value - value
        
        action_loss=(log_prob * (log_prob - log_prob_target).detach()).mean()
        
        mean_loss=self.meanlamda*mean.pow(2).mean()
        std_loss=self.stdlamda*log_std.pow(2).mean()
        z_loss=self.zlamda*z.pow(2).sum(1).mean()
        action_loss+=mean_loss + std_loss + z_loss
        
        return action_loss, value_loss,q_loss
    
    
    
    def learn(self, batch_size):
        sample=self.mem.sample(batch_size)
        action_loss, value_loss,q_loss=self.compute_loss(sample)
        
        self.s_o.zero_grad()
        q_loss.backward()
        self.s_o.step()
        
        self.v_o.zero_grad()
        value_loss.backward()
        self.v_o.step()
        
        self.A_o.zero_grad()
        action_loss.backward()
        self.A_o.step()
        
        for target_param,param in zip(self.Value_target.parameters(),self.Value.parameters()):
            target_param.data.copy_(
            target_param.data * (1.0 - self.tau) + param.data * self.tau
        )
            
        return value_loss,action_loss,q_loss
    
    
    
    def save_train_model(self):
        torch.save(self.ACTOR.state_dict(), f"{MODEL_PATH}/ACTOR.pth")
        

    def load_net(self, file):
        self.ACTOR.load_state_dict(torch.load(f"{file}/ACTOR.pth"))
            
    def predict(self,observation: Union[np.ndarray, Dict[str, np.ndarray]],state,episode_start,deterministic):
        """
        这函数就是输出动作的，动作的形式要转换为nparray，其中shape为1*actiondim
        
        """
        n_batch = observation.shape[0]
        observation=torch.tensor(observation.astype(np.float32)).to(device)
        if not TEST:
            action = np.array([np.array(self.get_action(observation).cpu().item()) for _ in range(n_batch)])
        else:
            action = np.array([self.get_action(observation).view(-1,action_dim).cpu().data.numpy()[0] for _ in range(n_batch)])
        
        return action,state
    
    
def main():
    model=SAC()
    if  not TEST:
        
        torch.set_num_threads(1)
        envs=[env for i in range(NUMACTOR)]
        # envs=SubprocVecEnv(envs) if NUMACTOR > 1 else DummyVecEnv(envs)
        envs=SubprocVecEnv(envs) if NUMACTOR > 1 else env
        
        obs_shape = envs.observation_space.shape
        current_obs = torch.zeros(NUMACTOR, *obs_shape,
                        device=device, dtype=torch.float)
        def update_current_obs(obs):
            shape_dim0 = envs.observation_space.shape[0]
            obs = torch.from_numpy(obs.astype(np.float64)).to(device)
            current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
            current_obs[:, -shape_dim0:] = obs
        
        obs = envs.reset()
        if NUMACTOR==1:
            obs=obs[0]
        update_current_obs(obs)
        model.rollouts.observations[0].copy_(current_obs)
        writer = SummaryWriter(f'{SAVE_PATH_PREFIX}/{datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}-SAC-{EPISODES}')
    episode_rewards = np.zeros(NUMACTOR, dtype=np.float64)
    final_rewards = np.zeros(NUMACTOR, dtype=np.float64)
    if TEST:
        model.load_net(MODEL_PATH)
    for i in range(EPISODES):
        """
        如果要放到周期里面
        1、某个环境done了之后，怎么处理
        会直接重置    
    
        """
        if TEST:
            print("EPISODE: ", i)
            state,info=env.reset()
            ep_reward = 0
            while(True):
                state=torch.from_numpy(state.astype(np.float32)).to(device)
                action = model.get_action(state)
                action = action.view(-1,action_dim).cpu().data.numpy()
                action=action[0]
                next_state, reward, done, truncated, info = env.step(action)  # observe next state and reward
                state=next_state
                ep_reward+=reward
                env.render()
                if done:
                    print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
                    break
                
        if not TEST:
            if i%100==0 and NUMACTOR!=1:
                non_zero_rewards = final_rewards[final_rewards != 0]
                print(f"EPISODE: {int(i/100)} reward is {np.mean(non_zero_rewards):.2f} mid is {np.median(non_zero_rewards):.2f}")
                
            s=time.time()
            for step in range(ROLLOUT):
                """
                开始在envs采样，对应dqn中的batchsample
                
                """
                with torch.no_grad():
                    actions = model.get_action(model.rollouts.observations[step])
                cpu_actions = actions.view(-1,action_dim).cpu().data.numpy()

                if NUMACTOR==1:
                    cpu_actions=cpu_actions[0]
                    obs, reward, done, _ , _ = envs.step(cpu_actions)
                    reward=np.array(reward, dtype=np.float32)
                    if done:
                        done=1
                    else:
                        done=0
                    done = np.array([done], dtype=np.float32)
                else:
                    obs, reward, done, _ = envs.step(cpu_actions)
                
                episode_rewards += reward
                masks = 1. - done.astype(np.float32)
                final_rewards *= masks
                #episode是累计奖励,fianl直接用最后episode的奖励
                final_rewards += (1. - masks) * episode_rewards
                episode_rewards *= masks

                rewards = torch.from_numpy(reward.astype(np.float32)).view(-1, 1).to(device)
                masks = torch.from_numpy(masks).to(device).view(-1, 1)
                current_obs *= masks
                update_current_obs(obs)
                model.rollouts.insert(current_obs, actions.view(-1, action_dim), rewards, masks)
                if NUMACTOR!=1 and torch.any(masks == 0).item():
                    non_zero_rewards = final_rewards[final_rewards != 0]
                    writer.add_scalar('reward', np.mean(non_zero_rewards), global_step=i)

            e=time.time()
            if i%100==0:
                print(f"rollout: {e-s}")

            state,action,next_state,mask,reward,exp=model.rollouts.compute_returns(GAMMA)
            #采样完毕放入回放池
            data=(np.copy(state), np.copy(action),np.copy(next_state), np.copy(mask), np.copy(reward),np.copy(exp))
            model.mem.push(data)
            
            if NUMACTOR==1 and masks==0:
                current_obs,done= envs.reset(seed=SEED),False
                current_obs = torch.tensor(current_obs[0]).view(NUMACTOR, -1).to(device)
                model.rollouts.observations[-1].copy_(current_obs)
                if done:
                    done=1
                else:
                    done=0
                done = np.array([done], dtype=np.float32)
                masks = 1. - done.astype(np.float32)
                masks = torch.from_numpy(masks).to(device).view(-1, 1)
                model.rollouts.masks[-1].copy_(masks)
                print(f"EPISODE: {int(i/100)} reward is {np.mean(final_rewards):.2f} mid is {np.median(final_rewards):.2f}")
                writer.add_scalar('reward', np.mean(final_rewards), global_step=i)
            
            if (len(model.mem)*NUMACTOR*ROLLOUT)<MIN or (len(model.mem)*NUMACTOR*ROLLOUT)<MINIBATCHSIE:
                model.rollouts.after_update()#state=nextstate
                continue
            s=time.time()
            value_loss, action_loss,q_loss = model.learn(MINIBATCHSIE)
            e=time.time()
            if i%100==0:
                print(f"train: {e-s}")
            model.rollouts.after_update()#state=nextstate
            writer.add_scalar('value_loss', value_loss, global_step=i)
            writer.add_scalar('action_loss', action_loss, global_step=i)
            writer.add_scalar('q_loss', q_loss, global_step=i)
    if not TEST:
        model.save_train_model()
        print(EPISODES)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, render=False)
        print(mean_reward, std_reward)
        envs.close()
    else:
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=3, render=False)
        print(mean_reward, std_reward)
        env.close()

    
    
"""
要求：
1、并行的采样，放入回放池
并行有什么好处？
采样速度快




SAC有几个特点？
1、软更新
不直接复制Vnetwork的参数，而是一次复制一部分的大小，间隔训练周期一般是1-2次
硬更新，直接复制所有的参数，间隔是1000左右

2、双网络
可以是双Q+双V
这个双Q不好插入啊，主要他是v网络和q网络互相更新的....


3、熵值的限制
对动作的概率进行一个限制，保持一个高熵值的状态，鼓励随机



只要都做到了，是不是就叫sac
但是他们都是在A2C的框架下完成的

总结：
1、连续动作的任务刚刚开始学习
参考1：https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition

复现效果不理想，他的计算loss和动作的方法，有点问题（感觉），论文里面的maxentropy的方法没看到哪里有，还有双Q，targetQ也没体现
其实是我的回放池写错了，导致了复现不好
效果超过了baseline3的水平




参考1：https://github.com/higgsfield-ai/higgsfield/blob/main/higgsfield/rl/rl_adventure_2/7.soft%20actor-critic.ipynb
待验证


"""


if __name__ == '__main__':
    main()
    