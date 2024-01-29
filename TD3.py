import os
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
import gymnasium as gym
import gc
import math
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

file_name = 'TD3'
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

new_directory1 = f'model/{file_name}'
new_directory2 = f'log/{file_name}'

model = os.path.join(current_directory, new_directory1).replace('\\', '/')
log = os.path.join(current_directory, new_directory2).replace('\\', '/')

SEED=0

EPISODES=int(1e6) 
EXPNOISE=0.1
EXPLORE=1.0
#这个参数很重要，这是控制整个动作的随机性的，其实可以应用epsilongreedy的方法
#如果要保持动作的探索，就可以像sac一样，变成个可以学习的参数，这个参数应该具备的特征是：1、当目前状态稳定时应该寻求随机
#整个系统要具备对当前形势的一个判断，整局的累计奖励？极大化这个目标可以吗？
BATCH=512
ROLLOUT=8
NUMACTOR=8
GAMMA=0.99    
TAU=0.005
PLYNOISE=0.2
NCLIP=0.5  
PFREQ=2
GAE=0
TEST=True
MODEL_PATH = model
SAVE_PATH_PREFIX = log
MAX=int(1e5*NUMACTOR*ROLLOUT)
MIN=int(25e2*NUMACTOR*ROLLOUT)
if TEST:
    EPISODES=10
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
        """
        转换到-1-1
        """
        low = self.action_space.low
        high = self.action_space.high
        
        return 2 * (action - low) / (high - low) - 1
    
    def action(self, action):
        return self._reverse_action(action)



    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if TEST:
    env=gym.make("HalfCheetah-v4",render_mode="human")
else:
    env=gym.make("HalfCheetah-v4")
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

#网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1
    
    
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

    def insert(self, current_obs, action, reward, mask):
        self.observations[self.step + 1].copy_(current_obs)
        self.actions[self.step].copy_(action)
        # self.value_preds[self.step].copy_(value_pred)
        self.rewards[self.step].copy_(reward)
        self.masks[self.step + 1].copy_(mask)
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])

    @torch.no_grad
    def compute_returns(self, gamma):
        # self.value_preds[-1] = next_value
        # gae=0
        
        #多步的rewards
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
            s, a, n,m,R,e=self.mem[one]
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
    
    
class TD3():
    def __init__(self,state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        discount=GAMMA,
        tau=TAU,
        policy_noise=PLYNOISE,
        noise_clip=NCLIP,
        policy_freq=PFREQ) -> None:
        super(TD3,self).__init__()
        
        
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0
        
        self.rollout = ROLLOUT
        self.num_agents=NUMACTOR
        self.rollouts = RolloutStorage(self.rollout, self.num_agents,
            state_dim, action_dim, device)
        self.mem=ReplayBuffer()
        
    @torch.no_grad()
    def get_action(self, state):
        """
        这里的算target操作完全是多余的....甚至是错的，应为targetQ应该当时用完就扔
        只是凑个形式
        
        """
        action = self.actor(state)

        return action
        
    @torch.no_grad()
    def get_values(self, state,action=None):
        
        if action is not None:
            
            noise = (
                    torch.randn_like(action) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)
            action = (
                    action + noise
                ).clamp(-self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target(state, action)
            target_Q = torch.min(target_Q1, target_Q2)
        else:
            target_Q,action=self.get_action(state)
        
        return target_Q
    
    def compute_loss(self, sample):
        state, action,next_state,mask,reward,exp=sample
        
        # target_Q=ret
        with torch.no_grad():
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            target_a=self.actor_target(next_state)
            next_action = (target_a + noise).clamp(-self.max_action, self.max_action)
            
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + mask * (self.discount**exp) * target_Q


        current_Q1, current_Q2 = self.critic(state, action)
        
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        

        
        return critic_loss
    
    def learn(self, batch_size):
        self.total_it += 1
        
        sample=self.mem.sample(batch_size)
        state, action,next_state,mask,reward,exp=sample
        critic_loss=self.compute_loss(sample)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        
        if self.total_it % self.policy_freq == 0:
            
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            #这里存在显存没释放
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            return actor_loss.item(),critic_loss.item()

        return 0,critic_loss.item()
                
                
    def save_train_model(self):
        torch.save(self.actor.state_dict(), f"{MODEL_PATH}/ACTOR.pth")
        

    def load_net(self, file):
        self.actor.load_state_dict(torch.load(f"{file}/ACTOR.pth"))
            
    def predict(self,observation: Union[np.ndarray, Dict[str, np.ndarray]],state,episode_start,deterministic):
            
        n_batch = observation.shape[0]
        observation=torch.tensor(observation.astype(np.float32)).to(device)
        if not TEST:
            action = np.array([np.array(self.get_action(observation).cpu().item()) for _ in range(n_batch)])
        else:
            action = np.array([self.get_action(observation) for _ in range(n_batch)])
        
        return action,state
    
    
def main():
    model=TD3()
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
        writer = SummaryWriter(f'{SAVE_PATH_PREFIX}/{datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}-TD3-{EPISODES}')
    episode_rewards = np.zeros(NUMACTOR, dtype=np.float32)
    final_rewards = np.zeros(NUMACTOR, dtype=np.float32)
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
            seed=SEED+100
            state,info=env.reset(seed=seed)
            ep_reward = 0
            while(True):
                state=torch.from_numpy(state.astype(np.float32)).to(device)
                action = model.get_action(state)
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
                if (len(model.mem)*NUMACTOR)*ROLLOUT<MIN:
                    actions = np.array([env.action_space.sample() for _ in range(NUMACTOR)])
                    actions = torch.tensor(actions).to(device)
                    cpu_actions = actions.view(-1,action_dim).cpu().data.numpy()
                else:
                    
                    actions = model.get_action(model.rollouts.observations[step])
                    cpu_actions = actions.view(-1,action_dim).cpu().data.numpy()
                    #我这里可以把并行的探索利用起来
                    # temp=np.random.normal(0, max_action * EXPNOISE, size=action_dim)
                    exnoise = np.random.uniform(0, EXPLORE, size=NUMACTOR)
                    temp = np.random.normal(0, max_action * exnoise[:, np.newaxis], size=(NUMACTOR, action_dim))
                    cpu_actions=(cpu_actions+temp).clip(-max_action, max_action)
                
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
                print(f"EPISODE: {int(i/100)} rollout time{e-s}")
                
                
            s=time.time()    
            state,action,next_state,mask,reward,exp=model.rollouts.compute_returns(GAMMA)
            #采样完毕放入回放池
            data=(np.copy(state), np.copy(action), np.copy(next_state), np.copy(mask), np.copy(reward),np.copy(exp))
            model.mem.push(data)
            e=time.time()
            if i%100==0:
                print(f"EPISODE: {int(i/100)} insert time{e-s}")
            
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
            if (len(model.mem)*NUMACTOR*ROLLOUT)<=MIN or (len(model.mem)*NUMACTOR*ROLLOUT)<BATCH:
                model.rollouts.after_update()#state=nextstate
                continue
            
            s=time.time()    
            action_loss, value_loss = model.learn(BATCH)
            e=time.time()
            if i%100==0:
                print(f"EPISODE: {int(i/100)} train time{e-s}")
                
            model.rollouts.after_update()#state=nextstate
            writer.add_scalar('value_loss', value_loss, global_step=i)
            writer.add_scalar('action_loss', action_loss, global_step=i)
    if not TEST:
        model.save_train_model()
        print(EPISODES)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, render=False)
        print(mean_reward, std_reward)
        envs.close()
    else:
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, render=False)
        print(mean_reward, std_reward)
        env.close()



"""
这个是我的并行式的TD3


"""
        
    
if __name__ == '__main__':
    main()
        
        
        