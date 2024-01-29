import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import gymnasium as gym
import datetime
import time
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import collections
from torch.distributions import Normal
from lib.kfac import KFACOptimizer
from lib.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from lib.dummy_vec_env import DummyVecEnv
from typing import Union,Dict


file_name = 'ACKTR'
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

new_directory1 = f'model/{file_name}'
new_directory2 = f'log/{file_name}'

model = os.path.join(current_directory, new_directory1).replace('\\', '/')
log = os.path.join(current_directory, new_directory2).replace('\\', '/')

#hyparameter
LR = 0.0003
GAMMA = 0.99
GAE=0.95
MODEL_PATH = model
SAVE_PATH_PREFIX = log
TEST = True
SEED=0
CLIP=0.2
PPO=10
MINIBATCHSIE=8
ROLLOUT=2048
NUMACTOR=8
EPISODES =int(1e6/(ROLLOUT/MINIBATCHSIE))              # 训练的更新步数，真实更新步数：EPISODES*NUMACTOR*ROLLOUT
ENLOSSWEIGHT=0.01
VALUELOSSWEIGHT=0.5
ACTIONLOSSWEIGHT=1.0
GRAD=0.5
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
file_name = "CTRPO"
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
    在ppo里面选择确定性动作
    
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
    
    
class Critic(nn.Module):
    """
    目的是求状态价值
    
    """
    def __init__(self, state_dim=state_dim,init_w=3e-3):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        
        self.l3.weight.data.uniform_(-init_w, init_w)
        self.l3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        q = F.relu(self.l1(state))
        q = F.relu(self.l2(q))
        return self.l3(q)
    
    
class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, device):
        self.observations = torch.zeros(num_steps + 1, num_processes, obs_shape).to(device)
        self.rewards = torch.zeros(num_steps, num_processes, 1).to(device)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1).to(device)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1).to(device)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1).to(device)
        self.actions = torch.zeros(num_steps, num_processes, action_dim).to(device, torch.long)
        self.masks = torch.ones(num_steps + 1, num_processes, 1).to(device)
        self.z = torch.zeros(num_steps, num_processes, action_dim).to(device)
        self.gae_tau=GAE

        self.num_steps = num_steps
        self.step = 0

    def insert(self, current_obs, action,action_log_prob, value_pred, reward, mask,z):
        self.observations[self.step + 1].copy_(current_obs)
        self.actions[self.step].copy_(action)
        self.action_log_probs[self.step].copy_(action_log_prob)
        self.value_preds[self.step].copy_(value_pred)
        self.rewards[self.step].copy_(reward)
        self.masks[self.step + 1].copy_(mask)
        self.z[self.step].copy_(z)

        
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, gamma):
        #如果要放入大的回放池在这里做,把里面的值取出来，用list的append去放，取出来，得用循环，然后再用torch包起来
        #2种方法：1、不改这个的基础上，在这里放入回放池，在取样的时候用torch.stack函数把对应的属性放到一起，在交出去 2、全部用np实现
        self.value_preds[-1] = next_value
        gae=0
        for step in reversed(range(self.rewards.size(0))):
            delta= self.rewards[step] + self.value_preds[step + 1] * gamma * self.masks[step + 1] - self.value_preds[step]
            gae=delta+GAMMA*self.gae_tau*self.masks[step+1]*gae
            self.returns[step]=gae+self.value_preds[step]
            
    def feed_forward_generator(self, advantages, num_mini_batch):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        assert batch_size >= num_mini_batch, (
            f"PPO requires the number processes ({num_processes}) "
            f"* number of steps ({num_steps}) = {num_processes * num_steps} "
            f"to be greater than or equal to the number of PPO mini batches ({num_mini_batch}).")
        mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
        for indices in sampler:
            observations_batch = self.observations[:-1].view(-1,
                                        *self.observations.size()[2:])[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,1)[indices]
            adv_targ = advantages.view(-1, 1)[indices]
            old_z = self.z.view(-1,action_dim)[indices]

            yield observations_batch, actions_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ,old_z
            
            
def normal_entropy(std):
    
    distentroy= 0.5 * torch.log(2 * torch.tensor(np.pi) * torch.tensor(np.e) * std**2)
    return distentroy.mean().to(device)
            
class cppo():
    def __init__(self):
        super(cppo, self).__init__()
        self.ACTOR = Actor().to(device)
        self.CRITIC = Critic().to(device)
        self.optimizer_c = torch.optim.Adam(self.CRITIC.parameters(), lr=LR)
        self.optimizer_a = KFACOptimizer(self.ACTOR, lr=LR)
        self.loss_func = nn.MSELoss()
        self.epsilon=1e-6
        self.clip_param=CLIP
        self.grad_norm_max=GRAD
        self.num_agents = NUMACTOR
        self.ppo_epoch=PPO
        self.num_mini_batch=MINIBATCHSIE
        self.rollout = ROLLOUT
        self.action_loss_weight=ACTIONLOSSWEIGHT
        self.value_loss_weight = VALUELOSSWEIGHT
        self.entropy_loss_weight = ENLOSSWEIGHT
        
        self.rollouts = RolloutStorage(self.rollout, self.num_agents,
            state_dim, action_dim, device)
        
    def get_action(self, s):
        """
        这里直接使用学习到的均值，在更新loss的时候在用采样的动作当作概率
        
        """
        s = torch.tensor(s, dtype=torch.float).view(-1,state_dim).clone().detach().to(device)
        mean,logstd = self.ACTOR(s)
        values = self.CRITIC(s)
        std = logstd.exp()
        normal = Normal(mean, std)
        z      = normal.sample()
        action = torch.tanh(z)
        logprob=normal.log_prob(z) - torch.log(1 - action.pow(2) + self.epsilon)
        if not TEST:
            logprob = logprob.sum(dim=1, keepdim=True)
        return values,action,logprob,z
    
    def get_values(self, s):
        values = self.CRITIC(s)
        
        return values
    
    def evaluate_actions(self, s,actions,old_z):
        
        mean,logstd = self.ACTOR(s)
        values = self.CRITIC(s)
        std=logstd.exp()
        normal = Normal(mean, std)
        logprob=normal.log_prob(old_z) - torch.log(1 - actions.pow(2) + self.epsilon)
        logprob = logprob.sum(dim=1, keepdim=True)
        dist_entropy = normal_entropy(std)
        
        return values, logprob, dist_entropy
    
    def compute_loss(self, sample):
        observations_batch, actions_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ,old_z = sample
        
        values, action_log_probs, dist_entropy = self.evaluate_actions(observations_batch,actions_batch,old_z)

        ratio = action_log_probs
        action_loss = ratio * adv_targ
        action_loss = -action_loss.mean()

        value_loss = F.mse_loss(return_batch, values)
        
        action_loss = self.action_loss_weight*action_loss- self.entropy_loss_weight * dist_entropy
        if torch.any(torch.isnan(action_loss)+torch.isinf(action_loss)):
            print("1")
        value_loss = self.value_loss_weight * value_loss
        
        
        return action_loss, value_loss,action_log_probs
    
    def learn(self, rollout):
        advantages = rollout.returns[:-1] - rollout.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        for e in range(self.ppo_epoch):
            data_generator = rollout.feed_forward_generator(
                advantages, self.num_mini_batch)
            for sample in data_generator:
                action_loss, value_loss,action_log_probs = self.compute_loss(sample)
           
                
                if self.optimizer_a.steps % self.optimizer_a.Ts ==0:
                    self.optimizer_a.zero_grad()
                    pg_fisher_loss = -action_log_probs.mean()
                    self.optimizer_a.acc_stats = True
                    pg_fisher_loss.backward(retain_graph=True)
                    self.optimizer_a.acc_stats = False
                
                self.optimizer_a.zero_grad()
                action_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ACTOR.parameters(), self.grad_norm_max)
                self.optimizer_a.step()
                
                
                self.optimizer_c.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.CRITIC.parameters(), self.grad_norm_max)
                self.optimizer_c.step()
                
                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
        value_loss_epoch /= (self.ppo_epoch * self.num_mini_batch)
        action_loss_epoch /= (self.ppo_epoch * self.num_mini_batch)
        
        return value_loss_epoch, action_loss_epoch
    
    
    def save_train_model(self):
        torch.save(self.ACTOR.state_dict(), f"{MODEL_PATH}/ACTOR.pth")
        
    def load_net(self, file):
        self.ACTOR.load_state_dict(torch.load(f"{file}/ACTOR.pth"))
            
    def predict(self,observation: Union[np.ndarray, Dict[str, np.ndarray]],state,episode_start,deterministic):
            
        n_batch = observation.shape[0]

        action = np.array([np.array(self.get_action(observation)[1][0].view(action_dim).cpu().numpy()) for _ in range(n_batch)])

        
        return action,state
    
def main():
    model=cppo()
    
    if  not TEST:
        torch.set_num_threads(1)
        envs=[env for i in range(NUMACTOR)]
        envs=SubprocVecEnv(envs) if NUMACTOR > 1 else DummyVecEnv(envs)
        
        obs_shape = envs.observation_space.shape
        current_obs = torch.zeros(NUMACTOR, *obs_shape,
                        device=device, dtype=torch.float)
        def update_current_obs(obs):
            shape_dim0 = envs.observation_space.shape[0]
            obs = torch.from_numpy(obs.astype(np.float64)).to(device)
            current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
            current_obs[:, -shape_dim0:] = obs
        
        obs = envs.reset()
        update_current_obs(obs)
        model.rollouts.observations[0].copy_(current_obs)
        writer = SummaryWriter(f'{SAVE_PATH_PREFIX}/{datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}-ACKTR-{EPISODES}')
    episode_rewards = np.zeros(NUMACTOR, dtype=np.float64)
    final_rewards = np.zeros(NUMACTOR, dtype=np.float64)
    if TEST:
        model.load_net(MODEL_PATH)
        
        
    for i in range(EPISODES):
        
        if TEST:
            print("EPISODE: ", i)
            state,info=env.reset()
            ep_reward = 0
            while(True):
                _,action,_,_ = model.get_action(state)
                action = action.view(action_dim).cpu().numpy()
                next_state, reward, done, truncated, info = env.step(action)  # observe next state and reward
                state=next_state
                ep_reward+=reward
                env.render()
                if done:
                    print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
                    break
                
        if not TEST:
            if i%10==0:
                non_zero_rewards = final_rewards[final_rewards != 0]
                print(f"EPISODE: {int(i/10)} reward is {np.mean(non_zero_rewards):.2f} mid is {np.median(non_zero_rewards):.2f}")
            s=time.time()
            for step in range(ROLLOUT):
                """
                开始在envs采样，对应dqn中的batchsample
                
                """
                with torch.no_grad():
                    values, actions, action_log_prob,z = model.get_action(model.rollouts.observations[step])
                cpu_actions = actions.view(-1,action_dim).cpu().numpy()
        
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

                model.rollouts.insert(current_obs, actions.view(-1,action_dim), action_log_prob, values, rewards, masks,z)
                if torch.any(masks == 0):
                    non_zero_rewards = final_rewards[final_rewards != 0]
                    writer.add_scalar('reward', np.mean(non_zero_rewards), global_step=i)
            e=time.time()
            if i%10==0:
                print(f"epsiod:{int(i/10)} rollout: {e-s}")
            with torch.no_grad():
                next_value = model.get_values(model.rollouts.observations[-1])
                
            model.rollouts.compute_returns(next_value, GAMMA)
            s=time.time()
            value_loss, action_loss = model.learn(model.rollouts)
            e=time.time()
            if i%10==0:
                print(f"epsiod:{int(i/10)} train: {e-s}")
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





   
if __name__ == '__main__':
    main()