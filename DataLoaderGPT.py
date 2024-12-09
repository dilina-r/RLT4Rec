import os
import numpy as np
import pandas as pd
from numpy.random import choice
import scipy.stats
import math
import random
import torch
import json
from user_model import UserModel

class DataLoader():

    SCALE = 1.0

    def __init__(self, data_directory, dataset, nyms, discount=1.0, max_ep_len=25, scale=1.0) -> None:
        self.data_directory = data_directory
        self.dataset = dataset
        self.nyms = nyms
        self.discount = discount
        self.max_ep_len = max_ep_len
        self.scale = scale

        # mu_filename = os.path.join(self.root_dir, "data", "mu_{}{}.csv".format(self.dataset, self.nyms))
        # sigma_filename = os.path.join(self.root_dir, "data", "sigma_{}{}.csv".format(self.dataset, self.nyms))
        # num_filename = os.path.join(self.root_dir, "data", "num_{}{}.csv".format(self.dataset, self.nyms))


        # self.mu = np.loadtxt(mu_filename, delimiter=",", dtype=np.float64)
        # self.sigma = np.sqrt(np.loadtxt(sigma_filename, delimiter=",", dtype=np.float64))
        # self.num = np.loadtxt(num_filename, delimiter=",", dtype=int)

        self.um = UserModel(os.path.join(self.data_directory, 'groups'), dataset=dataset, nyms=nyms, min_rating=1, max_rating=5)
        self.num_items = self.um.get_num_actions()

        self.train_data = None


    def get_num_actions(self):
        return self.num_items
    
    def get_states_dim(self):
        return self.nyms

    
    def calc_prob(self, items, ratings):

        sums = [0.0] * self.nyms
        prods = [1.0] * self.nyms
        probs = [0.0] * self.nyms
        # print(len(items), len(ratings), self.mu.shape, self.sigma.shape)
        # print(ratings)
        for i in range(0, len(items)):
            for g in range(0, self.nyms):
                sums[g] += ( (ratings[i] - self.mu[g][items[i]]) * (ratings[i] - self.mu[g][items[i]]) ) / ( self.sigma[g][items[i]] * self.sigma[g][items[i]] + 1e-10)
                prods[g] *= self.sigma[g][items[i]] + 1e-10
        
        sum_prob = 1e-10
        # print(np.sqrt(np.power(2 * np.pi, len(items))))
        for g in range(0, self.nyms):
            probs[g] = np.exp(-sums[g]/2.0) / (prods[g])
            sum_prob += probs[g]
        
        for g in range(0, self.nyms):
            probs[g] = probs[g] / sum_prob

        # print(probs)
        
        return probs
    

    def __get_states2(self, user_group, items):
        iters = len(items)
        num_groups = self.mu.shape[0]
        states = np.zeros(shape=(iters, num_groups))

        states[0, :] = 1.0/num_groups


        ratings = []
        for item in items:
            r = min(max(0, np.random.normal(self.mu[user_group, item], self.sigma[user_group, item])), 5)
            # r = np.random.normal(mu[user_group, item], sigma[user_group, item])
            ratings.append(round(r, 2))

        
        
        for iter in range(0, iters):
            probs = self.calc_prob(items[:iter+1], ratings=ratings[:iter+1])
            if iter < (iters-1):
                states[iter+1, :] = probs
        
        return ratings, states
    
    def get_rating(self, usergroup, item):
        r = min(max(1, np.random.normal(self.mu[usergroup, item], self.sigma[usergroup, item])), 5)
        return round(r, 2)

    def get_data(self, token_len, usergroup=None, popular=False):
        if usergroup is not None:
            assert(usergroup < self.nyms)
            g = usergroup
        else:
            g = random.randint(0, self.nyms-1)

        items = range(0, self.mu.shape[1])

        if popular:
            n = self.num[g, :]
            n = n / np.sum(n)
            # print(m.shape, s.shape, n.shape)
            #### Draw items based on popularity
            draw = choice(items, token_len, p=n, replace=False)
        else:
            #### Draw items uniformly
            draw = choice(items, token_len, replace=False)

        ratings = []
        for item in draw:
            ratings.append(self.get_rating(usergroup=g, item=item))

        actions = np.array(draw)
        timestamps  = np.array(range(1, token_len+1))
        rewards = np.array(ratings)

        return actions, rewards, timestamps
    
    def get_discounted_rtg(self, rewards):
        rtgs = []

        discount = 1.0
        for i in range(0, len(rewards)):
            r = 0
            for j in range(i, len(rewards)):
                r+= (rewards[j]* max(0.0, discount))
                discount = discount - self.discount
                if discount < 0:
                    break
            rtgs.append(r)
        return np.array(rtgs)
    
    def get_cummulative_rtg(self, rewards):
        rtgs = []

        discount = 1.0
        for i in range(0, len(rewards)):
            r = 0
            for j in range(i, len(rewards)):
                r+= (rewards[j]* max(0.0, discount))
            rtgs.append(r)
        return np.array(rtgs)

    def get_batch_offline(self, batch_size=256, shuffle=True):

        if self.train_data is None:
            self.train_data = pd.read_pickle(os.path.join(self.data_directory, 'train', f'{self.dataset}{self.nyms}_train.df'))  # read data statistics, includeing state_size and item_num

        a, targets, rtg, mask, timesteps = [], [], [], [], []

        batch = self.train_data.sample(n=batch_size)

        # print(batch)

        for index, row in batch.iterrows():
            user_items = list(row['items'])
            user_ratings = list(row['ratings'])
            seq_len = len(user_items)

            n = min(seq_len, self.max_ep_len)
            tlen = random.randint(1, n)

            # if shuffle:
            idx = list(range(0, seq_len))
            idx = random.sample(idx, tlen)

            items = [user_items[i] for i in idx]
            ratings = [user_ratings[i] for i in idx]

            a.append(np.array(items).reshape(1, -1, 1))
            targets.append(np.array(items).reshape(1, -1, 1))
            rtg.append(np.array(ratings).reshape(1, -1, 1))
            timesteps.append(max(1, tlen-self.max_ep_len))


            tlen = a[-1].shape[1]

            a[-1] = np.concatenate(
                [a[-1], np.ones((1, self.max_ep_len - tlen, 1)) * 0.0],
                axis=1,
            )
            targets[-1] = np.concatenate(
                [targets[-1], np.ones((1, self.max_ep_len - tlen, 1)) * -1],
                axis=1,
            )
            
            rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, self.max_ep_len - tlen, 1))], axis=1) / DataLoader.SCALE
            mask.append(np.concatenate([np.ones((1, tlen)), np.zeros((1, self.max_ep_len - tlen))], axis=1))

        a = torch.squeeze(torch.from_numpy(np.concatenate(a, axis=0)).long())
        targets = torch.squeeze(torch.from_numpy(np.concatenate(targets, axis=0)).long())
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).float()
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).float()
        timesteps = torch.from_numpy(np.array(timesteps).reshape(-1, 1, 1)).long()


        return a, rtg, timesteps, mask, targets



    def get_batch_online(self, batch_size, token_len, train=True, usergroup=None):

        a, targets, rtg, mask, timesteps = [], [], [], [], []

        for b in range(0, batch_size):

            g = random.randint(0, self.nyms-1)

            tlen = random.randint(1, token_len)
            
            items, rewards = self.um.get_sequence(usergroup=g, sequence_len=tlen, sampling='uniform', replace=False)

            a.append(np.array(items).reshape(1, -1, 1))
            targets.append(np.array(items).reshape(1, -1, 1))
            rtg.append(np.array(rewards).reshape(1, -1, 1))
            timesteps.append(max(1, tlen-self.max_ep_len))


            tlen = a[-1].shape[1]

            a[-1] = np.concatenate(
                [a[-1], np.ones((1, self.max_ep_len - tlen, 1)) * 0.0],
                axis=1,
            )
            targets[-1] = np.concatenate(
                [targets[-1], np.ones((1, self.max_ep_len - tlen, 1)) * -1],
                axis=1,
            )
            
            rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, self.max_ep_len - tlen, 1))], axis=1) / DataLoader.SCALE
            mask.append(np.concatenate([np.ones((1, tlen)), np.zeros((1, self.max_ep_len - tlen))], axis=1))

        a = torch.squeeze(torch.from_numpy(np.concatenate(a, axis=0)).long())
        targets = torch.squeeze(torch.from_numpy(np.concatenate(targets, axis=0)).long())
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).float()
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).float()
        timesteps = torch.from_numpy(np.array(timesteps).reshape(-1, 1, 1)).long()

        return a, rtg, timesteps, mask, targets
    
    # def preprocess(self, actions, rewards):
        

    #     # if self.future:
    #     #     rewards = self.get_cummulative_rtg(rewards)

    #     tlen = len(rewards)
        
    #     rewards = np.array(rewards).reshape(1, -1, 1)
    #     timesteps = np.array(range(0, tlen)).reshape(1, -1)

    #     if len(actions) > 0:
    #         actions = np.array(actions).reshape(1, -1)
    #         actions = np.concatenate(
    #             [actions, np.ones((1, self.max_ep_len - actions.shape[1])) * 0.0],
    #             axis=1,
    #         )
    #     else:
    #         actions = np.zeros(shape=(1, self.max_ep_len))

    #     rewards = np.concatenate([rewards, np.zeros((1, self.max_ep_len - rewards.shape[1], 1))], axis=1) / self.scale
    #     timesteps = np.concatenate([timesteps, np.zeros((1, self.max_ep_len - timesteps.shape[1]))], axis=1)
    #     mask = np.concatenate([np.ones((1, tlen)), np.zeros((1, self.max_ep_len - tlen))], axis=1)

    #     # print(states.shape, actions.shape, rewards.shape, timesteps.shape, mask.shape)

    #     atari_timesteps = torch.from_numpy(np.array([tlen]).reshape(1, -1, 1)).long()
        
        
    #     actions = torch.from_numpy(actions).long()
    #     rewards = torch.from_numpy(rewards).float()
    #     timesteps = torch.from_numpy(timesteps).long()
    #     mask = torch.from_numpy(mask).float()


    #     # print(states.shape, actions.shape, rewards.shape, timesteps.shape, mask.shape)

    #     return actions, rewards, timesteps, mask, atari_timesteps
    
 


    def evaluate_file(self, model, test_file, episode_len=25, target_return=5.0, device='cpu'):
        mean_ratings = 0.0
        data = []
        with open(test_file) as user_file:
            file_contents = user_file.read()
            parsed_json = json.loads(file_contents)
        
        count = 0

        for d in parsed_json:
            g = d["group"]
            items = []
            ratings = []
            for ep in range(0, episode_len):
                ratings.append(target_return)
                items.append(0)
                # timesteps = list(range(0, ep+1))
                # atari_timesteps =[0]
                timesteps = [max(1, ep+1-self.max_ep_len)]
                # actions, rewards, timesteps, mask, atari_timesteps = self.preprocess(actions=items, rewards=ratings)
                
                preds = model.get_action(items, ratings, timesteps=timesteps, device=device)
                action_preds = preds[-1, -1, :]
                action = torch.argmax(action_preds).cpu().item()
                while(action in items):
                    action_preds[action] = -10.0
                    action = torch.argmax(action_preds).cpu().item()
                
                items[-1] = action
                ratings[-1] = np.round(d["ratings"][action])
            d = {"group": g,
                "items": list(items),
                "ratings": list(ratings)}
            data.append(d)
            mean_ratings += np.mean(ratings)
            count += 1

        # mean_ratings = mean_ratings / (1.0 * count)

        return mean_ratings / (1.0 * count), data
    

    def evaluate_online(self, model, num_test_users=200, episode_len=25, target_return=5.0, device='cpu'):
        mean_ratings = 0.0
        data = []
        # with open(test_file) as user_file:
        #     file_contents = user_file.read()
        #     parsed_json = json.loads(file_contents)
        
        count = 0
        with torch.no_grad():
            for g in range(0, self.nyms):
                for t in range(num_test_users):
                    items = []
                    ratings = []
                    for ep in range(0, episode_len):
                        ratings.append(target_return)
                        items.append(0)
                        # timesteps = list(range(0, ep+1))
                        # atari_timesteps =[0]
                        timesteps = [max(1, ep+1-self.max_ep_len)]
                        # actions, rewards, timesteps, mask, atari_timesteps = self.preprocess(actions=items, rewards=ratings)
                        
                        preds = model.get_action(items, ratings, timesteps=timesteps, device=device)
                        action_preds = preds[-1, -1, :]
                        action = torch.argmax(action_preds).cpu().item()
                        while(action in items):
                            action_preds[action] = -10.0
                            action = torch.argmax(action_preds).cpu().item()
                        rating = self.um.get_rating(usergroup=g, item=action)
                        items[-1] = action
                        ratings[-1] = rating
                    d = {"group": g,
                        "items": list(items),
                        "ratings": list(ratings)}
                    data.append(d)
                    mean_ratings += np.mean(ratings)
                    count += 1

        # mean_ratings = mean_ratings / (1.0 * count)

        return mean_ratings / (1.0 * count), data





