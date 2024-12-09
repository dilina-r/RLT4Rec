import os
import sys
import torch
import numpy as np
# from tqdm import tqdm
from pprint import pprint
from DataLoaderGPT import DataLoader
import json
from model_gpt import GPT, GPTConfig
from utils import *


os.environ["CUDA_VISIBLE_DEVICES"]="0"


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
torch.device(device)


data_directory = "data"
dataset = "netflixM"
nyms = 8
max_count = 25
model_output_path = "saved_models/rlt4rec_{}{}.pth".format(dataset, nyms)
val_file = f"data/test_users/{dataset}{nyms}_test.json"


context_length = max_count

dataloader = DataLoader(data_directory=data_directory,
                        dataset=dataset, 
                        nyms=nyms, 
                        max_ep_len=max_count)
num_actions = dataloader.get_num_actions()
states_dim = dataloader.get_states_dim()


n_head= 4
n_layer = 2
n_embd = 128
model_type = 'reward_conditioned'
mconf = GPTConfig(num_actions, context_length*2,
                    state_dim = states_dim, n_layer=n_layer, n_head=n_head, n_embd=n_embd, model_type=model_type, max_timestep=2*max_count)
model = GPT(mconf)
# print(model)
print(f'Num model Parameters: {get_n_params(model=model)}')
model.to(device)



lr = 0.005
decay = 1e-5
optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=decay
    )
grad_norm_clip = 1.0

best_reward = 0.0
target_return = 5.0

batch_size = 128
epochs = 30
num_batches = 1000

train_info = {
    "dataset" : dataset,
    "num groups" : nyms,
    "optimizer" : optimizer,
    "batch_size" : batch_size,
    "num_batches" : num_batches,
    "epochs" : epochs,
    "target_return" : target_return,
    "num items" : num_actions,
    "n_head" : n_head,
    "n_embd" : n_embd,
    "n_layer" : n_layer
}

pprint(train_info)

for epoch in range(0, epochs):
    losses = []
    model.train()
    for iter in range(0, num_batches):
        actions, rewards, timesteps, attention, targets = dataloader.get_batch_online(batch_size, max_count, train=True)
        # actions, rewards, timesteps, attention, targets = dataloader.get_batch_offline(batch_size)

        targets = torch.clone(actions)
        preds, loss = model.forward(actions=actions.to(device), 
                                    targets=targets.to(device), 
                                    ratings=rewards.to(device), 
                                    timesteps=timesteps.to(device),
                                    attention=attention.to(device))
        
        optimizer.zero_grad()
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
        optimizer.step()

        losses.append(loss.detach().cpu().item())
        
        if iter % 1 == 0:
            print("Epoch {} -- iter {:0d}/{:0d} -- loss:{:0.04f}".format(
            epoch, iter, num_batches, np.mean(losses)))
            sys.stdout.write("\033[F")
    
    model.eval()

    mean_reward, results = dataloader.evaluate_file(model, test_file=val_file, episode_len=25, target_return=5.0, device=device)

    if mean_reward > best_reward:
        best_reward = mean_reward
        save(model, model_output_path)
    print("Epoch {} -- iter {:0d}/{:0d} -- loss:{:0.03f}-- eval:{:0.02f} -- best_eval:{:0.02f}".format(
            epoch, iter, num_batches, np.mean(losses), mean_reward, best_reward))


#### Run on simulated online users 
model, _ = load(model_path=model_output_path, model=model)
num_test_users = 200 # no. of test users per group.... total users = num_test_users*num_groups
ep_len = 25 # no. of recommendations per test user
mean_reward, results = dataloader.evaluate_online(model=model, 
                                                  num_test_users=num_test_users,
                                                  episode_len=ep_len, 
                                                  target_return=target_return, 
                                                  device=device)
print(f"Test on Simulated Online Users -- Mean Reward @{ep_len} = {np.round(mean_reward, 2)}")