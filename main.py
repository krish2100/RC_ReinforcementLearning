import numpy as np
import torch
from torch.distributions import Categorical
from pathlib import Path
from collections import deque
from tqdm import tqdm
from datasets import load_dataset

from DocGraphMultiHopper.DocumentGraph import DocumentGraph
from DocGraphMultiHopper.DocumentGraph.utils import sentence_featurizer

from DocGraphMultiHopper.rl import Env, Agent, PolicyNet
from DocGraphMultiHopper.rl.utils import generate_trajectory, beam_search

import csv
import pandas as pd

print("Reading dataset...")
wikihop = load_dataset("wiki_hop")
print("Read dataset.")

# print("Initializing environment...")
# env = Env(
#     dataset=wikihop,
#     subset="train", 
#     max_steps=10, 
#     seed=None,#42,
#     directed=True,
#     self_loop=True,
#     featurizer=sentence_featurizer,
# )
# print("Environment initialized.")

MODEL_PATH = Path("DocGraphMultihopperPolicy.pt")
LOG_PATH = Path("EpisodicReturn.log")

policy_net = PolicyNet()
if MODEL_PATH.exists():
    print(f"Loading from exisiting model at: {MODEL_PATH}")
    policy_net.load_state_dict(torch.load(MODEL_PATH))
    print("Model weights loaded.")

agent = Agent(policy_net=policy_net, eval_mode=False)

# print(generate_trajectory(env, agent))

train = False

if train:
    optimizer = torch.optim.Adam(agent.policy_net.parameters())

    num_episodes = 1500
    gamma = 0.99

    logging_rate = 1
    saving_rate = 50
    returns = list()
    # returns = deque(maxlen=100)

    for ep in tqdm(range(num_episodes)):
        loss, Gt = 0, 0
        trajectory = generate_trajectory(env, agent)
        baseline = np.mean(returns) if returns else 0

        for cur_state, action_list, action, reward in trajectory[::-1]:
            
            Gt = reward + gamma*Gt
            
            action_proba = agent.get_action_proba(env.query, cur_state, action_list)
            sampler = Categorical(action_proba)
            
            log_proba = -sampler.log_prob(torch.tensor([action_list.index(action)]))
            loss += log_proba*(Gt-baseline)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        returns.append(np.sum([i[-1] for i in trajectory]))

        if (ep+1)%logging_rate == 0:
            with open(LOG_PATH, "a") as f:
                f.write(f"{ep+1},{returns[-1].round(3)},{np.mean(returns).round(3)}\n")
            # print("Episode: {:4d}\tAvg. Return: {:4.2f}".format(ep+1, np.mean(returns)))

        if (ep+1)%saving_rate == 0:
            torch.save(agent.policy_net.state_dict(), MODEL_PATH)

else:
    eval_df = pd.DataFrame(
        columns=["subset", "supports", "question", "answer", 
            "candidates", "predicted_sent", "beam_score", "match"]
    )

    num_evals = 10
    # for i in tqdm(range(num_evals)):
    #     top_paths, top_scores = beam_search(env, agent, beam_width=3)
    #     eval_df = eval_df.append(
    #         dict(
    #             subset="train",
    #             supports=env.doc["supports"],
    #             question=env.query,
    #             answer=env.doc["answer"],
    #             candidates=env.doc["candidates"],
    #             predicted_sent=top_paths[0][-1],
    #             beam_score=top_scores[0],
    #             match=env.is_answer_node(top_paths[0][-1])
    #         ), 
    #         ignore_index=True,
    #     )
    #     env.reset()

    # train_match = eval_df.query(f"`subset` == 'train'")["match"].mean()
    # print(f"Train match = {train_match}")

    print("Initializing environment...")
    env = Env(
        dataset=wikihop,
        subset="validation", 
        max_steps=10, 
        seed=None,#42,
        directed=True,
        self_loop=True,
        featurizer=sentence_featurizer,
    )
    print("Environment initialized.")

    for i in tqdm(range(num_evals)):
        top_paths, top_scores = beam_search(env, agent, beam_width=3)
        eval_df = eval_df.append(
            dict(
                subset="val",
                supports=env.doc["supports"],
                question=env.query,
                answer=env.doc["answer"],
                candidates=env.doc["candidates"],
                predicted_sent=top_paths[0][-1],
                beam_score=top_scores[0],
                match=env.is_answer_node(top_paths[0][-1])
            ), 
            ignore_index=True,
        )
        env.reset()

    val_match = eval_df.query(f"`subset` == 'val'")["match"].mean()
    print(f"Val match = {val_match}")
    
    eval_df.to_csv("eval_df.csv", sep="\t", index=False, quoting=csv.QUOTE_NONNUMERIC)