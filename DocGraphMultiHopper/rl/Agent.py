import torch
from torch.distributions import Categorical

class Agent(object):
    def __init__(self, policy_net, eval_mode=True):
        self.policy_net = policy_net
        self.update_mode(eval_mode)
    
    def update_mode(self, eval_mode):
        self.eval_mode = eval_mode
        self.policy_net = (
            self.policy_net.eval() if eval_mode else self.policy_net.train()
        )
    
    def get_action_proba(self, query, cur_state, next_actions):
        context_emb = torch.cat((cur_state.embedding, query.embedding))
        action_emb = torch.stack([a.embedding for a in next_actions])
        
        if self.eval_mode:
            with torch.no_grad():
                action_proba = self.policy_net(context_emb, action_emb)
        else:
            action_proba = self.policy_net(context_emb, action_emb)
        
        return action_proba
    
    def get_action(self, query, cur_state, next_actions):
        action_proba = self.get_action_proba(query, cur_state, next_actions)                
        action_sampler = Categorical(action_proba)
        action_idx = action_sampler.sample()
        return next_actions[action_idx]
        # return random.choice(next_actions)