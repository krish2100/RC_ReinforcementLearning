import numpy as np
import itertools

EPS = 1e-8

def generate_trajectory(env, agent):
    env.reset()    
    trajectory = []
    
    done = False
    agent.update_mode(eval_mode=True)
    
    while not done:
        cur_state = env.state
        next_actions = env.get_actions(cur_state)
        
        action = agent.get_action(env.query, cur_state, next_actions)
        next_state, done, reward = env.step(action)
        
        trajectory.append((cur_state, next_actions, action, reward))

    agent.update_mode(eval_mode=False)
    
    return trajectory


def get_maxpathidx(top_paths):
    i = 0
    for path in top_paths:
        if not path:
            return i
        i=i+1
    return i

def get_topk(end_nodes, idx_nodes, scores, beam_width=3):
    if len(scores) < beam_width:
        maxlen = len(scores)
    else:
        maxlen = beam_width
    return sorted(list(zip(end_nodes,idx_nodes,scores)), reverse=True, key=lambda x: x[2])[:maxlen]


def UpdateBeams(beam_candidates, paths, scores):
    
    NewBeams =[[] for x in range(len(beam_candidates))]
    NewScores = np.zeros(len(beam_candidates))

    #print(paths)
    k=0
    for end_node, idx, score in beam_candidates:
        #print(paths[idx])
        NewBeams[k].extend(paths[idx])
        #print(paths[idx])
        NewBeams[k].append(end_node)
        #print(NewBeams)
        NewScores[k] = score
        k = k + 1
    
    return NewBeams,NewScores
        

def beam_search(env,agent,beam_width=3):
    # env.reset()
    init_state = env.state
    init_actions = env.get_actions(init_state)
    
    agent.update_mode(eval_mode=True)
    action_proba = agent.get_action_proba(env.query, init_state, init_actions)
    action_proba = action_proba.cpu().detach().numpy()
    
    top_paths = [ [] for i in range(beam_width)]
    top_scores = np.zeros(beam_width, dtype = float)
    
    joined_list = sorted(list(zip(init_actions,np.log(action_proba))), reverse=True, key=lambda x: x[1])
    
    for i in range(beam_width):
        if i < len(joined_list):
            top_scores[i] = joined_list[i][1]
            top_paths[i].append(joined_list[i][0])
        else:
            top_scores[i] = -np.inf
            
    
    # print(top_paths)
            
    
    for i in range(env.max_steps-1):
        path_idx = get_maxpathidx(top_paths)
        top_scores_list = [[] for x in range(path_idx)]
        end_nodes_list = [[] for x in range(path_idx)]
        idx_nodes_list = [[] for x in range(path_idx)]
        
        for j in range(path_idx):
            next_actions = env.get_actions(top_paths[j][i])

            action_proba = agent.get_action_proba(env.query, top_paths[j][i], next_actions)
            action_proba = action_proba.cpu().detach().numpy()
            # print(action_proba)
            multiplier = top_scores[j] if top_scores[j] != float("-inf") else 0
            log_prob = np.log(action_proba+EPS) + multiplier*np.ones(action_proba.shape)

            end_nodes_list[j].extend(next_actions)
            idx_nodes_list[j].extend([j for x in range(len(next_actions))])
            top_scores_list[j].extend(log_prob)
        
        beam_candidates =  get_topk(list(itertools.chain(*end_nodes_list)),
                                    list(itertools.chain(*idx_nodes_list)),
                                    list(itertools.chain(*top_scores_list)),
                                    beam_width) 
        
        # print(beam_candidates)
        
        top_paths,top_scores = UpdateBeams(beam_candidates, top_paths, top_scores)
        
        # print(top_paths)
        # print(top_scores)

    return top_paths, top_scores