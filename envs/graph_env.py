import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class GraphEnv(gym.Env): # customized environment
    '''如何選兩個id?
    隨機選：
        任意挑
    非隨機選：
        透過現有網路產出的embedding(n2) 用Q找到最好的兩個
    '''
    """Hotter Colder
    The goal of hotter colder is to guess closer to a randomly selected number
    
    After each step the agent receives an observation of:
    0 - No guess yet submitted (only after reset)
    1 - Guess is lower than the target
    2 - Guess is equal to the target
    3 - Guess is higher than the target
    
    The rewards is calculated as:
    
    (min(action, self.number) + self.range) / (max(action, self.number) + self.range)
    
    Ideally an agent will be able to recognize the 'scent' of a higher reward and
    increase the rate in which is guesses in that direction until the reward reaches
    its maximum
    """
    def __init__(self):
        self.range = 1000  # +/- the value number can be between
        self.bounds = dataset[0].num_edges #2000  # Action space bounds

        self.action_space = spaces.Discrete(self.bounds) 
        # spaces.Box(low=np.array([-self.bounds]), high=np.array([self.bounds]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,))#spaces.Discrete(4)

        self.number = 0
        self.modify_cnt = 0 # 已經更改的數量
        self.modify_max = 10 # 最大可以更改的數量
        
        self.observation = 0

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        '''where to declare graph?_________________?'''
        
        if isinstance(action, (int, float)):
            action = np.array([action])
        elif isinstance(action, list):
            action = np.array(action)

        assert self.action_space.contains(action)
        
        # action 會是兩個 nodes 的 ids list?
        ## adjust adj: add or drop edges
        
        n1_neighbors = edges[1][torch.where(edges[0]==n1)]

        ## edges shape會變動
        if n2 in n1_neighbors:
            # edge exists
            print("drop edge")
            n2_idx = torch.where(edges[0]==n1)[0][torch.where(edges[1][torch.where(edges[0]==n1)[0]]==n2)]
            new_edges = torch.cat((edges[:,:n2_idx], edges[:,n2_idx+1:]), axis = 1) # drop edge
            assert new_edges.shape[1] != edges.shape[1] - 1, "size error"
        else:
            # edge not exist
            print("add edge")
            new_edges = torch.cat((edges, edge_add), axis=1) # add edge
            assert new_edges.shape[1] != edges.shape[1] + 1, "size error"
        
        self.observation = new_edges #????
        '''要變成每個node看的東西不一樣：每個node自己的一階、二階鄰居的embedding'''
        
        # update graph edges
        graph = dataset[0]
        new_graph = Data(edge_index=new_edges, 
                 test_mask=graph.test_mask, 
                 train_mask=graph.train_mask, 
                 val_mask=graph.val_mask, 
                 x=graph.x, 
                 y=graph.y)
        
        # train GCN model: input - graph
        reward = train(new_graph) # total reward: self.modify_cnt越少越好
        print("====================")
        print("Reward: ", reward)
        print("====================")

        self.guess_count += 1
        done = self.guess_count >= self.guess_max

        return self.observation, reward[0], done, {"number": self.number, "guesses": self.guess_count}

    def reset(self):
        self.number = self.np_random.uniform(-self.range, self.range)
        self.guess_count = 0
        self.observation = 0

        return self.observation