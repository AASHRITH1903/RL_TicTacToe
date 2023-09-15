import numpy as np
from tqdm import tqdm
import copy
import pickle


class ValueIteration:

    def __init__(self):

        self.gamma = 0.6

    def compute(self, N, n_iter):

        self.N = N
        self.n_iter = n_iter
        self.S = 3**(N**2)

        self.wins = [-1]*self.S
        self.loses = [-1]*self.S
        self.ties = [-1]*self.S
        self.valids = [-1]*self.S
        
        V = self.get_values()
        policy = self.get_policy(V)

        return policy
        

    def get_policy(self, V):

        print('Getting policy from the value function ....')

        policy = [-1]*self.S

        for state in tqdm(range(self.S)):

            b_state = np.base_repr(state, 3)
            b_state = b_state.zfill(self.N*self.N)

            if not self.is_valid(b_state):
                continue

            max_val = np.NINF

            for action in self.get_actions(b_state):
                val = 0
                for p, next_state in self.get_next_states(b_state, action):
                    r = self.reward(b_state, action, next_state)
                    val += p * (r + self.gamma * V[int(next_state,3)])
                
                if val > max_val:
                    max_val = val
                    policy[state] = action
        
        return policy

    def get_values(self):

        V = [0] * self.S

        print('Computing optimal values for states ....')

        for iter in range(self.n_iter):

            print('Iteration', iter+1)

            V_new = self.bellman_optimality_operator(V)
            V = V_new
            # V_new = [0] * self.S

            # for state in tqdm(range(self.S)):
            #     # self.V[state] = self.get_new_state_value(state)
            #     V_new[state] = self.get_new_state_value(state, V)
            
            # V = copy.deepcopy(V_new)
            # del V_new

        return V

    def bellman_optimality_operator(self, V:list):

        V_new = [0] * self.S

        for state in tqdm(range(self.S)):
            # self.V[state] = self.get_new_state_value(state)

            b_state = np.base_repr(state, 3)
            b_state = b_state.zfill(self.N*self.N)

            # V(win) = V(lose) = V(tie) = 0
            if not self.is_valid(b_state):
                V_new[state] = 0
                continue

            max_val = np.NINF

            for action in self.get_actions(b_state):
                val = 0
                for p, next_state in self.get_next_states(b_state, action):
                    r = self.reward(b_state, action, next_state)
                    val += p * (r + self.gamma * V[int(next_state,3)])
                
                max_val = max(max_val, val)

            V_new[state] = max_val
            # V_new[state] = self.get_new_state_value(state, V)

        return V_new


    # def get_new_state_value(self, state, V):

    #     b_state = np.base_repr(state, 3)
    #     b_state = b_state.zfill(self.N*self.N)

    #     # V(win) = V(lose) = V(tie) = 0
    #     if self.is_win(b_state) or self.is_lose(b_state) or self.is_tie(b_state):
    #         return 0
    #     if not self.is_valid(b_state):
    #         return 0

    #     max_val = np.NINF

    #     for action in self.get_actions(b_state):
    #         val = 0
    #         for p, next_state in self.get_next_states(b_state, action):
    #             r = self.reward(b_state, action, next_state)
    #             val += p * (r + self.gamma * V[int(next_state,3)])
            
    #         max_val = max(max_val, val)

    #     return max_val

    def get_actions(self, b_state):
        
        actions = []
        for i in range(len(b_state)):
            if b_state[i] == '0':
                actions.append(i)
        return actions

    def get_next_states(self, b_state, action):

        next_states = []

        interim_state = b_state[:action] + '1' + b_state[action+1:]
        empty_cells = [i for i in range(self.N*self.N) if interim_state[i]=='0']

        if len(empty_cells) == 0:
            return next_states

        p = 1/len(empty_cells) # uniform probability

        for e in empty_cells:
            next_state = interim_state[:e] + '2' + interim_state[e+1:]
            next_states.append((p, next_state))

        return next_states

    def reward(self, b_state, action, next_state):
  
        if self.is_win(next_state):
            return 100
        elif self.is_lose(next_state):
            return -1000000
        elif self.is_tie(next_state):
            return 0
        else:
            return 0


    def is_win(self, b_state):

        state = int(b_state, 3)

        if self.wins[state] != -1:
            return self.wins[state]

        # check rows
        for i in range(self.N):
            if sum([b_state[i*self.N+j]=='1' for j in range(self.N)]) == self.N:
                self.wins[state] = True
                return True

        # check cols
        for j in range(self.N):
            if sum([b_state[i*self.N+j]=='1' for i in range(self.N)]) == self.N:
                self.wins[state] = True
                return True

        # check diags
        if sum([b_state[i*self.N+i]=='1' for i in range(self.N)]) == self.N:
            self.wins[state] = True
            return True

        if sum([b_state[i*self.N+(self.N-1-i)]=='1' for i in range(self.N)]) == self.N:
            self.wins[state] = True
            return True

        self.wins[state] = False
        return False


    def is_lose(self, b_state):
        
        state = int(b_state, 3)

        if self.loses[state] != -1:
            return self.loses[state]

        # check rows
        for i in range(self.N):
            if sum([b_state[i*self.N+j]=='2' for j in range(self.N)]) == self.N:
                self.loses[state] = True
                return True

        # check cols
        for j in range(self.N):
            if sum([b_state[i*self.N+j]=='2' for i in range(self.N)]) == self.N:
                self.loses[state] = True
                return True

        # check diags
        if sum([b_state[i*self.N+i]=='2' for i in range(self.N)]) == self.N:
            self.loses[state] = True
            return True

        if sum([b_state[i*self.N+(self.N-1-i)]=='2' for i in range(self.N)]) == self.N:
            self.loses[state] = True
            return True

        self.loses[state] = False
        return False

    def is_tie(self, b_state):

        state = int(b_state, 3)

        if self.ties[state]!=-1:
            return self.ties[state]

        # there should be no empty cells
        num_empty_cells = sum([b_state[i]=='0' for i in range(self.N*self.N)])

        if not self.is_win(b_state) and not self.is_lose(b_state) and num_empty_cells == 0:
            self.ties[state] = True
        else:
            self.ties[state] = False

        return self.ties[state]

    def is_valid(self, b_state):

        state = int(b_state, 3)

        if self.valids[state]!=-1:
            return self.valids[state]
        
        if self.is_win(b_state) or self.is_lose(b_state) or self.is_tie(b_state):
            self.valids[state] = False
            return False

        num_x = sum([b_state[i]=='1' for i in range(self.N*self.N)])
        num_o = sum([b_state[i]=='2' for i in range(self.N*self.N)])

        self.valids[state] = (num_x == num_o)

        return (num_x == num_o)



if __name__=='__main__':

    policy =  ValueIteration().compute(4, 10)
    
    with open('policy_vi_4x4.pkl', 'wb') as f:
        pickle.dump(policy, f)

    # with open('policy_4x4.pkl', 'rb') as f:
    #     policy = pickle.load(f)
    
    # for _ in range(10):

    #     state = input()
    #     action = policy[int(state,3)]
    #     print(action)