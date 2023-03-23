import numpy as np
import pickle
from tqdm import tqdm

class PolicyIteration:

    def __init__(self):

        self.gamma = 0.6

    def compute(self, N):

        self.N = N
        self.S = 3**(N**2)

        self.wins = [-1]*self.S
        self.loses = [-1]*self.S
        self.ties = [-1]*self.S
        self.valids = [-1]*self.S

        # start with a random policy
        print('Getting a random policy')
        policy = [-1] * self.S

        for state in tqdm(range(self.S)):

            b_state = np.base_repr(state, 3)
            b_state = b_state.zfill(self.N*self.N)

            if not self.is_valid(b_state):
                continue

            policy[state] = self.get_actions(b_state)[0]
        

        iter_i = 1

        while True:

            print('Iteration', iter_i)
            iter_i += 1

            V = self.policy_eval(policy)
            policy_new = self.policy_improvement(V)

            if policy == policy_new:
                break
            else:
                policy = policy_new
            
        return policy

    def policy_eval(self, policy):

        print('Policy Evaluation')

        V = [0] * self.S

        for _ in tqdm(range(10)):

            V_new = self.bellmann_operator(policy, V)
            V = V_new

        return V

    def policy_improvement(self, V):

        print('Policy Improvement')

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

    def bellmann_operator(self, policy, V):

        V_new = [0] * self.S

        for state in range(self.S):

            b_state = np.base_repr(state, 3)
            b_state = b_state.zfill(self.N*self.N)

            if not self.is_valid(b_state):
                V_new[state] = 0
                continue

            action = policy[state]

            val = 0
            for p, next_state in self.get_next_states(b_state, action):
                r = self.reward(b_state, action, next_state)
                val += p * (r + self.gamma * V[int(next_state,3)])
                
            V_new[state] = val

        return V_new

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



if __name__ == '__main__':

    policy = PolicyIteration().compute(4)

    with open('policy_4x4_p.pkl', 'wb') as f:
        pickle.dump(policy, f)

    # for _ in range(10):

    #     state = input()
    #     action = policy[int(state,3)]
    #     print(action)