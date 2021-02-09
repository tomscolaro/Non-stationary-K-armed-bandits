from io import RawIOBase
import random
import sys
import numpy as np
import timeit

class k_bandit():

    def __init__(self) -> None:
        self.arms = 10
        self.epsilon = .1
        self.alpha = .1
        self.steps = 10000
        self.iterations = 300
        self.reward_var = 1.0
        self.walk_var = .01
       


    def setup_Q(self):
        #setup Q
        x = np.zeros(self.arms)
        env = [x]
        for i in range(1,self.steps):
            x = x + np.random.default_rng().normal(0, np.sqrt(self.walk_var), self.arms)
            env.append(x)
        env = np.array(env)
        return env


    def choose(self, q): 
        if self.epsilon >= np.random.uniform(0,1):
            choice = np.random.choice([i for i in range(self.arms)], 1)[0]  
            return choice
        else:    
            choice = np.argmax(q)
            return choice
    
    def learn(self):
        #intialize placeholders for sample average measurement 
        sa_optimal_actions_taken = np.zeros(self.steps)
        sa_rewards_from_actions = np.zeros(self.steps)
        sa_selection_count = np.zeros(self.arms)
        sa_q = np.zeros(self.arms)
        
        #intialize placeholders for constant rate measurement 
        cr_optimal_actions_taken = np.zeros(self.steps)
        cr_rewards_from_actions = np.zeros(self.steps)
        #cr_selection_count = np.ones(self.arms)
        cr_q = np.zeros(self.arms)

        #create environment
        env = self.setup_Q()

        for n in range(self.steps):
            #generate rewards
            rewards = np.random.default_rng().normal(env[n] , self.reward_var)
            optimal_choice = np.argmax(rewards)

            #using sample averaging
            sa_choice = self.choose(sa_q)
            sa_selection_count[sa_choice] += 1
            sa_q[sa_choice] = sa_q[sa_choice] + ((rewards[sa_choice] - sa_q[sa_choice])/sa_selection_count[sa_choice] )

            #metric keeping
            sa_rewards_from_actions[n] = rewards[sa_choice]
            if optimal_choice == sa_choice:
                sa_optimal_actions_taken[n] = 1

            #using constant rate learing    
            cr_choice = self.choose(cr_q)
            cr_q[cr_choice] = cr_q[cr_choice] + (self.alpha * (rewards[cr_choice] - cr_q[cr_choice]))
                 
            #metric keeping
            cr_rewards_from_actions[n] = rewards[cr_choice]
            if optimal_choice == cr_choice:
                cr_optimal_actions_taken[n] = 1
            
        return sa_rewards_from_actions,  sa_optimal_actions_taken, cr_rewards_from_actions,  cr_optimal_actions_taken
    

    def run(self):
        sample_actions_output = np.zeros(self.steps)
        sample_rewards_output = np.zeros(self.steps)
        constant_actions_output = np.zeros(self.steps)
        constant_rewards_output = np.zeros(self.steps)

        for i in range(self.iterations):
            # print('Iteration {}'.format(i))
            sample_rewards, sample_acts, cr_rewards, cr_acts =  self.learn()
            
            sample_actions_output = sample_actions_output + sample_acts
            sample_rewards_output = sample_rewards_output + sample_rewards
            constant_actions_output = constant_actions_output + cr_acts
            constant_rewards_output = constant_rewards_output + cr_rewards

        sample_actions_output = sample_actions_output/self.iterations
        sample_rewards_output = sample_rewards_output/self.iterations
        constant_actions_output = constant_actions_output/self.iterations
        constant_rewards_output = constant_rewards_output/self.iterations

        #print(constant_actions_output)
        
        return [sample_rewards_output, sample_actions_output, constant_rewards_output, constant_actions_output]

    def output(self, output):
        results = self.run()
        with open(output, "ab") as f:
            for i in results:
                np.savetxt(f, i.reshape(1, self.steps), newline="\n") 

if __name__ == "__main__":
    output_file = sys.argv[1]
    start = timeit.default_timer()
    c = k_bandit()
    c.output(output_file)
    stop = timeit.default_timer()
    total_time = stop - start
    print(total_time)
