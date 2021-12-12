import numpy as np
from collections import defaultdict
from itertools import product
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns


class mastermind:
    def __init__(self, obj='random', guess_init='random', colors=['r', 'g', 'b', 'y', 'w', 'k'], obj_length=4):
        self.n_colors = len(colors)
        self.colors = colors
        self.obj_length = obj_length
        self.obj = np.random.choice(self.colors,self.obj_length).tolist() if obj == 'random' else obj
        self.guess_init = guess_init
        self.solved = False
        
        if self.guess_init == 'random':
            self.first_guess = np.random.choice(self.colors,self.obj_length).tolist()
        elif 'color' in self.guess_init:
            self.first_guess = (np.random.choice(self.colors,int(guess_init[0]), replace=False)
                                .tolist() * 20)[:self.obj_length]
        
        self.guesses = [(self.first_guess, self.scorer(self.first_guess))]
        
        self.solutions = list(map(list, product(self.colors, repeat=self.obj_length)))
        
    def check_if_solved(self):
        return True if (sum(self.guesses[-1][1]) == self.obj_length) else False
    
    def scorer(self, guess):
        blacks = [1 for o, g in zip(self.obj,guess) if o == g]
        whites = [min([self.obj.count(x), guess.count(x)]) for x in set(guess)]
        score = blacks + [0]*(sum(whites)-sum(blacks))
        return score
    
    def eval_colors(self):
        guess, score = self.guesses[-1]
        rc = len(score) 
        tmp_sols = []
        for sol in self.solutions:
            tmp_guess = guess.copy()
            common = [e for e in sol if e in tmp_guess and tmp_guess.pop(tmp_guess.index(e))]
            if len(common) == rc:
                tmp_sols.append(sol)
        self.solutions = tmp_sols
                
    def eval_positions(self):
        guess, score = self.guesses[-1]
        blacks = sum(score)
        tmp_sols = []
        for sol in self.solutions:
            count = sum([1 for g, s in zip(guess, sol) if g == s])
            if count == blacks:
                tmp_sols.append(sol)
        self.solutions = tmp_sols
                
    def evaluate_guess(self):
        self.eval_colors()
        self.eval_positions()
        self.solved = self.check_if_solved()
        if not self.solved:
            new_guess = self.solutions[np.random.choice(range(len(self.solutions)), 1)[0]]
            self.guesses.append((new_guess, self.scorer(new_guess)))
        
    def start_game(self, verbose=False):
        while not self.solved:
            self.evaluate_guess()
        if verbose:
            print('Solved in %s turns' % len(self.guesses))


def plot_simulation(data,title=''):
    plt.hist(data,bins=np.arange(1, 11) - .5)
    plt.axvline(np.mean(data), linestyle=':', c='r', label='Mean: %.02f\nStd: %.02f' % (np.mean(data), np.std(data)))
    plt.xlim(1, 10); plt.xlabel('Turns to win'); plt.title('%s\n%s sample games' % (title, len(data))); 
    plt.legend()


def game_runner(guess_init, n_games):
    turns = []
    for i in tqdm(range(n_games)):
        game = mastermind(guess_init=guess_init)
        game.start_game()
        turns.append(len(game.guesses))
    return turns