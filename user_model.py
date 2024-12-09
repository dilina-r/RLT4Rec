import os
import numpy as np
from numpy.random import choice
import math
import random


class UserModel():

    SCALE = 1.0

    def __init__(self, data_directory, dataset, nyms, min_rating=1.0, max_rating=5.0) -> None:
        self.data_directory = data_directory
        self.dataset = dataset
        self.nyms = nyms
        self.min_rating = min_rating
        self.max_rating = max_rating

        mu_filename = os.path.join(self.data_directory, "mu_{}{}.csv".format(self.dataset, self.nyms))
        sigma_filename = os.path.join(self.data_directory, "sigma_{}{}.csv".format(self.dataset, self.nyms))
        num_filename = os.path.join(self.data_directory, "num_{}{}.csv".format(self.dataset, self.nyms))

        self.mu = np.loadtxt(mu_filename, delimiter=",", dtype=np.float64)
        self.sigma = np.sqrt(np.loadtxt(sigma_filename, delimiter=",", dtype=np.float64))
        self.num = np.loadtxt(num_filename, delimiter=",", dtype=int)

        self.num_items = self.mu.shape[1]
        self.item_list = list(range(0, self.num_items))

        assert(self.mu.shape == self.sigma.shape)
        assert(self.mu.shape[0] == self.nyms)

    def get_num_actions(self):
        return self.num_items
    
    def get_num_groups(self):
        return self.nyms

    def __normpdf(self, x, mean, sd):
        var = float(sd)**2
        denom = (2*math.pi*var)**.5
        num = math.exp(-(float(x)-float(mean))**2/(2*var))
        return num/denom
    
    def get_random_group(self):
        return random.randint(0, self.nyms-1)
    
    def calc_prob(self, items, ratings):
        sums = [0.0] * self.nyms
        prods = [1.0] * self.nyms
        probs = [0.0] * self.nyms
        for i in range(0, len(items)):
            for g in range(0, self.nyms):
                sums[g] += ( (ratings[i] - self.mu[g][items[i]]) * (ratings[i] - self.mu[g][items[i]]) ) / ( self.sigma[g][items[i]] * self.sigma[g][items[i]] + 1e-10)
                prods[g] *= self.sigma[g][items[i]] + 1e-10
        
        sum_prob = 1e-10
        for g in range(0, self.nyms):
            probs[g] = np.exp(-sums[g]/2.0) / (prods[g])
            sum_prob += probs[g]
        
        for g in range(0, self.nyms):
            probs[g] = probs[g] / sum_prob
        
        return probs
    
    def __rating(self, usergroup, item, precision=0):
        r = min(max(self.min_rating, np.random.normal(self.mu[usergroup, item], self.sigma[usergroup, item])), self.max_rating)
        return round(r, precision)
    
    def get_rating(self, usergroup, item):
        if type(item) == list:
            ratings = []
            for i in item:
                ratings.append(self.__rating(usergroup, i))
            return ratings
        else:
            return self.__rating(usergroup, item)

    
    
    def get_sequence(self, usergroup, sequence_len=10, sampling='uniform', replace=False):

        if sampling=='popular':
            n = self.num[usergroup, :]
            n = n / np.sum(n)
            items = choice(self.item_list, sequence_len, p=n, replace=replace)
        else:
            #### Draw items uniformly
            items = choice(self.item_list, sequence_len, replace=replace)

        ratings = []
        for item in items:
            ratings.append(self.__rating(usergroup=usergroup, item=item))
        
        return items, ratings
    
    def get_dense_ratings(self, usergroup):
        ratings = self.get_rating(usergroup, item=self.item_list)
        assert(len(ratings) == self.num_items)
        return ratings



    





