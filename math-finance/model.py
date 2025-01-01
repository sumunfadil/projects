import numpy as np
from scipy.stats import norm

"""
Main parts (interlinked):
- model e.g. Black-Scholes
- option e.g. vanilla
- pricing method e.g. Monte Carlo
"""

class BlackScholes:
    def __init__(self, S, K, T, r, sigma, type="call"):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.type = type

        self.__d1 = (np.log(self.S/self.K) + (r + 0.5*self.sigma*self.sigma)*self.T)/\
            (self.sigma*np.sqrt(self.T))
        self.__d2 = self.__d1 - self.sigma*np.sqrt(self.T)
        
        if self.type == "call":
            self.price = self.S*norm.cdf(self.__d1) - self.K*np.exp(-self.r*self.T)*norm.cdf(self.__d2)
            self.delta = norm.cdf(self.__d1)
            self.gamma = norm.pdf(self.__d1)/(self.S*self.sigma*np.sqrt(self.T))
            #self.gamma = self.K*np.exp(-self.r*self.T)*norm.pdf(self.__d2)/(self.S*self.S*self.sigma*np.sqrt(self.T))
            self.vega = self.S*norm.pdf(self.__d1)*np.sqrt(self.T)
            #self.vega = self.K*np.exp(-self.r*self.T)*norm.pdf(self.__d2)*np.sqrt(self.T)
            self.theta = -(self.S*norm.pdf(self.__d1)*self.sigma)/(2*np.sqrt(self.T)) -\
                self.r*self.K*np.exp(-self.r*self.T)*norm.cdf(self.__d2)
            self.rho = self.K*self.T*np.exp(-self.r*self.T)*norm.cdf(self.__d2)

        elif self.type == "put":
            self.price =  self.K*np.exp(-self.r*self.T)*norm.cdf(-self.__d2) - self.S*norm.cdf(-self.__d1)
            self.delta = -norm.cdf(-self.__d1)
            self.gamma = norm.pdf(self.__d1)/(self.S*self.sigma*np.sqrt(self.T))
            self.vega = self.S*norm.pdf(self.__d1)*np.sqrt(self.T)
            self.theta = -(self.S*norm.pdf(self.__d1)*self.sigma)/(2*np.sqrt(self.T)) +\
                self.r*self.K*np.exp(-self.r*self.T)*norm.cdf(-self.__d2)
            self.rho = -self.K*self.T*np.exp(-self.r*self.T)*norm.cdf(-self.__d2)