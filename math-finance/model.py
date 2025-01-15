import numpy as np
from scipy.stats import norm
from optimisation import newton_raphson

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

        self.__dd1dT2 = (self.r+0.5*self.sigma*self.sigma)*self.T/(self.sigma*np.sqrt(self.T)) -\
            0.5*self.__d1/self.T
        self.__dd2dT2 = self.__dd1dT2 - 0.5*self.sigma/np.sqrt(self.T)
        
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

            self.epsilon = -self.S*self.T*norm.cdf(self.__d1)
            self.lambda_ = self.delta * self.S / self.price 

            self.vanna = -norm.pdf(self.__d1)*self.__d2/self.sigma
            self.charm = -norm.pdf(self.__d1)*(2*self.r*self.T - self.__d2*self.sigma*np.sqrt(self.T))/\
                (2*self.T*self.sigma*np.sqrt(self.T))
            self.vomma = self.vega*self.__d1*self.__d2/self.sigma
            self.vera = -self.K*self.T*np.exp(-self.r*self.T)*norm.pdf(self.__d2)*self.__d1/self.sigma
            self.veta = -self.S*norm.pdf(self.__d1)*np.sqrt(self.T)*( \
                ((self.r*self.__d1)/(self.sigma*np.sqrt(self.T))) - \
                    ((1+self.__d1*self.__d2)/(2*self.T)) )
            self.omega = (self.S*self.S*self.gamma)/(self.K*self.K)

            self.speed = - (self.gamma/self.S)*(self.__d1/(self.sigma*np.sqrt(self.T)) + 1)
            self.zomma = self.gamma*(self.__d1*self.__d2 - 1)/self.sigma
            self.color = -(norm.pdf(self.__d1)/(2*self.S*self.T*self.sigma*np.sqrt(self.T)))*\
                (1 + ((2*self.r*self.T - self.__d2*self.sigma*np.sqrt(self.T))/(self.sigma*np.sqrt(self.T))*self.__d1))
            self.ultima = -(self.vega/(self.sigma*self.sigma))*(self.__d1*self.__d2*(\
                1-self.__d1*self.__d2) + self.__d1*self.__d1 + self.__d2*self.__d2 )
            self.parmicharma = -(2*self.r*self.T-self.__d2*self.sigma*np.sqrt(self.T))/\
                (2*self.T*self.sigma*np.sqrt(self.T))*self.charm + norm.pdf(self.__d1)*\
                    (2*self.__d2*self.sigma*self.sigma*self.T - self.r*self.sigma*self.T*np.sqrt(self.T)-\
                     self.sigma*self.sigma*self.T*self.T*self.__dd2dT2)/\
                        (2*self.T*self.T*self.T*self.sigma*self.sigma)
            self.dual_delta = -np.exp(-self.r*self.T)*norm.cdf(self.__d2)
            self.dual_gamma = np.exp(-self.r*self.T)*norm.pdf(self.__d2)/(self.K*self.sigma*np.sqrt(self.T))


        elif self.type == "put":
            self.price =  self.K*np.exp(-self.r*self.T)*norm.cdf(-self.__d2) - self.S*norm.cdf(-self.__d1)
            self.delta = -norm.cdf(-self.__d1)
            self.gamma = norm.pdf(self.__d1)/(self.S*self.sigma*np.sqrt(self.T))
            self.vega = self.S*norm.pdf(self.__d1)*np.sqrt(self.T)
            self.theta = -(self.S*norm.pdf(self.__d1)*self.sigma)/(2*np.sqrt(self.T)) +\
                self.r*self.K*np.exp(-self.r*self.T)*norm.cdf(-self.__d2)
            self.rho = -self.K*self.T*np.exp(-self.r*self.T)*norm.cdf(-self.__d2)

            self.epsilon = self.S*self.T*norm.cdf(-self.__d1)
            self.lambda_ = self.delta * self.S / self.price 

            self.vanna = -norm.pdf(self.__d1)*self.__d2/self.sigma
            self.charm = -norm.pdf(self.__d1)*(2*self.r*self.T - self.__d2*self.sigma*np.sqrt(self.T))/\
                (2*self.T*self.sigma*np.sqrt(self.T))
            self.vomma = self.vega*self.__d1*self.__d2/self.sigma
            self.vera = -self.K*self.T*np.exp(-self.r*self.T)*norm.pdf(self.__d2)*self.__d1/self.sigma
            self.veta = -self.S*norm.pdf(self.__d1)*np.sqrt(self.T)*( \
                ((self.r*self.__d1)/(self.sigma*np.sqrt(self.T))) - \
                    ((1+self.__d1*self.__d2)/(2*self.T)) )
            self.omega = (self.S*self.S*self.gamma)/(self.K*self.K)

            self.speed = - (self.gamma/self.S)*(self.__d1/(self.sigma*np.sqrt(self.T)) + 1)
            self.zomma = self.gamma*(self.__d1*self.__d2 - 1)/self.sigma
            self.color = -(norm.pdf(self.__d1)/(2*self.S*self.T*self.sigma*np.sqrt(self.T)))*\
                (1 + ((2*self.r*self.T - self.__d2*self.sigma*np.sqrt(self.T))/(self.sigma*np.sqrt(self.T))*self.__d1))
            self.ultima = -(self.vega/(self.sigma*self.sigma))*(self.__d1*self.__d2*(\
                1-self.__d1*self.__d2) + self.__d1*self.__d1 + self.__d2*self.__d2 )
            self.parmicharma = -(2*self.r*self.T-self.__d2*self.sigma*np.sqrt(self.T))/\
                (2*self.T*self.sigma*np.sqrt(self.T))*self.charm + norm.pdf(self.__d1)*\
                    (2*self.__d2*self.sigma*self.sigma*self.T - self.r*self.sigma*self.T*np.sqrt(self.T)-\
                     self.sigma*self.sigma*self.T*self.T*self.__dd2dT2)/\
                        (2*self.T*self.T*self.T*self.sigma*self.sigma)
            self.dual_delta = np.exp(-self.r*self.T)*norm.cdf(-self.__d2)
            self.dual_gamma = np.exp(-self.r*self.T)*norm.pdf(self.__d2)/(self.K*self.sigma*np.sqrt(self.T))


            
            
def implied_volatility(S0: float, r: float, K: float, T: float, market_price: float, 
                       tol: float = 1e-5, type: str = "call", return_list: bool = False) -> float | list[float]:
    """
    This is basically a root-finding problem, we model the underlying and 
    come up with a price for some derivative as a function of volatility
    sigma. We want to find the volatility which gives the market price i.e.
    C_model(sigma) = C_market i.e. f(sigma) = C_model(sigma) - C_market. We
    can use Newton-Raphson method. Note: We need C(S0, r, sigma, K, T) and 
    the delta. So we fix S0, r, K and T. 

    Arguments:
    ----------
    S0 : float
        Initial price of underlying.
    r : float
        Risk-free interest rate.
    K : float
        Strike price.
    T : float
        Maturity date.
    market_price: float
        Market price of option quoted from market.
    tol : float
        Tolerance level.
    type : str
        Type of option "call" or "put".


    Returns: 
    --------
    float or List[float]
        The implied volatility.
    """

    # Some bug somewhere: maybe implementation of Black-Scholes class

    max_iter = 200
    initial_sigma = 0.30
    
    black_scholes_price = lambda vol : BlackScholes(S0, K, T, r, vol, type=type).price - market_price
    black_scholes_delta = lambda vol : BlackScholes(S0, K, T, r, vol, type=type).delta
    
    return newton_raphson(black_scholes_price, black_scholes_delta, initial_sigma,
                          tol=tol, max_iter=max_iter, return_list=return_list)


            