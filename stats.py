class BasicStat(object):
    @staticmethod
    def mean(X):
        return sum(X) / float(len(X))

    @staticmethod
    def variance(X, sample=False):
        means = BasicStat.mean(X)
        if sample == True:
            return sum([(i-means)**2 for i in X]) / (len(X)-1)
        return sum([(i-means)**2 for i in X]) / len(X)

    @staticmethod
    def standard_dev(X, sample=False):
        import math
        return math.sqrt(BasicStat.variance(X, sample=sample))

    @staticmethod
    def covariance(X, Y):
        mean_x = BasicStat.mean(X)
        mean_y = BasicStat.mean(Y)
        result = [(X[i] - mean_x)*(Y[i] - mean_y) for i,_ in enumerate(X)]
        return(sum(result)/(len(X)))

    @staticmethod
    def pearson_corr(X, Y):
        return BasicStat.covariance(X, Y) / (BasicStat.standard_dev(X) * BasicStat.standard_dev(Y))


class NormalDistribution(object):

    def __init__(self, mean, std):
        from scipy.stats import norm
        self._mean = round(mean, 2)
        self._std = round(std, 2)
        self.dist = norm(mean, std)

    def get_z(self, x):
        return round(float(x - self._mean) / self._std, 2)

    def pdf(self, x):
        return round(self.dist.pdf(x), 4)

    def ppf(self, percent):
        ''' convert percent to z-score 
        example: 95% OR 0.95 should return 1.96
        '''
        return self.dist.ppf(percent)

    def cdf(self, x, less=True):
        ''' Get Probability using z-table
        (cumulative distribution function), area under the curve
        '''
        if less == True:
            return round(self.dist.cdf(x), 4)
        return 1 - round(self.dist.cdf(x), 4)


class LinearRegression(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.gradient = None
        self.intercept = None

    def train(self):
        '''
        :param X: input data
        :param Y: output data
        :return w_0, w_1: (intercept, gradient)
        '''
        import numpy as np

        n = len(X)  # number of data
        X = np.array(X)
        Y = np.array(Y)
        sum_x = X.sum()
        sum_y = Y.sum()
        sum_xy = (X * Y).sum()
        sum_xx = (X * X).sum()

        w_1 = (sum_xy - ((sum_x * sum_y)) / n) / (sum_xx - (sum_x * sum_x) / n)
        w_0 = (sum_y / n) - (w_1 * sum_x / n)
        self.gradient = w_1
        self.intercept = w_0
        return w_0, w_1

    def predict(self, X):
        if self.gradient == None or self.intercept == None:
            return 0
        X = np.array(X)
        return self.gradient * X + self.intercept

