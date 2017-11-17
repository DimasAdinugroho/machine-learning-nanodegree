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
        # sample use n-1, this is bessel's correction
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
        '''
        input x: x values
        return: z-score
        '''
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
        input x: x values, not the z-score
        '''
        if less == True:
            return round(self.dist.cdf(x), 4)
        return 1 - round(self.dist.cdf(x), 4) - 0.0003  # correction

    def plot(self):
        import matplotlib.pyplot as plt
        min_max = (self._mean - self._std * 3, self._mean + self._std * 3)
        rng = range(min_max[0], min_max[1])
        self.dist.pdf(r)

        fig = plt.figure(figsize=(10, 8))
        plt.subplot(311)  # Creates a 3 row, 1 column grid of plots, and renders the following chart in slot 1.
        plt.plot(rng, self.dist.pdf(rng), 'r', linewidth=2)
        plt.title('Probability density function of normal distribution')

        # Plot probability density function and of this distribution.
        plt.subplot(312)
        plt.plot(rng, self.dist.cdf(rng))
        plt.title('Cumulutative distribution function of normal distribution')

        # Draw 1000 samples from the random variable.
        sample = self.dist.rvs(size=10000)

        print "Sample descriptive statistics:"
        print pd.DataFrame(sample).describe()

        # Plot a histogram of the samples.
        plt.subplot(313)
        plt.hist(sample, bins=100, normed=True)
        plt.plot(rng, self.dist.pdf(rng), 'r--', linewidth=2)
        plt.title('10,000 random samples from normal distribution')


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
