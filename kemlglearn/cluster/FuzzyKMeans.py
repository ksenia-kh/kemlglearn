"""
.. module:: HarmonicKMeans

FuzzyKMeans
*************

:Description: FuzzyKMeans


"""

# Author: Ksenia Kharitonova <ksenia.kharitonova@gmail.com>
# License: BSD 3 clause

import numpy as np

from sklearn.base import TransformerMixin, ClusterMixin, BaseEstimator
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_random_state

class FuzzyKMeans(TransformerMixin, ClusterMixin, BaseEstimator):
    """
    Fuzzy K-means

    Reference
    ---------
    Hamerly, G. & Elkan, C.  
    Alternatives to the k-means algorithm that find better clusterings. 
    Proceedings of the Eleventh International Conference on 
    Information and Knowledge Management (CIKM-02), 
    ACM Press, 2002, 600-607
    """

    # def __init__(self, n_clusters=3, r=2., tol=1e-3, 
    #             init_method='forgy',max_iter=300, random_state=None):
    #     ''' init_method by Pena, Lozano and Larranaga. An empirical comparison of
    #         four initialization methods for the k-means algorithm.
    #         Pattern recognition letters,20:1027-1040, 1999.

    #         The Forgy method choose n_clusters data points from the dataset at random 
    #         and uses them as initial centers.
    #         The Random Partition method assigns each datapoint to a random center, then 
    #         computes the initial location of each center as the centroid of its assigned 
    #         points.

    #         r is a fuzzy degree, the closer r to 1, the closer the method to classical 
    #         K Means, the larger r the fuzzier is the method.
             
            
    #     '''
    #     self.n_clusters = n_clusters
    #     self.r = r
    #     self.tol = tol
    #     self.init_method = init_method
    #     self.max_iter = max_iter
    #     self.random_state = random_state

    # def _calculate_centers(self, X):
    #     # distances between cluster centers and X with the degree p+2
    #     M1 = ((X[:,np.newaxis]-self.cluster_centers_)**(-2/(self.r-1))).sum(axis=2)
        
    #     # calculate membership m 
    #     self.m_ = M1/np.sum(M1, axis=1).reshape(-1,1)

    #     # calculate centers
    #     self.cluster_centers_ = np.dot((self.m_).T,X)/(self.m_).sum(axis=0).reshape(-1,1)

    # def fit_predict(self, X):
    #     '''Returns hard partition (labels) for the data X on which the instance was fitted.
    #     '''
        
    #     n_samples, n_features = X.shape
    #     vdata = np.mean(np.var(X, 0))

    #     random_state = check_random_state(self.random_state)
         
    #     if self.init_method=='forgy':
    #         self.cluster_centers_ = X[random_state.choice(np.arange(n_samples), self.n_clusters)]
    #     elif self.init_method=='random_partition':
    #         ra = random_state.randint(self.n_clusters, size=n_samples)
    #         self.cluster_centers_ = np.zeros((self.n_clusters,n_features))
    #         for i in range(self.n_clusters):
    #             self.cluster_centers_[i] = X[ra==i].mean(axis=0)

    #     for i in range(self.max_iter):
    #         centers_old = self.cluster_centers_.copy()

    #         self._calculate_centers(X)

    #         if np.sum((centers_old - self.cluster_centers_) ** 2) < self.tol * vdata:
    #             break

    #     return self.m_.argmax(axis=1)

    ''' The realization is taken from sklearn-extensions (c) Mathieu Blondel
    http://wdm0006.github.io/sklearn-extensions/fuzzy_k_means.html#third-party-docs
    '''

    def __init__(self, k, m=2, max_iter=100, random_state=0, tol=1e-4):
        """
        m > 1: fuzzy-ness parameter
        The closer to m is to 1, the closter to hard kmeans.
        The bigger m, the fuzzier (converge to the global cluster).
        """
        self.k = k
        assert m > 1
        self.m = m
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol

    def _e_step(self, X):
        D = 1.0 / euclidean_distances(X, self.cluster_centers_, squared=True)
        D **= 1.0 / (self.m - 1)
        D /= np.sum(D, axis=1)[:, np.newaxis]
        # shape: n_samples x k
        self.fuzzy_labels_ = D
        self.labels_ = self.fuzzy_labels_.argmax(axis=1)

    def _m_step(self, X):
        weights = self.fuzzy_labels_ ** self.m
        # shape: n_clusters x n_features
        self.cluster_centers_ = np.dot(X.T, weights).T
        self.cluster_centers_ /= weights.sum(axis=0)[:, np.newaxis]

    def fit_predict(self, X, y=None):
        n_samples, n_features = X.shape
        vdata = np.mean(np.var(X, 0))

        random_state = check_random_state(self.random_state)
        self.fuzzy_labels_ = random_state.rand(n_samples, self.k)
        self.fuzzy_labels_ /= self.fuzzy_labels_.sum(axis=1)[:, np.newaxis]
        self._m_step(X)

        for i in range(self.max_iter):
            centers_old = self.cluster_centers_.copy()

            self._e_step(X)
            self._m_step(X)

            if np.sum((centers_old - self.cluster_centers_) ** 2) < self.tol * vdata:
                break

        return self.labels_



class Hybrid1(TransformerMixin, ClusterMixin, BaseEstimator):
    """
    Hybrid1
    Hard membership like in KMeans
    Varying weights like in Harmonic KMeans

    Reference
    ---------
    Hamerly, G. & Elkan, C.  
    Alternatives to the k-means algorithm that find better clusterings. 
    Proceedings of the Eleventh International Conference on 
    Information and Knowledge Management (CIKM-02), 
    ACM Press, 2002, 600-607
    """
    def __init__(self, n_clusters=3, p=2., epsilon=1e-6, tol=1e-3, 
                init_method='forgy', max_iter=300, random_state=None):
        ''' init_method by Pena, Lozano and Larranaga. An empirical comparison of
            four initialization methods for the k-means algorithm.
            Pattern recognition letters,20:1027-1040, 1999.

            The Forgy method choose n_clusters data points from the dataset at random 
            and uses them as initial centers.
            The Random Partition method assigns each datapoint to a random center, then 
            computes the initial location of each center as the centroid of its assigned 
            points.

            p is an input degree and typically p >= 2
             
            Epsilon is small positive value that prevents numerical overflow when the center
            and a data point coincide.


        '''
        self.n_clusters = n_clusters
        self.p = p
        self.epsilon = epsilon
        self.tol = tol
        self.init_method = init_method
        self.max_iter = max_iter
        self.random_state = random_state

    def _calculate_centers(self, X):
        # distances between cluster centers and X with the degree 2
        dist_2 = ((X[:,np.newaxis]-self.cluster_centers_)**2).sum(axis=2)
        # distances between cluster centers and X with the degree p+2
        dist_p_plus_2 = ((X[:,np.newaxis]-self.cluster_centers_)**(self.p+2)).sum(axis=2)
        # distances between cluster centers and X with the degree p
        dist_p = ((X[:,np.newaxis]-self.cluster_centers_)**self.p).sum(axis=2)

        # check that we don't divide by 0 when inverting distances - Zhang
        dist_p_plus_2[dist_p_plus_2<=self.epsilon]=self.epsilon
        dist_p[dist_p<=self.epsilon]=self.epsilon

        # calculate membership m 
        self.m_ = np.where(dist_2 == dist_2.min(axis=1).reshape(-1, 1), 1, 0)

        # calculate weight w
        M1 = dist_p_plus_2**(-1)
        M2 = dist_p**(-1)
        self.w_ = M1.sum(axis=1).reshape(-1,1)/(M2.sum(axis=1).reshape(-1,1))**2

        # calculate centers
        self.cluster_centers_ = np.dot((self.m_*self.w_).T,X)/(self.m_*self.w_).sum(axis=0).reshape(-1,1)

    def fit_predict(self, X):
        '''Returns hard partition (labels) for the data X on which the instance was fitted.
        '''
        n_samples, n_features = X.shape
        vdata = np.mean(np.var(X, 0))

        random_state = check_random_state(self.random_state)
        
        # centers initialization 
        if self.init_method=='forgy':
            self.cluster_centers_ = X[random_state.choice(np.arange(n_samples), self.n_clusters)]
        elif self.init_method=='random_partition':
            ra = random_state.randint(self.n_clusters, size=n_samples)
            self.cluster_centers_ = np.zeros((self.n_clusters,n_features))
            for i in range(self.n_clusters):
                self.cluster_centers_[i] = X[ra==i].mean(axis=0)

        for i in range(self.max_iter):
            centers_old = self.cluster_centers_.copy()

            self._calculate_centers(X)

            if np.sum((centers_old - self.cluster_centers_) ** 2) < self.tol * vdata:
                break

        return self.m_.argmax(axis=1)

class Hybrid2(TransformerMixin, ClusterMixin, BaseEstimator):
    """
    Hybrid1
    Soft membership like in Harmonic KMeans
    Constant weights like in KMeans and Fuzzy KMeans

    Reference
    ---------
    Hamerly, G. & Elkan, C.  
    Alternatives to the k-means algorithm that find better clusterings. 
    Proceedings of the Eleventh International Conference on 
    Information and Knowledge Management (CIKM-02), 
    ACM Press, 2002, 600-607
    """
    def __init__(self, n_clusters=3, p=2., epsilon=1e-6, tol=1e-3, 
                init_method='forgy', max_iter=300, random_state=None):
        ''' init_method by Pena, Lozano and Larranaga. An empirical comparison of
            four initialization methods for the k-means algorithm.
            Pattern recognition letters,20:1027-1040, 1999.

            The Forgy method choose n_clusters data points from the dataset at random 
            and uses them as initial centers.
            The Random Partition method assigns each datapoint to a random center, then 
            computes the initial location of each center as the centroid of its assigned 
            points.

            p is an input degree and typically p >= 2
             
            Epsilon is small positive value that prevents numerical overflow when the center
            and a data point coincide.


        '''
        self.n_clusters = n_clusters
        self.p = p
        self.epsilon = epsilon
        self.tol = tol
        self.init_method = init_method
        self.max_iter = max_iter
        self.random_state = random_state

    def _calculate_centers(self, X):
        # distances between cluster centers and X with the degree p+2
        dist_p_plus_2 = ((X[:,np.newaxis]-self.cluster_centers_)**(self.p+2)).sum(axis=2)
        
        # check that we don't divide by 0 when inverting distances - Zhang
        dist_p_plus_2[dist_p_plus_2<=self.epsilon]=self.epsilon
        
        # calculate membership m 
        M1 = dist_p_plus_2**(-1)
        self.m_ = M1/np.sum(M1, axis=1).reshape(-1,1)

        # calculate centers (weight is constant)
        self.cluster_centers_ = np.dot(self.m_.T,X)/self.m_.sum(axis=0).reshape(-1,1)

    def fit_predict(self, X):
        '''Returns hard partition (labels) for the data X on which the instance was fitted.
        '''
        n_samples, n_features = X.shape
        vdata = np.mean(np.var(X, 0))

        random_state = check_random_state(self.random_state)
        
        # centers initialization 
        if self.init_method=='forgy':
            self.cluster_centers_ = X[random_state.choice(np.arange(n_samples), self.n_clusters)]
        elif self.init_method=='random_partition':
            ra = random_state.randint(self.n_clusters, size=n_samples)
            self.cluster_centers_ = np.zeros((self.n_clusters,n_features))
            for i in range(self.n_clusters):
                self.cluster_centers_[i] = X[ra==i].mean(axis=0)

        for i in range(self.max_iter):
            centers_old = self.cluster_centers_.copy()

            self._calculate_centers(X)

            if np.sum((centers_old - self.cluster_centers_) ** 2) < self.tol * vdata:
                break

        return self.m_.argmax(axis=1)