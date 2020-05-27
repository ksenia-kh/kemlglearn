"""
.. module:: HarmonicKMeans

HarmonicKMeans
*************

:Description: HarmonicKMeans


"""

# Author: Ksenia Kharitonova <ksenia.kharitonova@gmail.com>
# License: BSD 3 clause

import numpy as np

from sklearn.base import TransformerMixin, ClusterMixin, BaseEstimator
from sklearn.utils import check_random_state

class HarmonicKMeans(TransformerMixin, ClusterMixin, BaseEstimator):
    """
    Harmonic K-means

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
        # distances between cluster centers and X with the degree p
        dist_p = ((X[:,np.newaxis]-self.cluster_centers_)**self.p).sum(axis=2)

        # check that we don't divide by 0 when inverting distances - Zhang
        dist_p_plus_2[dist_p_plus_2<=self.epsilon]=self.epsilon
        dist_p[dist_p<=self.epsilon]=self.epsilon

        # calculate membership m 
        M1 = dist_p_plus_2**(-1)
        self.m_ = M1/np.sum(M1, axis=1).reshape(-1,1)

        # calculate weight w
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

    def predict(self, X):
        '''Predicts the hard partition for the new data X based on the fitted instance
        and converged self.cluster_centers_ (if the cluster_centers_ attribute does not exist 
        raises an exception).
        '''
        
        try:
            dist_p_plus_2 = ((X[:,np.newaxis]-self.cluster_centers_)**(self.p+2)).sum(axis=2)
        
            # check that we don't divide by 0 when inverting distances - Zhang
            dist_p_plus_2[dist_p_plus_2<=self.epsilon]=self.epsilon
        
            # calculate membership m 
            M1 = dist_p_plus_2**(-1)
            m_ = M1/np.sum(M1, axis=1).reshape(-1,1)
    
        except AttributeError:
            raise NotFittedError("This instance is not fitted yet. Call 'fit_predict' method before using this method")
        
        return m_.argmax(axis=1)

    def predict_proba(self, X):
        '''Predicts the soft partition for the new data X based on the fitted instance
        and converged self.cluster_centers_ (if the cluster_centers_ attribute does not exist 
        raises an exception).
        '''
        try:
            dist_p_plus_2 = ((X[:,np.newaxis]-self.cluster_centers_)**(self.p+2)).sum(axis=2)
        
            # check that we don't divide by 0 when inverting distances - Zhang
            dist_p_plus_2[dist_p_plus_2<=self.epsilon]=self.epsilon
        
            # calculate membership m 
            M1 = dist_p_plus_2**(-1)
            m_ = M1/np.sum(M1, axis=1).reshape(-1,1)
    
        except AttributeError:
            raise NotFittedError("This instance is not fitted yet. Call 'fit_predict' method before using this method")
        
        return m_


if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    centers = [[1, 1], [-1, -1], [1, -1]]
    n_clusters = len(centers)
    X, labels_true = make_blobs(n_samples=1200, centers=centers, cluster_std=0.3, random_state=42)

    km = HarmonicKMeans(n_clusters=3, init_method='random_partition', max_iter=100, random_state=42)
    labels = km.fit_predict(X)
    print(km.cluster_centers_)
    print(labels[:10])
    print(km.m_[:10])