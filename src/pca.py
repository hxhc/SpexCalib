import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import f
from sklearn.decomposition import PCA


def auto_scale(X):
    return (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)


class pca:

    def __init__(self, X, n_components=0.85, autoscale=True):
        self.X = X
        self.n_components = n_components

        if autoscale:
            self.X_ = auto_scale(X)
        else:
            self.X_ = X

        self.pca_model = PCA(n_components=self.n_components, svd_solver="full")
        self.pca_model.fit(X)

    def scores(self):
        scores = self.pca_model.transform(self.X_)
        return scores

    def loadings(self):
        loadings = self.pca_model.components_.T
        return loadings

    def Hotelling_T2(self):
        hotelling_t2 = np.sum((self.scores() / self.scores().std(ddof=1, axis=0)) ** 2, axis=1)
        return hotelling_t2

    def Hotelling_T2_limit(self, alpha_list=[0.95]):
        n = self.X_.shape[0]
        limit_list = []
        for alpha in alpha_list:
            hotelling_t2_limit = [i * (n - 1) / (n - i) * f.ppf(alpha, i, n - i) for i in [self.n_components]]
            limit_list.append(hotelling_t2_limit)
        return limit_list

    def reconstruct_X(self):
        return self.pca_model.inverse_transform(self.scores())

    def SPE(self):
        error_matrix = self.X_ - self.reconstruct_X()
        spe_matrix = np.linalg.norm(error_matrix, axis=1)
        return spe_matrix

    # def SPE_limit(self):

    def plot_2d_score(self, axis_x=0, axis_y=1, alpha_list=[0.95]):
        limit_x_list = []
        limit_y_list = []
        n = self.X_.shape[0]
        for alpha in alpha_list:
            limit_x_ = self.scores()[:, axis_x].std(ddof=1) * np.sqrt(2 * (n - 1) / (n - 2) * f.ppf(alpha, 2, n - 2))
            limit_y_ = self.scores()[:, axis_y].std(ddof=1) * np.sqrt(2 * (n - 1) / (n - 2) * f.ppf(alpha, 2, n - 2))
            limit_x_list.append(limit_x_)
            limit_y_list.append(limit_y_)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(len(alpha_list)):
            ellipse = Ellipse((0, 0), limit_x_list[i] * 2, limit_y_list[i] * 2, fill=None, edgecolor="r")
            ax.add_patch(ellipse)
        ax.scatter(self.scores()[:, axis_x], self.scores()[:, axis_y], alpha=0.7, c='g')
        ax.grid(color="lightgray", linestyle="--")

        return limit_x_list, limit_y_list
