#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
from sklearn.datasets import load_digits
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GaussianMixture as GMM


# ## Kernel Density Estimation

# In[49]:


def kernel_density_estimation():
    
    digits = load_digits()

    pca = PCA(n_components=15, whiten=False)
    data = pca.fit_transform(digits.data)

    
    params = {'bandwidth': np.logspace(0, 1, 20)}
    
    #print(params)
    
    grid = GridSearchCV(KernelDensity(), params, cv=5)
    grid.fit(data)

    print("best bandwidth: ",grid.best_estimator_.bandwidth)

    kde = grid.best_estimator_

    generate_new_data(kde,pca)


# In[30]:


def plot_kde_data(new_data):
    fig, ax = plt.subplots(4, 12, subplot_kw=dict(xticks=[], yticks=[]))
    for j in range(12):

        for i in range(4):

            im = ax[i, j].imshow(new_data[i, j].reshape((8, 8)),cmap=plt.cm.binary, interpolation='nearest')
            im.set_clim(0, 16)

    ax[0, 5].set_title('"New" digits sampled from the kernel density model')
    plt.show()


# ## Generating New Data from KDE

# In[34]:


def generate_new_data(kde,pca):
    
    new_data = kde.sample(48)
    new_data = pca.inverse_transform(new_data)

    new_data = new_data.reshape((4, 12, -1))

    plot_kde_data(new_data)


# In[50]:


kernel_density_estimation()


# ## Gaussian Mixture Model Density Estimation

# In[53]:


def gmm_density_estimation():
    
    digits = load_digits()
    digits.data.shape

    pca = PCA(n_components=29)
    data = pca.fit_transform(digits.data)

    print(data.shape)

    n_components = np.arange(50, 200, 10)
    models = [GMM(n, covariance_type='full', random_state=0) for n in n_components]

    plt.plot(n_components, [model.fit(data).bic(data) for model in models], label='BIC')
    plt.legend(loc='best')
    plt.xlabel('n_components');

    gmm = GMM(115, covariance_type='full')
    gmm.fit(data)
    print("Convergence: ",gmm.converged_)

    data_new_X, data_new_Y = gmm.sample(48)
    
#     print(data_new_X.shape)
#     print(data_new_Y.shape)

    generate_data_GMM(pca,data_new_X)


# In[37]:


def plot_digits(data):
    
    fig, ax = plt.subplots(6, 8, figsize=(8, 8),subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(8, 8), cmap='binary')
        im.set_clim(0, 16)


# ## Generating New Data Using GMM Density Estimation

# In[17]:


def generate_data_GMM(pca,data):
    
    digits_new = pca.inverse_transform(data)
    plot_digits(digits_new)


# In[54]:


gmm_density_estimation()

