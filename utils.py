import numpy as np
from numpy.linalg import LinAlgError
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
from scipy.special import gamma
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import learning_curve

def randncor(n, N, C):
    try:
        A = cholesky(C)
    except LinAlgError:
        m = 0
        print('A is not positive definite')
    m = n
    u = np.random.randn(m, N)
    x = A.conj().transpose().dot(u)
    return x

def score(p,p_,scoring = 'mea'):
    if scoring == 'mea':
        err = np.average(np.abs(p-p_))
    if scoring == 'mse':
        err  = np.average((p-p_)**2)
    return err

def parzen_window(x,train,h,kernel = 'gauss_diag'):

    """Probablity density estimation with given kernel function.
    
    Parameters
    ----------
    x : array-like, shape = [n_samples] or [n_dimensions, n_samples]
        
        
    train : array-like, shape = [n_samples] or [n_dimensions, n_samples]
        Training data
    
    h : float
        Window bandwidth
    
    kernel: {'gauss_diag','gauss_cov','exp','rect','tri'}
        Kernel to use in computations:
        
        - 'gauss_diag'
        
        - 'gauss_cov'
        
        - 'exp'
        
        - 'rect'
        
        - 'tri'
        
    Returns
    -------
    p_ : array, 
    Probability density estimation.
    
    """

    if x.ndim ==1:
        n1 = 1
        mx = x.shape[0]
    else:
        n1,mx = x.shape

    if train.ndim ==1:
        n2 = 1
        N = train.shape[0]
    else:
        n2,N = train.shape


    if n1 != n2:
        raise ValueError("Number of dimensions in x and train does not correspond:"
                         " %d != %d" % (n1, n2))
    if kernel not in ('gauss_diag','gauss_cov','exp','rect','tri','custom'):
        raise ValueError('Kernel %s not understood' % kernel)

    n = n1
    C = np.ones([n,n])
    C_ = C.copy()

   
    if kernel == 'gauss_cov' and N>1:
        C = np.zeros([n,n])
        m_ = np.transpose(np.mean(np.transpose(train),axis = 0))
        for i in range(N):
            C = C + np.matrix(train[:,i] - m_).T*(train[:,i] - m_)
        C=C/(N-1)
        C_=C**(-1)
    
    fit = np.zeros([N,mx])
    for i in range(N):
        p_k = np.zeros([n,mx])
        mx_i = np.tile(train[:,i].reshape(-1,1),[1,mx])

        if kernel == 'gauss_diag':
            ro = np.sum(np.asarray(x-mx_i)**2,0)
            fit[i,:] = np.exp(-ro/(2*h**2))/((2*np.pi)**(n/2)*(h**n))

        elif kernel == 'gauss_cov':
            d = x-mx_i
            dot = np.dot(C_,d)
            ro=np.sum(np.multiply(dot,d),0)
            fit[i,:]=np.exp(-ro/(2*h**2))*((2*np.pi)**(-n/2)*(h**(-n))*(np.linalg.det(C)**(-0.5)))


        elif kernel == 'exp':
            ro=np.abs(x-mx_i)/h
            fit[i,:]=np.prod(np.exp(-ro),0)/(2*h**n)

        elif kernel == 'rect':
            ro=np.abs(x-mx_i)/h
            for k in range(n):
                ind= np.nonzero(ro[k,:]<1)[1]
                p_k[k,[ind]]=1/2
            fit[i,:]=np.prod(p_k,axis = 0)/h**n

        elif kernel == 'tri':
            ro=np.abs(x-mx_i)/h
            for k in range(n):
                if n == 1:
                    ind= np.nonzero(ro[k,:]<1)[1]
                else:
                    ind= np.nonzero(ro[k,:]<1)
                p_k[k,[ind]]= 1-ro[k,[ind]]
            fit[i,:]=np.prod(p_k,axis = 0)/h**n
        
        elif kernel == 'custom':
            ro = np.asarray((x-mx_i)/h)**2
            fit[i,:] = (1/(np.pi*h))*(1/(1+ro))
        if N>1:
            p_ = np.sum(fit,0)/N
        else:
            p_ = fit
    return p_

def knn(x,XN,k):
#x-массив векторов (точек), для которых роводится оценка плотности
#XN-входная обучающая выборка данных
#k - число ближайших соседей

    if x.ndim ==1:
        n1 = x.shape[0]
        mx = 1
    else:
        n1,mx = x.shape

    if XN.ndim ==1:
        n2 = 1
        N = XN.shape[0]
    else:
        n2,N = XN.shape
        
    if n1 != n2:
        raise ValueError("Number of dimensions in x and train does not correspond:"
                         " %d != %d" % (n1, n2))
    if k>N:
        raise ValueError("Number of neighbors is greater than number of training vectors"
                        " %k > %N"%(k,N))
    nn = NearestNeighbors(n_neighbors = k)
    n=n1 
    p_=np.zeros([1,mx])
    Cn=2*(np.pi**(n/2))/(n*gamma(n/2));
    nn.fit(XN.T)
    if mx == 1:
        d, ind= nn.kneighbors(x.reshape(1,-1))
    else:
        d, ind= nn.kneighbors(x.T)
    r=d[:,k-1]
    V=Cn*r**n;
    p_=(k/N)/V.T
    return p_


def plot_err(h_N,errors):
    plt.plot(h_N,errors,'m');
    plt.xlabel('Ширина окна, h');
    plt.ylabel('Cредняя абсолютная ошибка');
    plt.title('График зависимости ошибки оценивания от величины параметра оконной функции');

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 10)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Размер выборки")
    plt.ylabel("Точность")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.grid()
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Обучение")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Кросс-валидация")

    plt.legend(loc="best")
    return plt