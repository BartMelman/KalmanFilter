import numpy as np


def KalmanSmoother(y, A, C, Q, R, zStart, Sigma, epsilonInverse=1e-3):
    T = y.shape[1]
    n = y.shape[0]
    numStates = zStart.shape[0]
    
    # Initialise Matrices
    zt1t = np.zeros((numStates,T)) # z(t+1|t)
    ztt = np.zeros((numStates,T+1)) # z(t|t)
    ztT = np.zeros((numStates,T+1)) # z(t|T)
    Pt1t = np.zeros((numStates,numStates,T)) # P(t+1|t)
    Ptt = np.zeros((numStates,numStates,T+1)) # P(t|t)
    PtT = np.zeros((numStates,numStates,T+1)) # P(t|T)
    Pt1tT = np.zeros((numStates, numStates,T)) # P(t+1,t|T)
    L = np.zeros((numStates,numStates, T+1))
    K = np.zeros((numStates, n, T+1)) # Kalman Gain

    ### Forward pass
    # initialise
    Ptt[:,:,0] = Sigma
    ztt[:, 0:1] = zStart

    for t in range(T):
        zt1t[:,t] = A @ ztt[:,t]
        Pt1t[:,:,t] = A @ Ptt[:,:,t] @ A.T + Q 
        K[:,:,t+1] = Pt1t[:,:,t] @ C.T @ np.linalg.inv(C @ Pt1t[:,:,t] @ C.T + R)
        ztt[:,t+1:t+2] = zt1t[:,t:t+1] + K[:,:,t+1] @ (y[:,t:t+1] - C @ zt1t[:,t:t+1])
        Ptt[:,:,t+1] = Pt1t[:,:,t] - K[:,:,t+1] @ C @ Pt1t[:,:,t]

    ### Backward pass
    # initialise
    PtT[:,:,T] = Ptt[:,:,T]
    ztT[:,T] = ztt[:,T]
    Pt1tT[:,:,T-1] = ( np.eye(numStates) - K[:,:,T] @ C) @ A @ Ptt[:,:,T-1]

    for t in range(T-1,-1,-1):
        L[:,:,t] = Ptt[:,:,t] @ A.T @ np.linalg.inv(Pt1t[:,:,t]) # +np.eye(Pt1t.shape[0])*0.00001
        ztT[:,t] = ztt[:,t] + L[:,:,t] @ (ztT[:,t+1]-zt1t[:,t])
        PtT[:,:,t] = Ptt[:,:,t] + L[:,:,t] @ (PtT[:,:,t+1] - Pt1t[:,:,t]) @ L[:,:,t].T 
        if t <= T-2:  
            Pt1tT[:,:,t] = Ptt[:,:,t+1] @ L[:,:,t].T + L[:,:,t+1] @ (Pt1tT[:,:,t+1] - A @ Ptt[:,:,t+1]) @ L[:,:,t].T

    return ztT, PtT, Pt1tT, zt1t, Pt1t, ztt, zt1t


def generateTimeSeries(A, C, Q, R, T):

    numStates = A.shape[0]
    n = C.shape[0]

    z = np.zeros((numStates,T))
    y = np.zeros((n,T))

    for t in range(T-1):
        z[:,t+1:t+2] = A @ z[:,t:t+1] + np.random.multivariate_normal(np.zeros((numStates)), Q).reshape(-1,1)

    for t in range(T):
        y[:,t:t+1] = C @ z[:,t:t+1] + np.random.multivariate_normal(np.zeros((n)), R).reshape(-1,1)
    
    return z, y

def getExperiment(experiment):
    if experiment == 1:
        Q = np.asarray([[1, 0], [0, 1]])
        R = np.asarray([[0.3, 0], [0, 0.3]])
        A = np.asarray([[0.9, 0.1], [0.1, 0.9]])
        C = np.asarray([[1, 0], [0, 1]])
    elif experiment == 2:
        Q = np.asarray([[1]])
        R = np.asarray([[1]])
        A = np.asarray([[1]])
        C = np.asarray([[1]])
    elif experiment == 3:
        Q = np.asarray([[1, 0.1], [0.1, 1]])
        R = np.asarray([[0.3, 0], [0, 0.3]])
        A = np.asarray([[0.9, 0.1], [-0.2, 0.9]])
        C = np.asarray([[1, 0], [0, 1]])
    else:
        raise ValueError('experiment {experiment} not in options', experiment)
    
    return A, Q, C, R


def getParams(ztT, PtT, Pt1tT, y):
    """Get the matrices D, E, F, H, M used within the EM algorithm. 

    Args:
        ztT ([type]): numStates x (T+1)
        PtT ([type]): numStates x numStates x (T+1)
        Pt1tT ([type]): numStates x numStates x (T+1)
        y ([type]): n x T

    Returns:
        [type]: [description]
    """    
    n, T = y.shape
    numStates = ztT.shape[0]

    D = np.zeros((numStates, numStates))
    E = np.zeros((numStates, numStates))
    F = np.zeros((numStates, numStates))
    H = np.zeros((n, n))
    M = np.zeros((n, numStates))

    for t in range(T):
        D += PtT[:,:,t] + ztT[:,t:t+1] @ ztT[:,t:t+1].T
        E += Pt1tT[:,:,t] + ztT[:,t+1:t+2] @ ztT[:,t:t+1].T
        F += PtT[:,:,t+1] + ztT[:,t+1:t+2] @ ztT[:,t+1:t+2].T
        H += y[:,t:t+1] @ y[:,t:t+1].T
        M += y[:,t:t+1] @ ztT[:,t+1:t+2].T
    
    return D, E, F, H, M


def getApproxParameters(ztT, PtT, Pt1tT, y, A=None, C=None, Q=None, R=None):
    """[summary]

    Args:
        ztT (np.array): numStates x (T+1)
        PtT (np.array): numStates x numStates x (T+1)
        Pt1tT (np.arry): numStates x numStates x (T+1)
        y (np.array): n x T
        A (np.array, optional): numStates x numStates. Defaults to None.
        C (np.array, optional): n x numStates. Defaults to None.
        Q (np.array, optional): numStates x numStates. Defaults to None.
        R (np.array, optional): n x n. Defaults to None.

    Returns:
        [type]: [description]
    """    
    T = y.shape[1]
    D, E, F, H, M = getParams(ztT, PtT, Pt1tT, y)
    
    if A is None:
        DInv = np.linalg.inv(D)
        A = E @ DInv
        if Q is None:
            Q = 1 / T * (F - E @ DInv @ E.T)
    else:
        if Q is None:
            Q = 1 / T * (F - E @ A.T - A @ E.T + A @ D @ A.T)

    
    if C is None:
        FInv = np.linalg.inv(F)
        C = M @ FInv
        if R is None:
            R = 1 / T * (H - M @ FInv @ M.T)
    else:
        if R is None:
            R = 1 / T * (H - M @ C.T - C @ M.T + C @ F @ C.T)

    return A, C, Q, R


def getExpectedLogLikelihood(ztT, PtT, Pt1tT, y, A, C, Q, R, 
Sigma, mu): 
    T = y.shape[1]
    D, E, F, H, M = getParams(ztT, PtT, Pt1tT, y)
    LL =  - 0.5 * np.log(np.linalg.det(Sigma)) 
    LL += - 0.5 * np.trace(np.linalg.inv(Sigma) @ (PtT[:,:,0] + (ztT[:,0:1]- mu) @ (ztT[:,0:1] - mu).T))
    LL += - 0.5 * T * np.log(np.linalg.det(Q)) 
    LL += - 0.5 * np.trace(np.linalg.inv(Q) @ (F - E @ A.T - A @ E.T + A @ D @ A.T)) 
    LL += - 0.5 * T * np.log(np.linalg.det(R)) 
    LL += - 0.5 * np.trace(np.linalg.inv(R) @ (H - M @ C - C @ M.T + C @ F @ C.T))
    for t in range(T+1):
        LL += 0.5*np.log(np.linalg.det(PtT[:,:,t]))
    LL = LL / T 
    return LL

def getConditionalLogLikelihood(y, zt1t, Pt1t, C, R):
    T = y.shape[1]
    LL = 0
    for t in range(T):
        Sigma = C @ Pt1t[:,:,t] @ C.T + R
        SigmaInv = np.linalg.inv(Sigma)
        u = C @ zt1t[:,t:t+1]
        LL += -0.5 * np.log(np.linalg.det(Sigma))
        LL += -0.5 * (y[:,t:t+1]-u).T @ SigmaInv @ (y[:,t:t+1]-u)
    LL = LL / T
    return LL[0][0]