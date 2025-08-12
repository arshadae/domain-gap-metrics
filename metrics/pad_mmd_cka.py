import numpy as np, math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def compute_pad(feats_a, feats_b, C=1.0):
    X = np.vstack([feats_a, feats_b])
    y = np.hstack([np.zeros(len(feats_a)), np.ones(len(feats_b))])
    mu, sd = X.mean(0), X.std(0)+1e-8
    X = (X-mu)/sd
    n = len(X); split = int(0.7*n)
    idx = np.random.permutation(n); tr, te = idx[:split], idx[split:]
    clf = LogisticRegression(max_iter=2000, C=C, n_jobs=-1)
    clf.fit(X[tr], y[tr])
    acc = accuracy_score(y[te], clf.predict(X[te])); err = 1-acc
    pad = 2*(1-2*err)
    return float(pad), float(acc)

def compute_mmd_rbf(Xa, Xb, gammas=(1e-3,1e-4,1e-5)):
    def rbf(x,y,g):
        x2 = (x**2).sum(1)[:,None]; y2 = (y**2).sum(1)[None,:]
        return np.exp(-g*(x2 + y2 - 2*x@y.T))
    m,n = Xa.shape[0], Xb.shape[0]; v=0.0
    for g in gammas:
        Kaa = rbf(Xa,Xa,g); np.fill_diagonal(Kaa,0)
        Kbb = rbf(Xb,Xb,g); np.fill_diagonal(Kbb,0)
        Kab = rbf(Xa,Xb,g)
        v += Kaa.sum()/(m*(m-1)) + Kbb.sum()/(n*(n-1)) - 2*Kab.mean()
    return float(v/len(gammas))

def compute_linear_cka(X, Y):
    n = min(len(X), len(Y)); X, Y = X[:n], Y[:n]
    X = (X-X.mean(0))/(X.std(0)+1e-8); Y = (Y-Y.mean(0))/(Y.std(0)+1e-8)
    Xc = X - X.mean(0); Yc = Y - Y.mean(0)
    import numpy.linalg as la
    hsic = la.norm(Xc.T @ Yc, 'fro')**2
    xkx = la.norm(Xc.T @ Xc, 'fro')**2
    yky = la.norm(Yc.T @ Yc, 'fro')**2
    return float(hsic / (math.sqrt(xkx)*math.sqrt(yky) + 1e-12))
