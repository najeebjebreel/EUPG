import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from tqdm import tqdm

def dist(x,y):
    return np.linalg.norm(x-y, axis=1)

def poprow(arr, y, i):
    pop_a = arr[i]
    pop_y = y[i]
    new_array = np.vstack((arr[:i],arr[i+1:]))
    new_y = np.hstack((y[:i], y[i+1:]))

    return new_array, new_y, pop_a, pop_y

def cluster(X, y, px, py, k, dist_to_xr):
    cx = [px]
    cy = [py]

    if dist_to_xr is None:
        distances = dist(px,X)
    else:
        distances = dist_to_xr
        
    X = X[np.argpartition(distances, k-1)]
    y = y[np.argpartition(distances, k-1)]
    cx.extend(X[:k-1])
    cy.extend(y[:k-1])
    X = X[k-1:]
    y = y[k-1:]

    return X, y, np.array(cx), np.array(cy)
    
def mdav(X, y, k):
    D = X
    clusters = []
    labels = []
    
    # Test feature. progress bar
    pbar = tqdm(total=len(D))
    
    while len(D) >= 3*k:
        # Centroid
        xm = np.mean(D, axis=0)
        # Furthest from centroid
        xri = np.argmax(dist(xm,D))
        D, y, xr, yr = poprow(D, y, xri)
        # Furthest from furthest from centroid
        dist_to_xr = dist(xr,D)
        xsi = np.argmax(dist_to_xr)
        dist_to_xr = np.append(dist_to_xr[:xsi], dist_to_xr[xsi+1:], axis=0)
        D, y, xs, ys = poprow(D, y, xsi) 

        #cluster of xr
        D, y, cx, cy = cluster(D, y, xr, yr, k, dist_to_xr)
        clusters.append(cx)
        labels.append(cy)
        
        # Test feature. progress bar
        pbar.update(k)
        
        #cluster of xs
        D, y, cx, cy = cluster(D, y, xs, ys, k, None)
        clusters.append(cx)
        labels.append(cy)
        
        # Test feature. progress bar
        pbar.update(k)
        
    if len(D) >= 2*k and len(D) < 3*k:
        # Centroid
        xm = np.mean(D, axis=0)
        # Furthest from centroid
        xri = np.argmax(dist(xm,D))
        D, y, xr, yr = poprow(D, y, xri)
        #cluster of xr
        D, y, cx, cy = cluster(D, y, xr, yr, k, None)
        clusters.append(cx)
        labels.append(cy)
        
        # Test feature. progress bar
        pbar.update(k)
        
        # rest of points
        clusters.append(D[:])
        labels.append(y[:])
        
        # Test feature. progress bar
        pbar.update(len(D))
        
    else:
        # rest of points
        clusters.append(D[:])
        labels.append(y[:])
        
        # Test feature. progress bar
        pbar.update(len(D))
    
    centroids = np.array([np.mean(c,axis=0) for c in clusters], copy=False)
    data_k = np.vstack(np.repeat(c.mean(0).reshape(1, -1), len(c), axis = 0) for c in clusters)

    y_k = None
    for i in range(len(labels)):
        yc = labels[i]
        if y_k is None:
            y_k = yc
        else:
            y_k = np.hstack((y_k, yc))
    
    return centroids, clusters, labels, data_k, y_k


def print_stats(clusters, centroids):
    ss = []
    for c,cen in zip(clusters, centroids):
        #cen = np.mean(c, axis=0)
        s = np.mean(dist(cen[0],c), axis=0)
        ss.append(s)
        
    print(f'Number of clusters: {len(clusters)}')
    print(f'Mean of mean distances to centroids: {np.mean(ss, axis=0)}')


    
