# import numpy as np
# from sklearn.cluster import KMeans

# X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
# kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
# print("cluster label: ",kmeans.labels_)
# print("predict label: ",kmeans.predict([[0, 0], [12, 3]]))
# print("center: ",kmeans.cluster_centers_)

"""tu lm"""
# import numpy as np

# X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])


# def distance(x1, x2):
#     return np.sqrt(np.sum((x1-x2)**2))


# def clustering(n, X):
#     center = X[np.random.choice(X.shape[0], size=n, replace=False), :]
#     print(X.shape[0])
#     label = np.zeros(X.shape[0])
#     while True:
#         cluster = [[] for i in range(n)]
#         for i in range(X.shape[0]):
#             min=float('inf')
#             for j in range(n):
#                 if distance(X[i], center[j]) < min:
#                     min = distance(X[i], center[j])
#                     label[i] = j # gan nhan cho diem
#             cluster[int(label[i])].append(X[i])# phan diem vao cum
#         # calculate new center by average of cluster
#         newCenter = np.array([np.mean(cluster[i], axis=0) for i in range(n)])
#         # print(newCenter, center)
#         print((newCenter==center).all())
#         if(center == newCenter).all():
#             break
#         else:
#             center = newCenter
#     print(label)


# clustering(2, X)

"""code trên mạng"""
import numpy as np
from scipy.spatial.distance import cdist
def kmeans_init_centers(X, k):
    # randomly pick k rows of X as initial centers
    return X[np.random.choice(X.shape[0], k, replace=False)]

def kmeans_assign_labels(X, centers):
    # calculate pairwise distances btw data and centers
    D = cdist(X, centers)
    # return index of the closest center
    return np.argmin(D, axis = 1)

def kmeans_update_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        # collect all points assigned to the k-th cluster 
        Xk = X[labels == k, :]
        # take average
        centers[k,:] = np.mean(Xk, axis = 0)
    return centers

def has_converged(centers, new_centers):
    # return True if two sets of centers are the same
    return (set([tuple(a) for a in centers]) == 
        set([tuple(a) for a in new_centers]))
    
def kmeans(X, K):
    centers = [kmeans_init_centers(X, K)]
    labels = []
    it = 0 
    while True:
        labels.append(kmeans_assign_labels(X, centers[-1]))
        new_centers = kmeans_update_centers(X, labels[-1], K)
        if has_converged(centers[-1], new_centers):
            break
        centers.append(new_centers)
        it += 1
    return (centers, labels, it)

X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
K=2
(centers, labels, it) = kmeans(X, K)
print('Centers found by our algorithm:')
print(centers[-1])