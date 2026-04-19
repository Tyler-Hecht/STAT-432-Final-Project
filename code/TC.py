import numpy as np

def TC(C: np.ndarray):
    N = C.sum()
    M = len(C)
    ns = C.sum(axis = 0)

    cost = 0
    for i in range(M):
        for j in range(M):
            gamma = (N - ns[j])/ns[i]
            cost += C[i, j] * gamma * abs(i - j)
    
    return cost

if __name__ == "__main__":
    # Example usage
    # C should be a confusion matrix where true labels would be on top and predicted labels on the side
    C = np.diag([3,6,8,5])
    C[1,2] = 1
    C[1,3] = 2
    C[3,0] = 1
    print(C)

    print(TC(C))