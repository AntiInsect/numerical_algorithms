from pprint import pprint
# only use np.array
import numpy as np 


def generate_Hilbert(n):
    Hilbert = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Hilbert[i][j] = 1 / (i+j+1)
    return Hilbert

def cholesky_decomposition(Hilbert):
    # Initializing an nxn matrix of zeros
    L = np.zeros((len(Hilbert), len(Hilbert)))
    # Initializing the first element in the matrix
    L[0][0] = (Hilbert[0][0])**0.5
    
    # Initializing the first column of the matrix
    for i in range(1, len(Hilbert)):
        L[i][0] = Hilbert[0][i] / L[0][0]
    
    # Filling-in elsewhere
    for i in range(1, len(Hilbert)):
        for j in range(1, i+1):
            # Filling the main diagonal
            if i == j:
                L[i][j] = (Hilbert[i][j] - sum((L[i][k]**2) for k in range(0, i)))**0.5
            # Filling below the main diagonal
            else:
                L[i][j] = (1/L[j][j])*(Hilbert[i][j] - sum(L[i][k]*L[j][k] for k in range(0, min(i,j)))) 
    return L

if __name__ == "__main__":
    N = 5 
    hilbert_mat = generate_Hilbert(N)
    matrix = np.matrix(cholesky_decomposition(hilbert_mat))
    pprint(hilbert_mat)
    pprint(np.array(matrix) @ np.array(matrix).T)

    # from scipy.linalg import hilbert
    # print(np.linalg.cholesky(hilbert(N)))
