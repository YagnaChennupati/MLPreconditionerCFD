import numpy as np                          # numpy is for numbers and arrays
import matplotlib.pyplot as plt             # for plotting convergence curves
from scipy.sparse import diags, kron, eye   # sparse data, diags for creating sparse matrices from diags, eye creates identitiy matrix, kron bigger matrix from smaller matrices.
from scipy.sparse.linalg import spsolve, cg, LinearOperator    # sparse direct solve, conjugate gradient solve, can be used since Poisson matrix is both symmetric and positive definite, for jacobi preconditioning (using diag A as a preconditioner, ignores neighbours)

# 1D Poission Matrix #
N = 32                                      # number of points
main_diag = 2 * np.ones(N)                  # main diagonal (point in consideration)
off_diag  = -1 * np.ones(N-1)               # off diagonal (upper and lower neighbours for the point in consideration)

# telling Scipy where to place the defined main_diag and off_diag, 1D Poisson Matrix T

T = diags(
    diagonals=[off_diag, main_diag, off_diag], 
    offsets=[-1,0,1],
    shape=(N,N), 
    format ="csr"                          # storage format (compressed sparse row - efficient for sparse matrix operations)
    )

I = eye(N, format="csr")                   # identity matrix

# We can build 2D Matrix (A in this case) by combining 1D Matrix with an Identity Matrix
# kron (A,B) means take every number in matrix A and replace it with that number multiplied by the whole matrix B
A = kron(I,T)+kron(T,I)                    # (I,T) takes care of left and right, whereas (T,I) takes care of top,bottom

# So far, T is the 1D Poisson Matrix, I is the Identity Matrix and A is the 2D Poisson Matrix
# The kron operation stores zeroes too, so we convert the kronecker A to csr format, and eliminate all zeroes
A = A.tocsr()
A.eliminate_zeros()
## print("Number of non-zero values:", A.nnz)
# we can't use np.cond on sparse matrices, it throws an error so we first gotta convert it into a dense matrix
A_dense = A.toarray()
A_cond = np.linalg.cond(A_dense)
print("Condition Number of A=",A_cond)

# create a simple b for now with 1's
n = N * N
b = np.ones(n)

# direct sparse solve for baseline
x_direct = spsolve(A, b)
## print(x_direct.reshape(N, N))

residual = b - A @ x_direct
## print("Residual:", residual)
## print("Residual norm:", np.linalg.norm(residual)) # single number for the whole matrix, sqrt(r1^2+r2^2+...r16^2)

# residual norm almost 0 for direct solve, so solution/implementation is valid
# spsolve solves directly, cg starts with a guess, improves step by step and stops when residual small
x_cg, info = cg(A, b)   # using conjugate gradient to solve Ax=b, x_cg is soln found through CG and info is status code, 0 converged, < 0 smth wrong, >0, didn't converge fully in defined no.of iterations
## print("info:", info)

# Seeing difference between direct and CG solve
difference = x_direct - x_cg
## print("Difference between direct and CG solution:", difference)
## print("Norm of difference:", np.linalg.norm(difference))

# xk is the current CG soln at this iteration, save/append the residual details to residuals_cg
residuals_cg = [] # creates an empty list, where residual norm at each CG iteration is stored
def record_residual(xk):
    residual = b - A @ xk
    residual_norm = np.linalg.norm(residual)
    residuals_cg.append(residual_norm)
x_cg_tracked, info_cg_tracked = cg(A, b, callback= record_residual) 
print("CG info:", info_cg_tracked)
print("Number of CG iterations:", len(residuals_cg))
print("CG residuals:", residuals_cg)

diag_A = A.diagonal()
## print(diag_A)
M = np.diag(diag_A) # M equals A diagonal for Jacobi
M_inv = np.linalg.inv(M)
A_jacobipreconditioned = M_inv @ A_dense
A_jacobipreconditioned_cond = np.linalg.cond(A_jacobipreconditioned)
print("Condition Number of Jacobi-preconditioned A=",A_jacobipreconditioned_cond)

# residual = what is still wrong
# z = correction we want
# ideal:   A z = residual
# Jacobi:  diag(A) z = residual
# so:      z = residual / diag(A)
def jacobi_apply(r):
    return r / diag_A
M_jacobi = LinearOperator(
    shape=A.shape,
    matvec=jacobi_apply
)

residuals_jacobi = []
def record_residual_jacobi(xk):
    residual = b - A @ xk
    residual_norm = np.linalg.norm(residual)
    residuals_jacobi.append(residual_norm)
x_jacobi, info_jacobi = cg(A, b, M=M_jacobi, callback= record_residual_jacobi) 
print("CG + Jacobi info:", info_jacobi)
print("Number of CG + Jacobi iterations:", len(residuals_jacobi))
print("CG + Jacobi residuals:", residuals_jacobi)

# Plot convergence comparison
iterations_cg = np.arange(1, len(residuals_cg) + 1)
iterations_jacobi = np.arange(1, len(residuals_jacobi) + 1)
plt.figure(figsize=(8, 5))

plt.semilogy(iterations_cg, residuals_cg, marker="o", label="CG")
plt.semilogy(iterations_jacobi, residuals_jacobi, marker="s", linestyle="--", label="CG + Jacobi")

plt.xlabel("Iteration")
plt.ylabel("Residual norm")
plt.title("2D Poisson - Convergence Comparison: CG vs CG + Jacobi")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()