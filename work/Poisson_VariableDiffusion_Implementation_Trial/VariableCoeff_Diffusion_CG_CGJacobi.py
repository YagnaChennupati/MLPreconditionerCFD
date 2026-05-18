import numpy as np # for numbers and arrays
import matplotlib.pyplot as plt # for convergence plots
from scipy.sparse import lil_matrix # list of lists sparse matrix format (allows us to build matrix row by row) - manual matrix entries
from scipy.sparse.linalg import spsolve, cg, LinearOperator # (direct solve, conjugate gradient, used in jacobi)

N = 32    # number of grid points
n = N*N  # total number of unknowns (since 2D)

# Creating an empty sparse matrix (Full of zeroes at this stage, cuz no elements in matrix defined yet)
A = lil_matrix((n,n))

# Creating the spatially varying diffusion coefficient k(x,y)
x = np.linspace(0,1,N)  # creates N points from 0 to 1 in the x-direction
y = np.linspace (0,1,N) # creates N points from 0 to 1 in the y-direction

X, Y = np.meshgrid(x,y) # creates 2D coordinate arrays

k = 1.0 + 9.0*X         # k(x,y) = 1+9x so k varies from left to right, left k = 1, right k = 10 (diffusion weaker on left, stronger on right)
# k depends only on x so as you move left to right, k increases

## print(k.shape)
## print(k)

# Convert a 2D grid index (i, j) into a 1D index for the vector/matrix system.
#
# The grid unknowns are arranged in 2D, but Ax = b stores them as a 1D vector.
# We flatten row by row:
#
# For N = 4:
# (0,0)->0, (0,1)->1, (0,2)->2, (0,3)->3
# (1,0)->4, (1,1)->5, ...
def idx(i,j):
    return i*N+j
# Loop over every grid point.
# Each grid point gives one equation, so each (i,j) corresponds to one row of A.
for i in range(N):
    for j in range(N):

        # Convert the current grid point (i,j) into the corresponding matrix row.
        row = idx(i, j)

        # This will collect the coefficient of the current point u_ij.
        # For variable diffusion, the center coefficient is the sum of
        # the diffusion strengths to the left, right, top, and bottom faces.
        center_coeff = 0.0
        
        # Left neighbour contribution
        if j > 0:
            # Diffusion strength at the face between current point (i,j)
            # and its left neighbour (i,j-1).
            # We take the average of k at the two neighbouring points.
            k_left = 0.5 * (k[i, j] + k[i, j - 1])

            # Off-diagonal entry: coupling to the left neighbour.
            # It is negative, similar to the -1 in the constant Poisson stencil.
            A[row, idx(i, j - 1)] = -k_left

            # Add this face contribution to the center coefficient.
            center_coeff += k_left
        else:
            # If there is no left neighbour, this is a boundary face.
            # For zero Dirichlet boundary conditions, the boundary value is known
            # and not included as an unknown, but its face contribution still
            # adds to the center coefficient.
            center_coeff += k[i, j]

        # Right neighbour contribution
        if j < N - 1:
            # Diffusion strength at the face between current point (i,j)
            # and its right neighbour (i,j+1).
            # We take the average of k at the two neighbouring points.
            k_right = 0.5 * (k[i, j] + k[i, j + 1])

            # Off-diagonal entry: coupling to the right neighbour.
            # It is negative, similar to the -1 in the constant Poisson stencil.
            A[row, idx(i, j + 1)] = -k_right

            # Add this face contribution to the center coefficient.
            center_coeff += k_right
        else:
            # If there is no right neighbour, this is a boundary face.
            # For zero Dirichlet boundary conditions, the boundary value is known
            # and not included as an unknown, but its face contribution still
            # adds to the center coefficient.
            center_coeff += k[i, j]

        # Top neighbour contribution
        if i > 0:
            # Diffusion strength at the face between current point (i,j)
            # and its top neighbour (i-1,j).
            # We take the average of k at the two neighbouring points.
            k_top = 0.5 * (k[i, j] + k[i - 1, j])

            # Off-diagonal entry: coupling to the top neighbour.
            # It is negative, similar to the -1 in the constant Poisson stencil.
            A[row, idx(i - 1, j)] = -k_top

            # Add this face contribution to the center coefficient.
            center_coeff += k_top
        else:
            # If there is no top neighbour, this is a boundary face.
            # For zero Dirichlet boundary conditions, the boundary value is known
            # and not included as an unknown, but its face contribution still
            # adds to the center coefficient.
            center_coeff += k[i, j]

        # Bottom neighbour contribution
        if i < N - 1:
            # Diffusion strength at the face between current point (i,j)
            # and its bottom neighbour (i+1,j).
            # We take the average of k at the two neighbouring points.
            k_bottom = 0.5 * (k[i, j] + k[i + 1, j])

            # Off-diagonal entry: coupling to the bottom neighbour.
            # It is negative, similar to the -1 in the constant Poisson stencil.
            A[row, idx(i + 1, j)] = -k_bottom

            # Add this face contribution to the center coefficient.
            center_coeff += k_bottom
        else:
            # If there is no bottom neighbour, this is a boundary face.
            # For zero Dirichlet boundary conditions, the boundary value is known
            # and not included as an unknown, but its face contribution still
            # adds to the center coefficient.
            center_coeff += k[i, j]

        # Diagonal entry for the current point.
        # This is the coefficient of u_ij itself.
        # It is the sum of all left/right/top/bottom face contributions.
        A[row, row] = center_coeff
A = A.tocsr()
A.eliminate_zeros()
# we can't use np.cond on sparse matrices, it throws an error so we first gotta convert it into a dense matrix
A_dense = A.toarray()
A_cond = np.linalg.cond(A_dense)
print("Condition Number of A=",A_cond)
## print("Matrix shape:", A.shape)
## print("Number of non-zero values:", A.nnz)
## print("First 16 diagonal values:")
## print(A.diagonal()[:16])

b = np.ones(n)

x_direct = spsolve(A, b)

residual = b - A @ x_direct
## print("Direct solve residual norm:", np.linalg.norm(residual))

residuals_cg = []

def record_residual(xk):
    residual = b - A @ xk
    residual_norm = np.linalg.norm(residual)
    residuals_cg.append(residual_norm)

x_cg, info_cg = cg(A, b, callback=record_residual)

print("CG info:", info_cg)
print("Number of CG iterations:", len(residuals_cg))
print("Final CG residual:", residuals_cg[-1])

diag_A = A.diagonal()
M = np.diag(diag_A) # M equals A diagonal for Jacobi
M_inv = np.linalg.inv(M)
A_jacobipreconditioned = M_inv @ A_dense
A_jacobipreconditioned_cond = np.linalg.cond(A_jacobipreconditioned)
print("Condition Number of Jacobi-preconditioned A=",A_jacobipreconditioned_cond)

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

x_jacobi, info_jacobi = cg(
    A,
    b,
    M=M_jacobi,
    callback=record_residual_jacobi
)

print("CG + Jacobi info:", info_jacobi)
print("Number of CG + Jacobi iterations:", len(residuals_jacobi))
print("Final CG + Jacobi residual:", residuals_jacobi[-1])

# Plot convergence comparison
iterations_cg = np.arange(1, len(residuals_cg) + 1)
iterations_jacobi = np.arange(1, len(residuals_jacobi) + 1)

plt.figure(figsize=(8, 5))

plt.semilogy(iterations_cg, residuals_cg, marker="o", label="CG")
plt.semilogy(iterations_jacobi, residuals_jacobi, marker="s", linestyle="--", label="CG + Jacobi")

plt.xlabel("Iteration")
plt.ylabel("Residual norm")
plt.title("Variable-Coefficient Diffusion - CG vs CG + Jacobi")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()