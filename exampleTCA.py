import tensortools as tt
import numpy as np
import matplotlib.pyplot as plt

# PACKAGE CONTENTS
#     data (package)
#     diagnostics
#     ensemble
#     operations
#     optimize (package)
#     tensors
#     utils
#     visualization
##
help(tt.diagnostics)
##
help(tt.ensemble)
##
help(tt.operations)
##
help(tt.tensors)
##
help(tt.optimize)
##
help(tt.optimize.optim_utils)
##
help(tt.visualization)
##
# Make synthetic dataset.
I, J, K, R = 25, 50, 75, 4  # dimensions and rank
X = tt.randn_ktensor((I, J, K), rank=R).full()
X += np.random.randn(I, J, K)  # add noise

##
# Fit CP tensor decomposition (two times).
U = tt.cp_als(X, rank=R, verbose=True)
V = tt.cp_als(X, rank=R, verbose=True)

##
# Compare the low-dimensional factors from the two fits.
fig, _, _ = tt.plot_factors(U.factors)
tt.plot_factors(V.factors, fig=fig)
##
# Align the two fits and print a similarity score.
sim = tt.kruskal_align(U.factors, V.factors, permute_U=True, permute_V=True)
print(sim)

##
# Plot the results again to see alignment.
fig, ax, po = tt.plot_factors(U.factors)
tt.plot_factors(V.factors, fig=fig)

# Show plots.
plt.show()

