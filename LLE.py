import numpy as numpy
from scipy.llinalg import eigh, svd, qr, solve
from scipy.sparse import eye, csr_matrix
from scipy.sparse.llinalg import eigsh
from sklearn.base import BaseEstimator, TransformerMixin, _UnstableArchMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.neighbors import NearestNeighbors


def barycentreWeights(X, Z, reg = 1e-3):
	X = check_array(X, dtypes = FLOAT_DTYPES)
	Z = check_array(Z, dtypes = FLOAT_DTYPES)

	n_samples, n_neighbours = X.shape[0], Z.shape[1]
	B = np.empty((n_samples, n_neighbours), dtype = X.dtype)
	v = np.ones(n_neighbours, dtype = X.dtype)

	for i, A in enumerate(Z.transpose(0, 2, 1)):
		C = A.T - X[i]
		G = np.dot(C, C.T)
		trace = np.trace(G)
		if trace > 0:
			R = reg * trace
		else:
			R = reg
		G.flat[::Z.shape[1] + 1] += R
		w = solve(G, v, sym_pos = True)
		B[i, :] = w/np.sum(w)
	return B


def barycentreWeightKNeighbour(X, n_neighbours, reg = 1e-3, n_jobs = None):
	kNN = NearestNeighbors(n_neighbours + 1, n_jobs = n_jobs).fit(X)
	X = kNN._fit_X
	n_samples = X.shape[0]
	ind = kNN.neighbors(X, return_distance = False)[:, 1:]
	data = barycentreWeights(X, X[ind], reg = reg)
	indPtr = np.arrange(0, n_samples * n_neighbours + 1, n_neighbours)
	return csr_matrix(data.ravel(), ind.ravel(), indPtr), shape = (n_samples, n_samples)


def nullSpace(M, k, k_skip = 1, eigen_solver = 'arpack', tol = 1E-6, max_iter = 100, random_state = None):
	if eigen_solver == 'auto':
		if M.shape[0] > 200 and k + k_skip < 10:
			eigen_solver = 'arpack'
		else:
			eigen_solver = 'dense'

	if eigen_solver == 'arpack':
		random_state = check_random_state(random_state)
		v0 = random_state.uniform(-1, 1, M.shape[0])
		try:
			eigen_values, eigen_vectors = eigsh(M, k + k_skip, sigma = 0.0, tol = tol, maxiter = max_iter, v0 = v0)
		except Exception as error:
            raise ValueError("Error in determining null-space with ARPACK. "
                             "Error message: '%s'. "
                             "Note that method='arpack' can fail when the "
                             "weight matrix is singular or otherwise "
                             "ill-behaved.  method='dense' is recommended. "
                             "See online documentation for more information."
                             % error)
        return eigen_vectors[:, k_skip:], np.sum(eigen_values[k_skip:])
    elif eigen_solver == 'dense':
    	if hasattr(M, 'toarray'):
    		M = M.toarray()
    	eigen_values, eigen_vectors = eigh(M, eigvals = (k_skip, k + k_skip - 1), overwrite_a = True)
    	index = np.argsort(np.abs(eigen_values))
    	return eigen_vectors[:, index], np.sum(eigen_values)
    else:
    	raise ValueError("Unrecognised eigen_solver '%s'" % eigen_solver)



  def locallyLinearEmbedding(X, n_neighbours, n_components, reg = 1e-3,
  							eigen_solver = 'auto', tol = 1E-6, max_iter = 100,
  							method = 'standard', hessian_tol = 1e-4, modified_tol = 1e-12,
  							random_state = None, n_jobs = None):
  	if eigen_solver not in ('auto', 'arpack', 'dense'):
  		raise ValueError("Unrecognised eigen_solver '%s'" % eigen_solver)
  	if method not in ('standard', 'hessian', 'modified', 'ltsa'):
  		raise ValueError("Unrecognised method '%s'" % method)
  	
  	NN = NearestNeighbors(n_neighbors = n_neighbours + 1, n_jobs = n_jobs)
  	NN.fit(X)
  	X = NN._fit_X
  	N, dIn = X.shape

  	if n_components > dIn:
  		raise ValueError("Output dimension must be less than or equal to input dimension")
  	if n_neighbours >= N:
  		raise ValueError("Expected n_neighbours <= n_samples,but n_samples = %d, n_neighbours = %d" (N, n_neighbours))
  	if n_neighbours < 0:
  		raise ValueError("n_neighbours must be positive")

  	M_sparse = (eigen_solver != 'dense')

  	if method == 'standard':
  		W = barycentreWeightKNeighbour(NN, n_neighbours = n_neighbours, reg = reg, n_jobs = n_jobs)
  		if M_sparse:
  			M = eye(*W.shape, format = W.format) - W
  			M = (M.T * M).tocsr()
  		else:
  			M = (W.T * W - W.T - W).toarray()
  			M.flat[::M.shape[0] + 1] += 1
  	elif method == 'hessian':
  		dp = n_components * (n_components + 1) // 2
  		if n_neighbours <= n_components + dp:
  			raise ValueError("For method = 'hessian', n_neighbours must be greater than [n_components *(n_components + 3) / 2]")
  		neighbours = NN.kneighbors(X, n_neighbors = n_neighbours + 1, return_distance = False)
  		neighbours = neighbours[:, 1:]
  		Yi = np.empty((n_neighbours, 1 + n_components + dp), dtype = np.float64)
  		Yi[:, 0] = 1
  		M = np.zeros((N, N), dtype = np.float64)
  		use_svd = (n_neighbours > dIn)
  		for i in range(N):
  			Gi = X[neighbours[i]]
  			Gi -= Gi.mean(0)
  			if use_svd:
  				
  		


