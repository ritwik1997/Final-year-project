from sklearn.manifold import Isomap, LocallyLinearEmbedding, SpectralEmbedding, MDS, TSNE
from sklearn.mainfold import load_digits
import numpy as np 
import matplotlib as plt

X, _ = load_digits(return_X_y = True)
print X.shape

class ManifoldLearning(object):
	"""docstring for ManifoldLearning"""
	def __init__(self, dataset):
		super(ManifoldLearning, self).__init__()
		self.dataset = X

	def doIsomap(self, dataset):
		embed = Isomap(n_components = 2)
		X_transformed = embed.fit_transform(dataset)
		return X_transformed

	def doLocallyLinearEmbedding(self, dataset):
		embed = LocallyLinearEmbedding(n_components = 2)
		X_transformed = embed.fit_transform(dataset)
		return X_transformed

	def doSpectralEmbedding(self, dataset):
		embed = SpectralEmbedding(n_components = 2)
		X_transformed = embed.fit_transform(dataset)
		return X_transformed

	def doMDS(self, dataset):
		embed = MDS(n_components = 2)
		X_transformed = embed.fit_transform(dataset)
		return X_transformed

	def doTSNE(self, dataset):
		embed = TSNE(n_components = 2)
		X_transformed = embed.fit_transform(dataset)
		return X_transformed
