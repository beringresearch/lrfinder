import math

import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import numpy as np

from tensorflow.keras.callbacks import LambdaCallback

class LRFinder:
	"""
	Learning rate range test detailed in Cyclical Learning Rates for Training Neural Networks by Leslie N. Smith.
	The learning rate range test is a test that provides valuable information about the optimal learning rate.
	During a pre-training run, the learning rate is increased linearly or exponentially between two boundaries.
	The low initial learning rate allows the network to start converging and as the learning rate is increased it
	will eventually be too large and the network will diverge.
	"""

	def __init__(self, model):
		self.model = model
		self.losses = []
		self.lrs = []
		self.best_loss = 1e9

	def on_batch_end(self, batch, logs):
		# Log the learning rate
		lr = K.get_value(self.model.optimizer.lr)
		self.lrs.append(lr)

		# Log the loss
		loss = logs['loss']
		self.losses.append(loss)

		# Check whether the loss got too large or NaN
		if batch > 5 and (math.isnan(loss) or loss > self.best_loss * 4):
			self.model.stop_training = True
			return

		if loss < self.best_loss:
			self.best_loss = loss

		# Increase the learning rate for the next batch
		lr *= self.lr_mult
		K.set_value(self.model.optimizer.lr, lr)


	def find(self, dataset, start_lr, end_lr, epochs=1, **kw_fit):

		if steps_per_epoch is None:
			try:
				steps_per_epoch = len(dataset)
			except (ValueError, NotImplementedError) as e:
				raise e('`steps_per_epoch=None` is only valid for a'
						' generator based on the '
						'`keras.utils.Sequence`'
						' class. Please specify `steps_per_epoch` '
						'or use the `keras.utils.Sequence` class.')

		self.lr_mult = (float(end_lr) / float(start_lr)) ** (float(1) / float(epochs * steps_per_epoch))
		
		# Save weights into a file
		initial_weights = self.model.get_weights()

		# Remember the original learning rate
		original_lr = K.get_value(self.model.optimizer.lr)

		# Set the initial learning rate
		K.set_value(self.model.optimizer.lr, start_lr)

		callback = LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))

		self.model.fit(dataset,
			epochs=epochs,
			callbacks=[callback],
			**kw_fit)

		# Restore the weights to the state before model fitting
		self.model.set_weights(initial_weights)

		# Restore the original learning rate
		K.set_value(self.model.optimizer.lr, original_lr)


	def plot_loss(self, n_skip_beginning=10, n_skip_end=5, x_scale='log'):
		"""
		Plots the loss.
		Parameters:
			n_skip_beginning - number of batches to skip on the left.
			n_skip_end - number of batches to skip on the right.
		"""

		f, ax = plt.subplots()

		ax.set_ylabel("loss")
		ax.set_xlabel("learning rate (log scale)")
		ax.plot(self.lrs[n_skip_beginning:-n_skip_end], self.losses[n_skip_beginning:-n_skip_end])
		ax.set_xscale(x_scale)
		return(ax)

	def plot_loss_change(self, sma=1, n_skip_beginning=10, n_skip_end=5, y_lim=(-0.01, 0.01)):
		"""
		Plots rate of change of the loss function.
		Parameters:
			sma - number of batches for simple moving average to smooth out the curve.
			n_skip_beginning - number of batches to skip on the left.
			n_skip_end - number of batches to skip on the right.
			y_lim - limits for the y axis.
		"""
		derivatives = self.get_derivatives(sma)[n_skip_beginning:-n_skip_end]
		lrs = self.lrs[n_skip_beginning:-n_skip_end]

		f, ax = plt.subplots()
		ax.set_ylabel("rate of loss change")
		ax.set_xlabel("learning rate (log scale)")
		ax.plot(lrs, derivatives)
		ax.set_xscale('log')
		ax.set_ylim(y_lim)
		
		return(ax)

	def get_derivatives(self, sma):
		assert sma >= 1
		derivatives = [0] * sma
		for i in range(sma, len(self.lrs)):
			derivatives.append((self.losses[i] - self.losses[i - sma]) / sma)
		return derivatives

	def get_best_lr(self, sma, n_skip_beginning=10, n_skip_end=5):
		derivatives = self.get_derivatives(sma)
		best_der_idx = np.argmin(derivatives[n_skip_beginning:-n_skip_end])
		return self.lrs[n_skip_beginning:-n_skip_end][best_der_idx]