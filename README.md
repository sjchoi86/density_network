# Density modeling with TensorFlow

We implement two density modeling methods:
1. Unsupervised Gaussian mixture model (GMM): [notebook](https://github.com/sjchoi86/density_network/blob/master/src/demo_fit_MoG.ipynb)
2. Mixture density network (MDN): [notebook](https://github.com/sjchoi86/density_network/blob/master/src/demo_mdn_reg.ipynb)

We use `tf.contrib.distributions` to implement the computational graphs which supports `Categorical`, `MultivariateNormalDiag`, `Normal`, and the most important `Mixture`. 

`tf.contrib.distributions.Mixture` [api](https://www.tensorflow.org/api_docs/python/tf/contrib/distributions/Mixture) provides a number of useful apis such as `cdf`, `cross_entropy`, `entropy_lower_bound`, `kl_divergence`, `log_prob`, `prob`, `quantile`, and `sample`. 

#### Contact: Sungjoon Choi (sungjoon.s.choi@gmail.com)
