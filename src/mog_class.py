import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf

tfd = tf.contrib.distributions
class MoG_class(object):
    def __init__(self,_x_dim=2,_k=5,_sess=None):
        self.x_dim = _x_dim 
        self.k = _k # number of mixture
        self.sess = _sess
        self._build_graph()
        # Initialize parameters 
        self.sess.run(tf.global_variables_initializer())
    def _build_graph(self):
        # Placeholder
        self.x = tf.placeholder(dtype=tf.float32,shape=(None,self.x_dim),
                                name='x') # [N x x_dim]
        self.n = tf.shape(self.x)[0] # number of batch
        # Define pi, mu ,and variance
        pi_speed = 100
        pi_initializer = tf.truncated_normal_initializer(stddev=np.sqrt(0.1/pi_speed)) # make each mu to follow Gaussian with var=0.1
        self.pi_mtx = tf.get_variable(name='pi_mtx',shape=(pi_speed,self.k),
                        dtype=tf.float32,initializer=pi_initializer)
        self.pi = tf.reduce_sum(self.pi_mtx,axis=0,name='pi') # [k]
        self.pi = tf.nn.softmax(self.pi) # [k] sum to one
        mu_speed = 100
        mu_initializer = tf.truncated_normal_initializer(stddev=np.sqrt(1.0/mu_speed)) # make each mu to follow unit Gaussian
        self.mu_mtx = tf.get_variable(name='mu_mtx',shape=(mu_speed,self.x_dim,self.k),
                        dtype=tf.float32,initializer=mu_initializer)
        self.mu = tf.reduce_sum(self.mu_mtx,axis=0,name='mu') # [x_dim x k]
        logvar_speed = 100
        # logvar_initializer = tf.truncated_normal_initializer(stddev=0.01)
        logvar_initializer = tf.constant_initializer(value=-3.0/logvar_speed)
        self.logvar_mtx = tf.get_variable(name='logvar_mtx',
                            shape=(logvar_speed,self.x_dim,self.k),
                            dtype=tf.float32,initializer=logvar_initializer) # [N x x_dim]
        self.logvar = tf.reduce_sum(self.logvar_mtx,axis=0,name='logvar') # [x_dim x k]
        self.var = tf.exp(self.logvar) # [x_dim x k]
        
        # Sampler 
        self.n_sample = tf.placeholder(dtype=tf.int32,name='N_sample')
        cat = tfd.Categorical(probs=self.pi)
        components = [tfd.MultivariateNormalDiag(loc=self.mu[:,i],
                          scale_diag=tf.sqrt(self.var[:,i])) for i in range(self.k)]
        self.tfd_mog = tfd.Mixture(cat=cat,components=components)
        self.x_sample = self.tfd_mog.sample(self.n_sample) # [n x d]
        
        # Log likelihood
        self.log_liks = self.tfd_mog.log_prob(self.x)
        self.log_lik = tf.reduce_mean(self.log_liks)
        self.cost = -self.log_lik
        
        # Optimizer
        self.optm = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.cost)
        
    def plot_samples(self,_n_sample=1000,_x_train=None,_title_str=None):
        x_sample = self.sess.run(self.x_sample,feed_dict={self.n_sample:_n_sample})
        pi,mu,var = self.sess.run([self.pi,self.mu,self.var]) # [k], [x_dim x k], [x_dim x k]
        dim = x_sample.shape[1] # dimension 
        plt.figure(figsize=(8,6));plt.grid(True)
        if _x_train is not None:
            plt.plot(_x_train[:,0],_x_train[:,1],'r.') # plot training data
        plt.plot(x_sample[:,0],x_sample[:,1],'bx') # plot samples 
        if _title_str is not None:
            plt.title(_title_str,fontsize=15)
        plt.axis('equal'); plt.show()
        


class MoG_indep_class(object):
    def __init__(self,_x_dim=2,_k=5,_sess=None):
        self.x_dim = _x_dim 
        self.k = _k # number of mixture
        self.sess = _sess
        self._build_graph()
        # Initialize parameters 
        self.sess.run(tf.global_variables_initializer())
    def _build_graph(self):
        # Placeholder
        self.x = tf.placeholder(dtype=tf.float32,shape=(None,self.x_dim),
                                name='x') # [N x x_dim]
        self.n = tf.shape(self.x)[0] # number of batch
        # Define pi, mu ,and variance
        pi_speed = 100
        pi_initializer = tf.truncated_normal_initializer(stddev=np.sqrt(0.1/pi_speed)) # make each mu to follow Gaussian with var=0.1
        self.pi_mtx = tf.get_variable(name='pi_mtx',shape=(pi_speed,self.x_dim,self.k),
                        dtype=tf.float32,initializer=pi_initializer)
        self.pi = tf.reduce_sum(self.pi_mtx,axis=0,name='pi') # [x_dim x k]
        self.pi = tf.nn.softmax(self.pi) # [k] sum to one
        mu_speed = 100
        mu_initializer = tf.truncated_normal_initializer(stddev=np.sqrt(1.0/mu_speed)) # make each mu to follow unit Gaussian
        self.mu_mtx = tf.get_variable(name='mu_mtx',shape=(mu_speed,self.x_dim,self.k),
                        dtype=tf.float32,initializer=mu_initializer)
        self.mu = tf.reduce_sum(self.mu_mtx,axis=0,name='mu') # [x_dim x k]
        logvar_speed = 100
        # logvar_initializer = tf.truncated_normal_initializer(stddev=0.01)
        logvar_initializer = tf.constant_initializer(value=-3.0/logvar_speed)
        self.logvar_mtx = tf.get_variable(name='logvar_mtx',
                            shape=(logvar_speed,self.x_dim,self.k),
                            dtype=tf.float32,initializer=logvar_initializer) 
        self.logvar = tf.reduce_sum(self.logvar_mtx,axis=0,name='logvar') # [x_dim x k]
        self.var = tf.exp(self.logvar) # [x_dim x k]
        
        # Sampler 
        self.n_sample = tf.placeholder(dtype=tf.int32,name='N_sample')
        cat = tfd.Categorical(probs=self.pi)
        components = [tfd.Normal(loc=self.mu[:,i],
                          scale=tf.sqrt(self.var[:,i])) for i in range(self.k)]
        self.tfd_mog = tfd.Mixture(cat=cat,components=components)
        self.x_sample = self.tfd_mog.sample(self.n_sample) # [n x d]
        
        # Log likelihood
        self.log_liks = self.tfd_mog.log_prob(self.x)
        self.log_lik = tf.reduce_mean(self.log_liks)
        self.cost = -self.log_lik
        
        # Optimizer
        self.optm = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.cost)
        
    def plot_samples(self,_n_sample=1000,_x_train=None,_title_str=None,_fontsize=15,
                     _figsize=(12,5),_wspace=0.1,_hspace=0.05):
        
        x_sample = self.sess.run(self.x_sample,feed_dict={self.n_sample:_n_sample}) # [n x d]
        pi,mu,var = self.sess.run([self.pi,self.mu,self.var]) # [d x k], [d x k], [d x k]
        dim = x_sample.shape[1] # dimension 
        nr,nc = 1,dim
        if nc>2: nc=2 # Upper limit on the number of columns
        gs = gridspec.GridSpec(nr,nc)
        gs.update(wspace=_wspace, hspace=_hspace)
        fig = plt.figure(figsize=_figsize)
        if _title_str is not None:
            fig.suptitle(_title_str, size=_fontsize)
        for i in range(nr*nc): # per each dimension            
            ax = plt.subplot(gs[i])
            # Plot GMM
            x_min,x_max = x_sample[:,i].min(),x_sample[:,i].max()
            xs = np.linspace(x_min,x_max,1000)
            curr_pi = pi[i,:] # [k]
            curr_mu = mu[i,:] # [k]
            curr_var = var[i,:] # [k]
            
            def pdf_Gaussian(_in,_mu,_var):
                prob = 1/(np.sqrt(2*np.pi*_var))*np.exp(-0.5/_var*(_in-_mu)**2)
                return prob
            def pdf_GMM(_ins,_pis,_mus,_vars):
                probs = np.zeros_like(_ins)
                for idx,_in in enumerate(_ins):
                    prob = 0
                    for j in range(self.k): # for each mixture
                        prob += _pis[j]*pdf_Gaussian(_in,_mus[j],_vars[j])
                    probs[idx] = prob
                return probs
            probs = pdf_GMM(xs,curr_pi,curr_mu,curr_var)
            # Plot GMM pdf
            plt.plot(xs,probs,'k-')
            # Plot histogram
            x_train_i = _x_train[:,i]
            x_sample_i = x_sample[:,i]
            plt.hist([x_train_i,x_sample_i],bins=100,
                     color=['r','b'],label=['train','sample'],density=True)
            plt.title('[%d]-th dimension'%(i+1),fontsize=13)
        plt.show()        
        
        
        