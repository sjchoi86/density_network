import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf

def plot_1d_graphs(_x1_list,_y1_list,_linestyles1,_markers1,_colors1,
                   _x2_list=None,_y2_list=None,_linestyles2=None,_markers2=None,_colors2=None,
                   _nR=1,_nC=10,_figsize=(15,2),
                   _title=None,_titles=None,_tfs=15,
                   _wspace=0.05,_hspace=0.05):
    nr,nc = _nR,_nC
    fig = plt.figure(figsize=_figsize)
    if _title is not None:
        fig.suptitle(_title, size=15)
    gs  = gridspec.GridSpec(nr,nc)
    gs.update(wspace=_wspace, hspace=_hspace)
    for i in range(_nR*_nC):
        ax = plt.subplot(gs[i])
        plt.plot(_x1_list[i],_y1_list[i],
                 linestyle=_linestyles1[i],marker=_markers1[i],color=_colors1[i])
        if _x2_list is not None:
            plt.plot(_x2_list[i],_y2_list[i],
                     linestyle=_linestyles2[i],marker=_markers2[i],color=_colors2[i])
        if _titles is not None:
            plt.title(_titles[i],size=_tfs)
    plt.show() 
    
    
def gpu_sess(): 
    config = tf.ConfigProto(); 
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    return sess    


class nzr(object):
    def __init__(self,_rawdata,_eps=1e-8):
        self.rawdata = _rawdata
        self.eps     = _eps
        self.mu      = np.mean(self.rawdata,axis=0)
        self.std     = np.std(self.rawdata,axis=0)
        self.nzd_data = self.get_nzdval(self.rawdata)
        self.org_data = self.get_orgval(self.nzd_data)
        self.maxerr = np.max(self.rawdata-self.org_data)
    def get_nzdval(self,_data):
        _n = _data.shape[0]
        _nzddata = (_data - np.tile(self.mu,(_n,1))) / np.tile(self.std+self.eps,(_n,1))
        return _nzddata
    def get_orgval(self,_data):
        _n = _data.shape[0]
        _orgdata = _data*np.tile(self.std+self.eps,(_n,1))+np.tile(self.mu,(_n,1))
        return _orgdata