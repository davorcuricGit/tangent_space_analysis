#import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import logm,expm, sqrtm
from scipy.stats import zscore

def get_reference_inv(ref_FC, reg = 1):
    ref = [logm(f + reg*np.identity(np.shape(f)[0])) for f in ref_FC];
    ref = expm(np.mean(ref, axis = 0))
    invref = sqrtm(np.linalg.inv(ref))
    
    return invref

def calc_tangentspace_FCs(test_FC, invref, reg = 1):
    TS = [logm(invref @ (f + reg*np.identity(np.shape(f)[0])) @ invref) for f in test_FC ];
    TS = np.array([f[np.triu_indices(np.shape(test_FC[0])[0],k = 1)] for f in TS])
    return TS

def tangent_space(test_FC, reg, ref_FC = None):
    
    if ref_FC is None:
        refinv = reg*np.identity(np.shape(test_FC[0])[0])
    else:
        refinv = get_reference_inv(ref_FC, reg)

    
    TS = calc_tangentspace_FCs(test_FC, refinv, reg)
    return TS

def get_ref_and_test_FC(FCs, rest_df, refidx, subject = None):
    if subject is None:
        ref_df = rest_df[((rest_df['session'] == refidx) & (rest_df['task'] == 'Rest'))]#  & (rest_df['subject'] == subject)]
        test_df = rest_df[((rest_df['session'] != refidx) | (rest_df['task'] != 'Rest'))]# & (rest_df['subject'] == subject)]#rest_df[(rest_df['session'] == 3) | (rest_df['session'] == 4 )]
    else:
        ref_df = rest_df[((rest_df['session'] == refidx) & (rest_df['task'] == 'Rest'))  & (rest_df['subject'] == subject)]
        test_df = rest_df[((rest_df['session'] != refidx) | (rest_df['task'] != 'Rest')) & (rest_df['subject'] == subject)]#rest_df[(rest_df['session'] == 3) | (rest_df['session'] == 4 )]

    refindex = ref_df.index.values
    testindex = test_df.index.values

    ref_FC = [FCs[i] for i in refindex]
    test_FC = [FCs[i] for i in testindex]
    return ref_FC, ref_df, test_FC, test_df

def get_regularization_flow(test_FC, test_df, start = -2, stop = 4, step = 5, ref_FC = None):
    if ref_FC is not None:
        ref_FC = np.array(ref_FC)
    reglist = np.linspace(start,stop,step)
    
    regflow = np.array([])
    subjects = np.tile(test_df['subject'].values, len(reglist))
    sessions = np.tile(test_df['session'].values, len(reglist))
    tasks = np.tile(test_df['task'].values, len(reglist))
    for i,val in enumerate(reglist):
        reg = 10**val
        
        TS = tangent_space(test_FC, reg, ref_FC); 
        #print(np.shape(TS))
        #TS = zscore(TS, axis = 0)
        
        if len(regflow) == 0:
            regflow = TS
        else:
            regflow = np.concatenate([regflow, TS])
    
    
    regflow = zscore(regflow, axis = 1)
    
    return regflow, subjects, sessions, tasks

def plot_pca_projections(Y, subjects,sessions,tasks, c1, c2, ax = None, fig = None, ):
    
    test = list(zip(subjects, sessions,tasks))
    uqtest = list(set(test))
    
    colors = plt.cm.jet(np.linspace(0,1,11))# Initialize holder for trajectories
    markers = {'Rest': 'o', 'Motor': 's', 'Memory': 'd', 'Mixed':'^'}
    linestyles = {'Rest': '-', 'Motor': '--', 'Memory': '-.', 'Mixed':':'}
    colordict = {'Rest': colors[0], 'Motor': colors[4], 'Memory': colors[7], 'Mixed': colors[10]}
    
    
    if ax is None:
        fig,ax = plt.subplots(2)
    
    for uq in uqtest:
        idx = [i for i,x in enumerate(test) if x == uq]
        
        
        x = Y[idx, c1]
        y = Y[idx, c2]
        ax[0].plot(x, y, alpha=0.7, color = colordict[uq[2]], linestyle = linestyles[uq[2]])
        ax[0].plot(x[-1], y[-1], alpha=0.7, color = colordict[uq[2]], marker = markers[uq[2]])
        #ax[0].plot(Y[idx[0],0], Y[idx[0],1], alpha=0.7, color = colors[uq[0]], marker = markers[uq[2]])
    
        x = Y[idx, c1]
        y = Y[idx, c2]
        ax[1].plot(x, y, alpha=0.7, color = colors[uq[0]], linestyle = linestyles[uq[2]])
        ax[1].plot(x[-1], y[-1], alpha=0.7, color = colors[uq[0]], marker = markers[uq[2]])
        
    fig.set_size_inches(9, 16)
    return fig,ax