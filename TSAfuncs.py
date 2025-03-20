import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import logm,expm, sqrtm
from scipy.stats import zscore
import pandas as pd

class FC:
    def __init__(self,fc, subject, state = 'rest', session = 1):
        self.fc = fc
        self.subject = subject
        self.state = state
        self.session = session
        self.size = np.shape(fc)
        self.tsfc = dict()

    def tangent_space_projection(self, reg, refinv = None):
        if refinv is None:
            #refinv = (1+1/reg)*np.identity(self.size[0])
            #if the reference is not given, assume identity. In that case, the tsfc calculation can be simplified
            tsfc = logm(1/(1+reg)*(self.fc + reg*np.identity(self.size[0])))#tsa.tangent_space(self.fc, reg, ref_FC = reference)
        else:
            tsfc = logm(refinv @ (self.fc + reg*np.identity(self.size[0])) @ refinv)#tsa.tangent_space(self.fc, reg, ref_FC = reference)
        self.tsfc[reg] = tsfc
        return tsfc


def get_regularization_flow(FCs, regvals):
    #for each FC get the regularization flow, return vectorized version
    regflow = list()
    df = pd.DataFrame(columns = ['subject_cat', 'subject', 'state_cat', 'state', 'reg', 'session'])
    subject = list()
    state = list()
    session = list()
    reg = list()
    idx = np.triu_indices(FCs[0].size[0],1)
    for fc in FCs:
        for key in regvals:
            regflow.append(zscore(fc.tsfc[key][idx]))
            subject.append(fc.subject)
            state.append(fc.state)
            session.append(fc.session)
            reg.append(key)
    

    regflow = zscore(regflow, axis = 0)
    df['session'] = session
    df['reg'] = reg

    #subject will often have unique ids which are annoying to deal with, here convert to code
    df['subject_cat'] = subject
    df['subject_cat'] = df['subject_cat'].astype('category')  
    df['subject'] = df['subject_cat'].cat.codes

    #states are often catagorical, here we want to convert them to numeric
    df['state_cat'] = state
    df['state_cat'] = df['state_cat'].astype('category')
    df['state'] = df['state_cat'].cat.codes

    return regflow, df


def get_join_labels(df):
    
    subjects = df['subject'].values
    sessions = df['session'].values
    states = df['state'].values
    joint_labels = list(zip(subjects, sessions,states))
    unique_joint_labels = list(set(joint_labels))
    return joint_labels, unique_joint_labels


def plot_regflow_pca(Y, c1, df, labels):
    #input: 
    # Y is the pca scores
    # c1 is principle component
    # df is the regflow dataframe
    # labels is the labels you want to colour
    
    c2 = c1 + 1
    unique_labels = list(set(labels))

    #first we need to make joint labels as this is what will let us find the curves
    joint_labels, unique_joint_labels = get_join_labels(df)


    label_colors = plt.cm.jet(np.linspace(0,1,len(unique_labels)))# Initialize holder for trajectories
    color_dict = dict()
    for i,val in enumerate(unique_labels):
        color_dict[val] = label_colors[i]

 
    fig,ax = plt.subplots(1)
    for label in unique_joint_labels:
        idx = [i for i,x in enumerate(joint_labels) if x == label]

        x = Y[idx, c1]
        y = Y[idx, c2]
        ax.plot(x, y, alpha=0.7, color = color_dict[labels[idx[0]]])


def get_reference_inv(ref_FC, reg = 1):
    ref = [logm(f + reg*np.identity(np.shape(f)[0])) for f in ref_FC];
    ref = expm(np.mean(ref, axis = 0))
    invref = sqrtm(np.linalg.inv(ref))
    
    return invref

def calc_tangentspace_FCs(test_FC, invref, reg = 1):
    TS = [logm(invref @ (f + reg*np.identity(np.shape(f)[0])) @ invref) for f in test_FC ];
    TS = np.array([f[np.triu_indices(np.shape(test_FC[0])[0],k = 1)] for f in TS])
    return TS

def tangent_space(fc, reg, ref_FC = None):
    
    if ref_FC is None:
        refinv = reg*np.identity(np.shape(fc[0])[0])
    else:
        refinv = get_reference_inv(fc, reg)

    
    TS = calc_tangentspace_FCs(fc, refinv, reg)
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
