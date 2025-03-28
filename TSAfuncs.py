import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import logm,expm, sqrtm
from scipy.stats import zscore, mode
import pandas as pd
import pdb

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

    def get_subject_fcs(FCs):
        subject_fcs = dict()
        for fc in FCs:
            if fc.subject not in subject_fcs:
                subject_fcs[fc.subject] = dict()
            if fc.state not in subject_fcs[fc.subject]:
                subject_fcs[fc.subject][fc.state] = dict()
            if fc.session not in subject_fcs[fc.subject][fc.state]:
                subject_fcs[fc.subject][fc.state][fc.session] = fc
        return subject_fcs


class test_retest:
    #goal: randomly split FC sessions into two objects, one for training (datbase) and one for testing (target)
    #input is a list of fc objects in
    #output will be two lists, database and target
    #database will be one random session from each subject
    #target will be all the other sessions from the subject
    def __init__(self, FCs):
        self.database = list()
        self.target = list()
        self.subject_fcs = FC.get_subject_fcs(FCs)
        self.split_sessions()


    def split_sessions(self):
        for subject in self.subject_fcs:
            for state in self.subject_fcs[subject]:  # Iterate over all second-layer keys
                sessions = list(self.subject_fcs[subject][state].keys())
                if len(sessions) > 1:
                    train_session = np.random.choice(sessions, 1)[0]
                    test_sessions = list(set(sessions) - set([train_session]))
                    for session in test_sessions:
                        self.database.append(self.subject_fcs[subject][state][train_session])
                        self.target.append(self.subject_fcs[subject][state][session])
    
    def ts_test_retest(self, **kwargs):
        #compute test restest using tangent space FCs across multiple regularization valyes.
        #refinv specifies the method with which the inverse of the reference point in the tangent space. 
        # If None, the identity is used, if 'logm' then the log mean of the database is used

        regvals = kwargs.get('regvals', [4])
        refinv = kwargs.get('refinv', None)

        if refinv is not None and refinv != 'logm':
            print('refinv must be None or logm')
        else:
            for reg in regvals:
                #ref = np.zeros(FCs[0].fc.shape)
                if refinv == 'logm':
                    refinv = np.linalg.inv(np.mean(np.array([logm(x.fc + reg) for x in self.database]), axis = 0))
                [x.tangent_space_projection(reg = reg, refinv = refinv) for x in self.database]
                [x.tangent_space_projection(reg = reg, refinv = refinv) for x in self.target ]
        
            regflow_db, df_db = get_regularization_flow(self.database, regvals)
            regflow_tg, df_tg = get_regularization_flow(self.target, regvals)

            df_tg = self.get_closest_database_subject(regflow_db, regflow_tg, df_db, df_tg, regvals)
            score = self.get_test_retest_score(df_tg)
            return score
    

    def classic_test_retest(self, regvals =[4]):
        #this computes the classif test retest using FCs instead of TSFCs.
        #However, FCs are equivalent to TSFCs with large regularization when the identity is used as the reference.
        #so we can reuse the ts_test_retest code
        score = self.ts_test_retest(np.array([10000]), refinv = None)
        return score


    #find which database subject is closest to which target subject
    def get_closest_database_subject(self,regflow_db, regflow_tg, df_db, df_tg, regvals):
        df_tg['closest subject'] = np.nan
        df_tg['closest state'] = np.nan

        for r in regvals:
            idx_tg = np.where(df_tg['reg'].values == r)[0]
            idx_db = np.where(df_db['reg'].values == r)[0]
            
            sub_regflow_db = regflow_db[idx_db][:]
            sub_regflow_tg = regflow_tg[idx_tg][:]
            
            for m in range(np.shape(sub_regflow_tg)[0]):
                D = np.zeros((len(sub_regflow_db),))
                for i, db in enumerate(sub_regflow_db):
                    D[i] = 1 - np.mean(np.multiply(zscore(sub_regflow_tg[m]), zscore(db)))
                df_tg.loc[idx_tg[m], 'closest subject'] = df_db['subject'].values[idx_db[np.argmin(D)]]
                df_tg.loc[idx_tg[m], 'closest state'] = df_db['state'].values[idx_db[np.argmin(D)]]
                
        return df_tg
    

    def get_test_retest_score(self,df_tg):

        joint_labels, unique_joint_labels = get_join_labels(df_tg)
        
        
        score = dict({'subject': 0, 'state': 0, 'condition': dict()})
        for label in unique_joint_labels:
            idx = [i for i,x in enumerate(joint_labels) if x == label]
            
            #majority rule, closest is the one with the most votes
            #the score for the subject discrimination
            subject_guess = mode(df_tg.loc[idx, 'closest subject'].values)[0]
            if subject_guess == label[0]:
                score['subject'] = score['subject'] + 1/len(unique_joint_labels)

            #the score for the state discrimination
            state_guess = mode(df_tg.loc[idx, 'closest state'].values)[0]
            if state_guess == label[2]:
                score['state'] = score['state'] + 1/len(unique_joint_labels)    

            #the score for the subject discrimination conditional on the condition
            if label[2] not in score['condition'].keys():
                score['condition'][label[2]] = 0
            else:
                if subject_guess == label[0]:
                    score['condition'][label[2]] = score['condition'][label[2]] + 1/len(df_tg[df_tg['state'] == label[2]])            
        return score


def vectorize_FC(fc):
    idx = np.triu_indices(np.shape(fc)[0],1)
    return fc[idx]

def get_regularization_flow(FCs, regvals):
    #for each FC get the regularization flow, return vectorized version
    regflow = list()
    df = pd.DataFrame(columns = ['subject_cat', 'subject', 'state_cat', 'state', 'reg', 'session'])
    subject = list()
    state = list()
    session = list()
    reg = list()
    #idx = np.triu_indices(FCs[0].size[0],1)
    for fc in FCs:
        for key in regvals:
            regflow.append(zscore(vectorize_FC(fc.tsfc[key])))#fc.tsfc[key][idx]))
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
