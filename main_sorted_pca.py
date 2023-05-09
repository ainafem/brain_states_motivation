import pandas as pd
import numpy as np
np.random.seed(seed=42)
import gudhi as gd
import scipy.signal as spsg
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import gudhi.representations as gdr
from scipy.io import loadmat
from sklearn.decomposition import PCA
import pymp


def filter_ts(data, freq_band):
    n_order = 3
    sampling_freq = 500. # sampling rate

    if freq_band=='alpha':
        low_f = 8./sampling_freq
        high_f = 15./sampling_freq
    elif freq_band=='beta':   
        # beta
        low_f = 15./sampling_freq
        high_f = 32./sampling_freq
    elif freq_band=='gamma':
        # gamma
        low_f = 32./sampling_freq
        high_f = 80./sampling_freq
    else:
        return data


    b,a = spsg.iirfilter(n_order, [low_f,high_f], btype='bandpass', ftype='butter')
    return spsg.filtfilt(b, a, data, axis=1)

def get_persistence(block, dim):
    Rips_complex_sample = gd.RipsComplex(block)
    Rips_simplex_tree_sample = Rips_complex_sample.create_simplex_tree(max_dimension=2)
    persistence = Rips_simplex_tree_sample.persistence()
    if dim == 0:
        return [np.array(persistence[i][1]) for i in range(len(persistence)) if persistence[i][0] == 0 and persistence[i][1][1] != float('inf')]
    elif dim == 1:
        return [np.array(persistence[i][1]) for i in range(len(persistence)) if persistence[i][0] == 1 and persistence[i][1][1] != float('inf')]

    
def get_candidates_sil(mtv0, mtv1, mtv2, dim, p = 1):
    persistence_0 = get_persistence(mtv0, dim)
    persistence_1 = get_persistence(mtv1, dim)
    persistence_2 = get_persistence(mtv2, dim)
    
    sil = gdr.Silhouette(weight = lambda x: np.power(x[1]-x[0], p), resolution=1000)
    res0 = sil.fit_transform([np.array(persistence_0)])
    res1 = sil.fit_transform([np.array(persistence_1)])
    res2 = sil.fit_transform([np.array(persistence_2)])
    
    return res0[0], res1[0], res2[0]

def get_landscapes(mtv0, mtv1, mtv2, dim):    
    persistence_0 = get_persistence(mtv0, dim)
    persistence_1 = get_persistence(mtv1, dim)
    persistence_2 = get_persistence(mtv2, dim)
    
    lands = gdr.Landscape(num_landscapes=5, resolution=1000)
    res0 = lands.fit_transform([np.array(persistence_0)])
    res1 = lands.fit_transform([np.array(persistence_1)])
    res2 = lands.fit_transform([np.array(persistence_2)])
    
    return res0, res1, res2

def get_distances(sample, mtv0, mtv1, mtv2, res0, res1, res2, dim, p = 1):
    pers0 = get_persistence(np.vstack([mtv0, sample]), dim)
    pers1 = get_persistence(np.vstack([mtv1, sample]), dim)
    pers2 = get_persistence(np.vstack([mtv2, sample]), dim)
    
    sil = gdr.Silhouette(weight = lambda x: np.power(x[1]-x[0], p), resolution=1000)
    res0s = sil.fit_transform([np.array(pers0)])[0]
    res1s = sil.fit_transform([np.array(pers1)])[0]
    res2s = sil.fit_transform([np.array(pers2)])[0]
    
    return [np.sum(np.square(res0s - res0)), np.sum(np.square(res1s - res1)), np.sum(np.square(res2s - res2))]
    
def preprocess_data(data, filt):
    dataf = filter_ts(data, filt)
    a = np.mean(np.abs(dataf), axis= 1)
    b = np.reshape(a, (a.shape[0], a.shape[1]*a.shape[2]), order='F').T
    
    labels = [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2]
    labels = np.repeat(labels, 108)
    
    invalid_ch = np.all(np.isnan(b), axis=1)
    valid_ch = np.logical_not(invalid_ch)
    b = b[valid_ch, :]
    labels = labels[valid_ch]
    
    invalid_ch = np.any(np.isnan(b), axis=0)
    valid_ch = np.logical_not(invalid_ch)
    b = b[:, valid_ch] 
           
    norms=np.linalg.norm(b,axis=1)
    no_outliers=abs(norms - np.mean(norms,axis=0)) < 2 * np.std(norms)
    b = b[no_outliers, :]
    labels = labels[no_outliers]
    

    return b, labels
    
    
def block_results(data, num, p):
    filterrs = ['no_filter', 'alpha', 'beta', 'gamma']
    
    results = []
    filterr = []
    size = []
    for filt in filterrs:
        p.print(num, filt)
        X, labels = preprocess_data(data, filt)

        for s in range(2, 11):
            p.print(num, filt, s)
            pca = PCA(n_components=s)
            X_r = pca.fit_transform(X)
        
            res = []
            for i in range(5):
                X_train, X_test, y_train, y_test = train_test_split(X_r, labels, test_size=0.2, stratify=labels)

                mtv0 = X_train[y_train == 0, :]
                mtv1 = X_train[y_train == 1, :]
                mtv2 = X_train[y_train == 2, :]

                res0_0, res1_0, res2_0 = get_candidates_sil(mtv0, mtv1, mtv2, 0)

                y_hat = []
                for sample in X_test:
                    d0 = get_distances(sample, mtv0, mtv1, mtv2, res0_0, res1_0, res2_0, 0)

                    y_hat.append(np.argmin(d0))

                results.append(np.sum(np.array(y_hat) == y_test)/len(y_test))
                filterr.append(filt)
                size.append(s)
            

        df = pd.DataFrame({'results' : results, 'filter': filterr, 'size': size})
        df.to_csv("Results/electrodesPCA/subject_"+str(num)+"_results_electrodes_pca_"+filt+".csv", index=False)
    return results, filterr, size   
    
    
    
    
    
def main():
    with pymp.Parallel(16) as p:
        for i in p.range(25, 36):
            load_file = loadmat('dataClean-ICA3-'+str(i)+'-T1.mat')
            data = load_file['dataSorted']
            p.print("LOADED", i)
            results, filterr, size = block_results(data, i, p)
            df = pd.DataFrame({'results' : results, 'filter': filterr, 'size': size})
            df.to_csv("Results/subject_"+str(i)+"_results_electrodes_PCA.csv", index=False)
            
main()
