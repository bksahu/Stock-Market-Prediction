from sklearn.decomposition import FastICA, PCA

def dimReduce(training_set, testing_set, n_components_pca=30, n_components_ica=11):
    pca = PCA(n_components=n_components_pca)
    pca.fit(training_set)
    training_set = pca.transform(training_set)
    testing_set = pca.transform(testing_set)
    
    
    ica = FastICA(n_components=n_components_ica)
    ica.fit(training_set)
    training_set = ica.transform(training_set)
    testing_set = ica.transform(testing_set)

    return training_set, testing_set