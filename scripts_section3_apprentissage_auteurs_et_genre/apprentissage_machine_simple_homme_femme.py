'''
Created on 29 avr. 2022

@author: olga

Ce script pourrait sans doute être amélioré par une sélection de traits qui accélera le processus.
'''



from sklearn.linear_model import LogisticRegression
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn.metrics import accuracy_score


def read_my_books(path_to_corpus, gender):
    """
    Sert à lire les livres dans un répertoire
    """
    my_texts = []
    my_genders = []
    onlyfiles = [f for f in listdir(path_to_corpus) if isfile(join(path_to_corpus, f))]
    if '.DS_Store' in onlyfiles:
        onlyfiles.remove('.DS_Store')
    for file in onlyfiles:
        with open(os.path.join(path_to_corpus,file), 'r') as f:
            my_string = f.read()
            my_texts.append(my_string)
            my_genders.append(gender)
    return(my_texts, my_genders)




(textes_femmes,Y_femmes) = read_my_books("CIDRE_homme_femme/corpus_femmes", 1)    
(textes_hommes,Y_hommes) = read_my_books("CIDRE_homme_femme/corpus_hommes", 0)    

"""
Pour changer pour le corpus Girls, mettre en commentaire les deux lignes de code ci-dessus et décommenter celles ci-dessous.
"""
#(textes_femmes,Y_femmes) = read_my_books("GIRLS_corpus_femmes", 1)    
#(textes_hommes,Y_hommes) = read_my_books("GIRLS_corpus_hommes", 0)  


vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,3), min_df=10)

#variable qui contient les comptages des n-grammes de taille 1 à 3
X_tout = []
X_tout.extend(textes_femmes)
X_tout.extend(textes_hommes)
X_df = pd.DataFrame({'textes': X_tout})
X_tout = vectorizer.fit_transform(X_df['textes'])
X_tout = X_tout.toarray()

    
#variable qui représente homme ou femme 0 1
Y_tout = []
Y_tout.extend(Y_femmes)
Y_tout.extend(Y_hommes)
Y_tout = np.array(Y_tout)


#validation croisée de 8  
kf = KFold(n_splits=8, shuffle=True, random_state=1)
kf.get_n_splits(X_tout)


all_results = []

for train_index, test_index in kf.split(X_tout):
    x_train = [X_tout[ind] for ind in train_index]
    y_train = [Y_tout[ind] for ind in train_index]
    x_test = [X_tout[ind] for ind in test_index]
    y_test = [Y_tout[ind] for ind in test_index]
    

    results_pred = []    

    model = LogisticRegression(class_weight='balanced', max_iter=5000)
    model.fit(x_train,y_train)

    predictions = model.predict(x_test)
    acc_score = accuracy_score(y_test, predictions)
    all_results.append(acc_score)
    print("accuracy score de ce pli : ", acc_score)
    

print("l'accuracy global",sum(all_results)/len(all_results)) 


