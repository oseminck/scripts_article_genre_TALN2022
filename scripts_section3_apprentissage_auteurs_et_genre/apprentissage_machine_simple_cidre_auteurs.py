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
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from os import listdir
from os.path import isfile, join
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



(aimard,Y_aimard) = read_my_books("CIDRE/aimard", ["1"])    
(balzac,Y_balzac) = read_my_books("CIDRE/balzac", ["2"])    
(feval,Y_feval) = read_my_books("CIDRE/feval", ["3"])    
(greville,Y_greville) = read_my_books("CIDRE/greville", ["4"])    
(lesueur,Y_lesueur) = read_my_books("CIDRE/lesueur", ["5"])  
(ponson,Y_ponson) = read_my_books("CIDRE/ponson", ["6"])    
(sand,Y_sand) = read_my_books("CIDRE/sand", ["7"]) 
(segur,Y_segur) = read_my_books("CIDRE/segur", ["8"])
(verne,Y_verne) = read_my_books("CIDRE/verne", ["9"])   
(zevaco,Y_zevaco) = read_my_books("CIDRE/zevaco", ["10"]) 
(zola,Y_zola) = read_my_books("CIDRE/zola", ["11"])


vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,3), min_df=10)


#variable qui représente les textes
X_tout = []
X_tout.extend(aimard)
X_tout.extend(balzac)
X_tout.extend(feval)
X_tout.extend(greville)
X_tout.extend(lesueur)
X_tout.extend(ponson)
X_tout.extend(sand)
X_tout.extend(segur)
X_tout.extend(verne)
X_tout.extend(zevaco)
X_tout.extend(zola)
 
    
#variable qui représente les auteurs par leur numéro 
Y_tout = []
Y_tout.extend(Y_aimard)
Y_tout.extend(Y_balzac)
Y_tout.extend(Y_feval)
Y_tout.extend(Y_greville)
Y_tout.extend(Y_lesueur)
Y_tout.extend(Y_ponson)
Y_tout.extend(Y_sand)
Y_tout.extend(Y_segur)
Y_tout.extend(Y_verne)
Y_tout.extend(Y_zevaco)
Y_tout.extend(Y_zola)

#Pour affichier le vecteur Y
#print(Y_tout)

#Pour voir le nombre d'instances
#print(len(Y_tout))


mlb = MultiLabelBinarizer()
Y_tout = mlb.fit_transform(Y_tout)


X_df = pd.DataFrame({'textes': X_tout})
X_tout = vectorizer.fit_transform(X_df['textes'])
X_tout = X_tout.toarray()



#évaluation par validation croisée en 9 plis
kf = KFold(n_splits=8, shuffle=True, random_state=1)
kf.get_n_splits(X_tout)


all_results = []

for train_index, test_index in kf.split(X_tout):
    x_train = [X_tout[ind] for ind in train_index]
    y_train = [Y_tout[ind] for ind in train_index]
    x_test = [X_tout[ind] for ind in test_index]
    y_test = [Y_tout[ind] for ind in test_index]
    
    results_pred = []    

    model = OneVsRestClassifier(LogisticRegression(class_weight='balanced', max_iter=5000))
    model.fit(x_train,y_train)

    predictions = model.predict(x_test)

    acc_score = accuracy_score(y_test, predictions)
    all_results.append(acc_score)
    print("accuracy de ce pli : ",acc_score)

print("l'accuracy global",sum(all_results)/len(all_results)) 


