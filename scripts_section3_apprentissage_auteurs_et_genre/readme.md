Les deux scripts ont été développés avec python 3.8 inclus dans Anaconda. 

# CLASSIFICATION DES AUTEURS

Ces deux scripts servent à faire de la détection d'auteurs dans le corpus CIDRE (à télécharger sur https://www.ortolang.fr/market/corpora/cidre et à mettre dans le même répertoire que les scripts).

Pour exécuter la classification par auteur, il faut tout simplement utiliser : 

```
python apprentisage_machine_simple_cidre_auteurs.py
```

# CLASSIFICATION DU GENRE

Pour faire de la classificaiton en homme et femme sur CIDRE, je vous conseille de créer un nouveau répertoire intitulé "CIDRE_homme_femme". 
Ensuite, créer deux sous-répertoire dedans : "corpus_femmes" et "corpus_hommes".

Mettre dans "corpus_femmes" tous les textes de greville, lesueur, sand et segur. Mettre tous les textes au même niveau. 

Mettre tous les textes des autres auteurs dans le sous-répetertoire "corpus_hommes". 

Pour faire de la classification en homme et femmes ur le corpus GIRLS, télécharger les données ici : https://www.ortolang.fr/market/corpora/girls.

Renommer le répertoire contenant les livres des femmes "GIRLS_corpus_femmes" et celui des hommes "GIRLS_corpus_hommes". Les mettre au même niveau que les scripts.

Modifier le script pour sélectionner le corpus CIDRE ou GIRLS selon le souhait (lignes 28 à 43 du script).

Pour exécuter : 

```
python apprentisage_machine_simple_homme_femme.py
```

