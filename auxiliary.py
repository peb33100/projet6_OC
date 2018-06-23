 ## Librairie contenant les fonctions locales utilises 
 ## par l'application

import pandas as pd
import numpy as np
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize


from sklearn.preprocessing import MultiLabelBinarizer


def line_treatment(line, list_words, tokenizer):
    """ Fonction qui traite une liste de mots en entrée
    et ressort un bag of words avec RegexTokenizer
    """
    list_to_include = []
    list_to_drop = []
    line = line.lower()
    for mot in list_words:
        if(mot in line.lower()):
            list_to_include.append(mot)
            list_to_drop.append(tokenizer.tokenize(mot)[0])
    
    vocab = tokenizer.tokenize(line)
    
    for inc in list_to_include:
        for w in vocab:
            if(w == tokenizer.tokenize(inc)[0]):
                vocab.remove(w)
                vocab.append(inc)
    return vocab
	
	
def remove_stopwords(vocab):
    """ Fonction qui traite les stopwords
    contenues dans la chaine de mots vocab
    """
    total_list = list(stopwords.words("English"))
    custom_list = ["to", "use", "can", "the", "get", "is", "doe", "way", "two"
                  "one", "an", "there", "are", "new", "like", "using", "vs", "without"]
    total_list = total_list + custom_list
    vocab_copy = vocab.copy()
    for w in vocab:
        if w in total_list:
            vocab_copy.remove(w)
    vocab_copy2 = vocab_copy.copy()
    for w in vocab_copy:
        if len(w) < 2:
            vocab_copy2.remove(w)
    return vocab_copy2
	
def get_wordnet_pos(treebank_tag):
        """
        return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v) 
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            # As default pos in lemmatization is Noun
            return wordnet.NOUN


def lemm_fonction(tokens):
    """ Fonction qui renvoie une liste lemmatizer"""
    lemm = []
    lemmatizer = WordNetLemmatizer()
    tagged = nltk.pos_tag(tokens)
    for word, tag in tagged:
        wntag = get_wordnet_pos(tag)
        if wntag is None:# not supply tag in case of None
            lemma = lemmatizer.lemmatize(word) 
        else:
            lemma = lemmatizer.lemmatize(word, pos=wntag)
        if wntag == wordnet.NOUN:
            lemm.append(lemma)
    return lemm
	
	
def tokenizer_idf(text):
    """Tokenizer traitant nos données d'entrée de tf_idf
    """
    capwork_tokenizer = nltk.tokenize.RegexpTokenizer(r'[a-zA-Z]+')
    list_words_to_keep = [".net", "c++", "c#","sql-server", "asp.net", 
                          'ruby-on-rails', 'objective-c', 'visual-studio-2008',
                         'cocoa-touch', 'vb.net', "visual-studio", ]
    response = line_treatment(text, list_words_to_keep, capwork_tokenizer)
    response3  = lemm_fonction(response)
    return remove_stopwords(response3)
	
def count_null(df, x):
    """Fonction qui compte le % de données nulles
    dans la colonne x d'un tableau df
    """
    a = round((df[x].count() / df.shape[0] * 100), 2)
    return float("%.2f" % a)
	
def print_top_words(model, feature_names, n_top_words):
    """ Fonction qui renvoie les mots les plus fréquents
    d'un theme identifié par model avec son dictionnaire
    feature_name
    s"""
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
		
def get_rwk(lamb, phi, pw):
    """ Fonction renvoyant la relevance des mots
    contenu dans ne vocabulaire
    """
    rw = np.array(phi)
    temp = np.array(phi)
    temp2 = np.array(pw)
    for w in range(temp.shape[1]):
        for k in range(temp.shape[0]):
            rw[k, w] = lamb * np.log(temp[k, w]) + (1-lamb) * np.log(temp[k, w] / temp2[w][0])
    return rw
	
def return_relevant_tags(topic, feature_names, rwk, n_top_words):
    """ Fonction qui renvoie les mots les plus adéquates 
    en fonction du rwk et du topic prédit
    """
    relevance = np.array(rwk)
    list_tags = list()
    for i in relevance[topic, :].argsort()[-1: -n_top_words: -1]:
         list_tags.append(feature_names[i])
    return list_tags
	
def return_main_topic(model_out):
    """ Fonction qui renvoie le topic prédit
    en allant chercher le thème majoritaire
    model_out
    """
    list_topics = list()
    for index, probabilities in enumerate(model_out):
        list_topics.append(probabilities.argsort()[-1])
    return list_topics
	
def return_frame_info(table_df):
    """Fonction qui renvoie la synthèse des éléments
    d'un tableau (remplissage, type, valeur unique)
    """
    df = pd.DataFrame(columns=["% de remplissage", "Type", "Unique"], index=table_df.columns)
    df["% de remplissage"] = df.index
    df["Type"] = df.index
    df["% de remplissage"] = df["% de remplissage"].apply(lambda x: count_null(table_df, x)) 
    df["Type"] = df["Type"].apply(lambda x: table_df[x].dtype) 
    df["Unique"] = df.index
    df["Unique"] = df["Unique"].apply(lambda x: table_df[x].nunique())
    return df
	
def traitement_des_tags(list_to_treat, list_tags):
    """ Fonction qui renvoie la liste des tags filtrées 
    et ordonnées
    """
    for element in list_to_treat:
        list_to_treat = set(list_to_treat)
        if (element not in list_tags):
            list_to_treat.remove(element)
        list_temp = list(list_to_treat)
    return sorted(list_temp)
	
def get_tag_from_proba(response_vect, nb_tag, labelizer):
    """ Fonction qui renvoie les tags correspondants
    en fonction des probabilités obtenues et du nombre
    de tags désirés
    """
    list_response = [list(labelizer.classes_[resp.argsort()[-1: -nb_tag - 1: -1]])
                     for resp in response_vect]
    return list_response
	
def get_tagging_score(y_tag_true, y_tag_predit):
    """ Fonction qui renvoie le nombre de tags vrai
    prédit dans la liste prédite"""
    score_test = list()
    for (x, y) in zip(y_tag_true, y_tag_predit):
        score_test.append(len(set(x) & set(y))/len(set(x)))
    return sum(score_test)/len(score_test)
	