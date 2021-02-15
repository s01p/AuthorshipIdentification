#!/usr/bin/env python
# coding: utf-8

# In[8]:


import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
import numpy as np 
import sklearn
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import classification_report, accuracy_score
import spacy
nlp = spacy.load('en')
from spacy import displacy
from nltk.corpus import stopwords


# In[13]:


import pandas as pd
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize




# In[14]:


dataset = pd.read_csv(r'C:\Users\DES\Desktop\major\Book1.csv')
print(dataset.shape)
dataset.head()


# In[15]:


dataset = dataset.replace(to_replace = ["EAP","HPL","MWS"],value = [0,1,2])


# In[16]:


def clean_text(text):
    
    text = text.lower()
    return text


# In[17]:


dataset['text'] = dataset.text.apply(lambda x: clean_text(x))


# In[18]:


dataset['word_count'] = dataset.text.str.split().apply(lambda x: len(x))
dataset.head() 


# In[19]:


dataset['sentence_count'] = dataset.text.str.split(",").apply(lambda x: len(x))
dataset.tail()


# In[20]:


dataset.head()


# In[21]:


def sentence_word_max(x):
    t=[]
    for y in x:
        p=y.split()
        t.append(len(p))
    return max(t)
def sentence_word_max_char(x):
    t=[]
    z=0
    for y in x:
        p=y.split()
        for k in p:
            z = z+len(k)
        t.append(z)
        
    return max(t)
def sentence_word_min_char(x):
    t=[]
    z=0
    for y in x:
        p=y.split()
        for k in p:
            z = z+len(k)
        t.append(z)
        
    return min(t)
def sentence_word_avg_char(x):
    t=[]
    z=0
    for y in x:
        p=y.split()
        for k in p:
            z = z+len(k)
        t.append(z)
        
    return sum(t)/len(t)

def sentence_word_min(x):
    s=[]
    for y in x:
        p=y.split()
        s.append(len(p))                           
    return min(s)
def sentence_word_avg(x):
    r=[]
    for y in x:
        p=y.split()
        r.append(len(p))
    return sum(r)/len(r)
 
        


# In[22]:


dataset['max_word'] = dataset.text.str.split(",").apply(lambda x: sentence_word_max(x))
    

dataset.head()


# In[23]:


dataset['min_word'] = dataset.text.str.split(",").apply(lambda x: sentence_word_min(x))  

dataset.head()


# In[76]:


dataset['average_word'] = dataset.text.str.split(",").apply(lambda x: sentence_word_avg(x))
dataset.head()


# In[77]:


dataset['average_char'] = dataset.text.str.split(",").apply(lambda x: sentence_word_avg_char(x))
dataset.head()


# In[78]:


dataset['max_char'] = dataset.text.str.split(",").apply(lambda x: sentence_word_max_char(x))
dataset.head()


# In[79]:


dataset['min_char'] = dataset.text.str.split(",").apply(lambda x: sentence_word_min_char(x))
dataset.head()


# In[80]:


def process_content(text):
    
    try:
        count=0
        for i in text:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            for j in tagged:
                if j[1]=='PRP' or j[1]=='PRP$'  or j[1]=='WP' or j[1]=='WP$':
                     count = count + 1
        return count

    except Exception as e:
        return 0


# In[81]:


dataset['frequency_pronoun']=dataset.text.str.split(",").apply(lambda x:process_content(x))


# In[82]:


dataset.head()


# In[83]:


def process_content_fun(text):
    count=0
    try:
        for i in text:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            for j in tagged:
                if j[1]=='PRP' or j[1]=='PRP$'  or j[1]=='WP' or j[1]=='WP$' or j[1]=='DT' or j[1]=='MD' or j[1]=='UH' or j[1]=='CC':
                       count = count + 1
        return count

    except Exception as e:
        return 0
dataset['frequency_functional']=dataset.text.str.split(",").apply(lambda x:process_content_fun(x))


# In[84]:


dataset.head()


# In[85]:




stop_words = set(stopwords.words('english'))

def gen_freq(text):
    
    word_list =[] #nltk.word_tokenize(text)

    
    for tw_words in text.split():
        word_list.extend(tw_words)

    
    word_freq = pd.Series(word_list).value_counts()
    
    
    word_freq = word_freq.drop(stop_words, errors='ignore')
    
    return word_freq


# In[86]:


word_freq = gen_freq(dataset.text.str)
print(word_freq)


# In[87]:


rare_100 = word_freq[-100:]
freuent_100 = word_freq[:100]
def any_rare(words, rare_100):
    sum =0
    for word in words:
        if word in rare_100:
            sum = sum + 1
    return sum
def any_freq(words, freuent_100):
    sum=0
    for word in words:
        if word in freuent_100:
            sum = sum+1
    return sum 
dataset['m_rare'] = dataset.text.str.split().apply(lambda x: any_rare(x, rare_100))
dataset['m_frequent'] = dataset.text.str.split().apply(lambda x: any_freq(x,freuent_100))


# In[88]:


dataset.tail()


# In[89]:


def syntactic_level_feature_pos(text):
    count=0
    try:
        for i in text:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            for j in tagged:
                if j[1]=='CD':
                    count=count+1
        return count

    except Exception as e:
        return 0
dataset['frequency_cardinal_digit']= dataset.text.str.split(",").apply(lambda x:syntactic_level_feature_pos(x))


# In[90]:


def syntactic_level_feature_pos1(text):
    count=0
    try:
        for i in text:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            for j in tagged:
                if j[1]=='DT':
                    count=count+1
        return count

    except Exception as e:
        return 0
dataset['frequency_determiner']= dataset.text.str.split(",").apply(lambda x:syntactic_level_feature_pos1(x))


# In[91]:


def syntactic_level_feature_pos2(text):
    count=0
    try:
        for i in text:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            for j in tagged:
                 if j[1]=='EX':
                        count=count+1
        return count

    except Exception as e:
        return 0
dataset['frequency_existential_there']= dataset.text.str.split(",").apply(lambda x:syntactic_level_feature_pos2(x))


# In[92]:


def syntactic_level_feature_pos3(text):
    count=0
    try:
        for i in text:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            for j in tagged:
                if j[1]=='FW':
                     count=count+1
        return count

    except Exception as e:
        return 0
dataset['frequency_foriegn_word']= dataset.text.str.split(",").apply(lambda x:syntactic_level_feature_pos3(x))


# In[93]:


def syntactic_level_feature_pos4(text):
    count=0
    try:
        for i in text:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            for j in tagged:
                if j[1]=='IN':
                     count=count+1
        return count

    except Exception as e:
        return 0
dataset['frequency_preposition']= dataset.text.str.split(",").apply(lambda x:syntactic_level_feature_pos4(x))


# In[94]:


def syntactic_level_feature_pos5(text):
    count=0
    try:
        for i in text:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            for j in tagged:
                 if j[1]=='JJ':
                        count=count+1
        return count

    except Exception as e:
        return 0
dataset['frequency_adj']= dataset.text.str.split(",").apply(lambda x:syntactic_level_feature_pos5(x))


# In[95]:


def syntactic_level_feature_pos6(text):
    count=0
    try:
        for i in text:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            for j in tagged:
                 if j[1]=='JJR':
                    count=count+1
        return count

    except Exception as e:
        return 0
dataset['frequency_comparative_adj']= dataset.text.str.split(",").apply(lambda x:syntactic_level_feature_pos6(x))


# In[96]:


def syntactic_level_feature_pos7(text):
    count=0
    try:
        for i in text:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            for j in tagged:
                if j[1]=='JJS':
                    count=count+1
        return count

    except Exception as e:
        return 0
dataset['frequency_super_adj']= dataset.text.str.split(",").apply(lambda x:syntactic_level_feature_pos7(x))


# In[97]:


def syntactic_level_feature_pos8(text):
    count=0
    try:
        for i in text:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            for j in tagged:
                if j[1]=='MD':
                    count=count+1
        return count

    except Exception as e:
        return 0
dataset['frequency_modal']= dataset.text.str.split(",").apply(lambda x:syntactic_level_feature_pos8(x))


# In[98]:


def syntactic_level_feature_pos9(text):
    count=0
    try:
        for i in text:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            for j in tagged:
                if j[1]=='JJS' or j[1]=='JJR' or j[1]=='JJ':
                     count=count+1
        return count

    except Exception as e:
        return 0
dataset['frequency_all_adj']= dataset.text.str.split(",").apply(lambda x:syntactic_level_feature_pos9(x))


# In[99]:


def syntactic_level_feature_pos10(text):
    count=0
    try:
        for i in text:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            for j in tagged:
                if j[1]=='NN' or j[1]=='NNS' or j[1]=='NNP' or j[1]=='NNPS':
                    count=count+1
        return count

    except Exception as e:
        return 0
dataset['frequency_noun']= dataset.text.str.split(",").apply(lambda x:syntactic_level_feature_pos10(x))


# In[100]:


def syntactic_level_feature_pos11(text):
    count=0
    try:
        for i in text:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            for j in tagged:
                if j[1]=='PDT':
                    count=count+1
        return count

    except Exception as e:
        return 0
dataset['frequency_predeterminer']= dataset.text.str.split(",").apply(lambda x:syntactic_level_feature_pos11(x))


# In[101]:


def syntactic_level_feature_pos12(text):
    count=0
    try:
        for i in text:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            for j in tagged:
                 if j[1]=='POS':
                    count=count+1
        return count

    except Exception as e:
        return 0
dataset['frequency_possesive_ending']= dataset.text.str.split(",").apply(lambda x:syntactic_level_feature_pos12(x))


# In[102]:


def syntactic_level_feature_pos13(text):
    count=0
    try:
        for i in text:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            for j in tagged:
                if j[1]=='RB' or j[1]=='RBR' or j[1]=='RBS':
                    count=count+1
        return count

    except Exception as e:
        return 0
dataset['frequency_adverb']= dataset.text.str.split(",").apply(lambda x:syntactic_level_feature_pos13(x))


# In[103]:


def syntactic_level_feature_pos14(text):
    count=0
    try:
        for i in text:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            for j in tagged:
                 if j[1]=='RP':
                    count=count+1
        return count

    except Exception as e:
        return 0
dataset['frequency_particle']= dataset.text.str.split(",").apply(lambda x:syntactic_level_feature_pos14(x))


# In[104]:


def syntactic_level_feature_pos15(text):
    count=0
    try:
        for i in text:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            for j in tagged:
                if j[1]=='TO':
                    count=count+1
        return count

    except Exception as e:
        return 0
dataset['frequency_to']= dataset.text.str.split(",").apply(lambda x:syntactic_level_feature_pos15(x))


# In[105]:


def syntactic_level_feature_pos16(text):
    count=0
    try:
        for i in text:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            for j in tagged:
                if j[1]=='UH':
                    count=count+1
        return count

    except Exception as e:
        return 0
dataset['frequency_interjection']= dataset.text.str.split(",").apply(lambda x:syntactic_level_feature_pos16(x))


# In[106]:


def syntactic_level_feature_pos17(text):
    count=0
    try:
        for i in text:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            for j in tagged:
                if j[1]=='VB' or j[1]=='VBD' or j[1]=='VBG' or j[1]=='VBN' or j[1]=='VBP' or j[1]=='VBZ':
                     count=count+1
        return count

    except Exception as e:
        return 0
dataset['frequency_verb']= dataset.text.str.split(",").apply(lambda x:syntactic_level_feature_pos17(x))


# In[107]:


def syntactic_level_feature_pos18(text):
    count=0
    try:
        for i in text:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            for j in tagged:
                 if j[1]=='WDT':
                    count=count+1
        return count

    except Exception as e:
        return 0
dataset['frequency_wh_determiner']= dataset.text.str.split(",").apply(lambda x:syntactic_level_feature_pos18(x))


# In[108]:


def syntactic_level_feature_pos19(text):
    count=0
    try:
        for i in text:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            for j in tagged:
                if j[1]=='WP':
                    count=count+1
        return count

    except Exception as e:
        return 0
dataset['frequency_wh_pronoun']= dataset.text.str.split(",").apply(lambda x:syntactic_level_feature_pos19(x))


# In[109]:


def syntactic_level_feature_pos20(text):
    count=0
    try:
        for i in text:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            for j in tagged:
                 if j[1]=='WP$':
                    count=count+1
        return count

    except Exception as e:
        return 0
dataset['frequency_poss_wh_pronoun']= dataset.text.str.split(",").apply(lambda x:syntactic_level_feature_pos20(x))


# In[110]:


def syntactic_level_feature_pos21(text):
    count=0
    try:
        for i in text:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            for j in tagged:
                if j[1]=='WRB':
                    count=count+1
        return count

    except Exception as e:
        return 0
dataset['frequency_wh_adv']= dataset.text.str.split(",").apply(lambda x:syntactic_level_feature_pos21(x))


# In[111]:


dataset.head()


# In[112]:




def syntactic_level_noun_phrase(text):
    try:
        sum = 0
        for i in text:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            chunkGram = r"""chunk: {<DT>+<NN.*>+}"""
            chunkGram1 = r"""chunk1: {<DT>*<JJ.*>+<NN.*>+}"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunkParser1 = nltk.RegexpParser(chunkGram1)
            chunked = chunkParser.parse(tagged)
            chunked1 =  chunkParser1.parse(tagged) 
            for subtree in chunked.subtrees():
                if subtree.label() == 'chunk':
                    sum = sum + 1
            for subtree in chunked1.subtrees():
                if subtree.label() == 'chunk1':
                    sum = sum + 1
        return sum          
    except Exception as e:
        return 0


# In[113]:


dataset['freq_noun_phrase'] = dataset.text.str.split(",").apply(lambda x:syntactic_level_noun_phrase(x))


# In[114]:


dataset.head()


# In[115]:


def syntactic_level_adjective_phrase(text):
    try:
        sum = 0
        for i in text:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            chunkGram = r"""chunk: {<jj.*><jj.*><jj.*>*<NN.*>+}"""
            
            chunkParser = nltk.RegexpParser(chunkGram)
            
            chunked = chunkParser.parse(tagged)
            
            for subtree in chunked.subtrees():
                if subtree.label() == 'chunk':
                    sum = sum + 1
            
        return sum          
    except Exception as e:
        return 0


# In[116]:


dataset['frequency_adjective_phrase']=dataset.text.str.split(",").apply(lambda x:syntactic_level_adjective_phrase(x))


# In[117]:


dataset.head()


# In[118]:


def syntactic_level_prepositional_phrase(text):
    try:
        sum = 0
        for i in text:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            chunkGram = r"""chunk: {<IN><DT>+<NN.*>+}"""
            chunkGram1 = r"""chunk1: {<DT>+<NN.*>+<IN>}"""
            chunkGram2 = r"""chunk2: {<IN><DT>*<JJ.*>+<NN.*>+}"""
            chunkGram3 = r"""chunk3: {<DT>*<JJ.*>+<NN.*>+<IN>}"""
            chunkGram4 = r"""chunk4: {<jj.*><jj.*><jj.*>*<NN.*>+<IN>}"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunkParser1 = nltk.RegexpParser(chunkGram1)
            chunkParser2 = nltk.RegexpParser(chunkGram2)
            chunkParser3 = nltk.RegexpParser(chunkGram3)
            chunkParser4 = nltk.RegexpParser(chunkGram4)
            chunked = chunkParser.parse(tagged)
            chunked1 =  chunkParser1.parse(tagged) 
            chunked2 =  chunkParser2.parse(tagged)
            chunked3 =  chunkParser3.parse(tagged)
            chunked4 =  chunkParser4.parse(tagged)
            for subtree in chunked.subtrees():
                if subtree.label() == 'chunk':
                    sum = sum + 1
            for subtree in chunked1.subtrees():
                if subtree.label() == 'chunk1':
                    sum = sum + 1
            for subtree in chunked2.subtrees(): 
                if subtree.label() == 'chunk2':
                    sum = sum + 1
            for subtree in chunked3.subtrees():
                if subtree.label() == 'chunk3':
                    sum = sum + 1
            for subtree in chunked4.subtrees(): 
                if subtree.label() == 'chunk4':
                    sum = sum + 1
        return sum          
    except Exception as e:
        return 0


# In[119]:


dataset['frequency_prepositional_phrase']=dataset.text.str.split(",").apply(lambda x:syntactic_level_prepositional_phrase(x))


# In[120]:


dataset.head()


# In[121]:


dataset.tail()


# In[122]:


def frequency_direct_obj(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='dobj':
                sum = sum + 1
                       
    return sum             
    


# In[123]:


dataset['frequency_direct_object'] = dataset.text.str.split(",").apply(lambda x: frequency_direct_obj(x))


# In[124]:


dataset.head()


# In[125]:


def frequency_indirect_obj(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='iobj':
                sum = sum + 1
                       
    return sum          


# In[126]:


dataset['frequency_indirect_object'] = dataset.text.str.split(",").apply(lambda x: frequency_indirect_obj(x))


# In[127]:


def frequency_prepositional_obj(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='pobj':
                sum = sum + 1
                       
    return sum             
    


# In[128]:


dataset['frequency_preposition_object'] = dataset.text.str.split(",").apply(lambda x: frequency_prepositional_obj(x))


# In[129]:


dataset.head()


# In[130]:


def frequency_adjectival_complemen(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='acomp':
                sum = sum + 1
                       
    return sum             
    


# In[131]:


dataset['frequency_adjectiv_complement'] = dataset.text.str.split(",").apply(lambda x: frequency_adjectival_complemen(x))


# In[132]:


def frequency_adverbial_clause_modifie(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='advcl':
                sum = sum + 1
                       
    return sum    


# In[133]:


dataset['frequency_adverbial_clause_modifier'] = dataset.text.str.split(",").apply(lambda x: frequency_adverbial_clause_modifie(x))


# In[134]:


def frequency_adverb_modifie(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='advmod':
                sum = sum + 1
                       
    return sum    


# In[135]:


dataset['frequency_adverb_modifier'] = dataset.text.str.split(",").apply(lambda x: frequency_adverb_modifie(x))


# In[136]:


def frequency_agen(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='agent':
                sum = sum + 1
                       
    return sum    


# In[137]:


dataset['frequency_agent'] = dataset.text.str.split(",").apply(lambda x: frequency_agen(x))


# In[138]:


def frequency_appositional_mod(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='appos':
                sum = sum + 1
                       
    return sum    


# In[139]:


dataset['frequency_appositional_modifier'] = dataset.text.str.split(",").apply(lambda x: frequency_appositional_mod(x))


# In[140]:


def frequency_adjectival_modifie(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='amod':
                sum = sum + 1
                       
    return sum    


# In[141]:


dataset['frequency_adjectival_modifier'] = dataset.text.str.split(",").apply(lambda x: frequency_adjectival_modifie(x))


# In[142]:


def frequency_auxiliar(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='aux':
                sum = sum + 1
                       
    return sum    


# In[143]:


dataset['frequency_auxiliary'] = dataset.text.str.split(",").apply(lambda x: frequency_auxiliar(x))


# In[144]:


def frequency_passive_auxiliar(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='auxpass':
                sum = sum + 1
                       
    return sum    


# In[145]:


dataset['frequency_passive_auxiliary'] = dataset.text.str.split(",").apply(lambda x:  frequency_passive_auxiliar(x))


# In[146]:


def frequency_coordination(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='cc':
                sum = sum + 1
                       
    return sum    


# In[147]:


dataset['frequency_coordination'] = dataset.text.str.split(",").apply(lambda x: frequency_coordination(x))


# In[148]:


def frequency_clausal_complemen(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='ccomp':
                sum = sum + 1
                       
    return sum    


# In[149]:


dataset['frequency_clausal_complement'] = dataset.text.str.split(",").apply(lambda x:frequency_clausal_complemen(x))


# In[150]:


def frequency_conjunc(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='conj':
                sum = sum + 1
                       
    return sum    


# In[151]:


dataset['frequency_conjunct'] = dataset.text.str.split(",").apply(lambda x: frequency_conjunc(x))


# In[152]:


def frequency_copul(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='cop':
                sum = sum + 1
                       
    return sum    


# In[153]:


dataset['frequency_copula'] = dataset.text.str.split(",").apply(lambda x:frequency_copul(x))


# In[154]:


def frequency_clausal_subj(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='csubj':
                sum = sum + 1
                       
    return sum    


# In[155]:


dataset['frequency_clausal_subject'] = dataset.text.str.split(",").apply(lambda x: frequency_clausal_subj(x))


# In[156]:


def frequency_clausal_passive_sub(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='csubjpass':
                sum = sum + 1
                       
    return sum    


# In[157]:


dataset['frequency_clausal_passive_subject'] = dataset.text.str.split(",").apply(lambda x: frequency_clausal_passive_sub(x))


# In[158]:


def frequency_dependan(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='dep':
                sum = sum + 1
                       
    return sum    


# In[159]:


dataset['frequency_dependant'] = dataset.text.str.split(",").apply(lambda x:  frequency_dependan(x))


# In[160]:


def frequency_determine(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='det':
                sum = sum + 1
                       
    return sum    


# In[161]:


dataset['frequency_determiner'] = dataset.text.str.split(",").apply(lambda x: frequency_determine(x))


# In[162]:


def frequency_discourse_elemen(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='discourse':
                sum = sum + 1
                       
    return sum    


# In[163]:


dataset['frequency_discourse_element'] = dataset.text.str.split(",").apply(lambda x:  frequency_discourse_elemen(x))


# In[164]:


def frequency_expletiv(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='expl':
                sum = sum + 1
                       
    return sum    


# In[165]:


dataset['frequency_expletive'] = dataset.text.str.split(",").apply(lambda x: frequency_expletiv(x))


# In[166]:


def frequency_goes_wit(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='goeswith':
                sum = sum + 1
                       
    return sum    


# In[167]:


dataset['frequency_goes_with'] = dataset.text.str.split(",").apply(lambda x: frequency_goes_wit(x))


# In[168]:


def frequency_marke(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='mark':
                sum = sum + 1
                       
    return sum    


# In[169]:


dataset['frequency_marker'] = dataset.text.str.split(",").apply(lambda x: frequency_marke(x))


# In[170]:


def frequency_multi_word_expr(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='mwe':
                sum = sum + 1
                       
    return sum    


# In[171]:


dataset['frequency_multi_word_expression'] = dataset.text.str.split(",").apply(lambda x: frequency_multi_word_expr(x))


# In[172]:


def frequency_negation_mod(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='neg':
                sum = sum + 1
                       
    return sum    


# In[173]:


dataset['frequency_negation_modifier'] = dataset.text.str.split(",").apply(lambda x: frequency_negation_mod(x))


# In[174]:


def frequency_noun_compound_mod(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='nn':
                sum = sum + 1
                       
    return sum    


# In[175]:


dataset['frequency_noun_compound_modifier'] = dataset.text.str.split(",").apply(lambda x: frequency_noun_compound_mod(x))


# In[176]:


def frequency_noun_phrase_as_adverbial_mod(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='npadvmod':
                sum = sum + 1
                       
    return sum    


# In[177]:


dataset['frequency_noun_phrase_as_adverbial'] = dataset.text.str.split(",").apply(lambda x: frequency_noun_phrase_as_adverbial_mod(x))


# In[178]:


def frequency_nominal_suj(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='nsubj':
                sum = sum + 1
                       
    return sum    


# In[179]:


dataset['frequency_nominal_subject'] = dataset.text.str.split(",").apply(lambda x: frequency_nominal_suj(x))


# In[180]:


def frequency_passive_nominal_sub(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='nsubjpass':
                sum = sum + 1
                       
    return sum    


# In[181]:


dataset['frequency_adjectival_modifier'] = dataset.text.str.split(",").apply(lambda x: frequency_passive_nominal_sub(x))


# In[182]:


def frequency_numeric_mod(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='num':
                sum = sum + 1
                       
    return sum    


# In[183]:


dataset['frequency_numeric_modifier'] = dataset.text.str.split(",").apply(lambda x: frequency_numeric_mod(x))


# In[184]:


def frequency_element_compound_num(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='number':
                sum = sum + 1
                       
    return sum    


# In[185]:


dataset['frequency_element_compound_number'] = dataset.text.str.split(",").apply(lambda x: frequency_element_compound_num((x)))


# In[186]:


def frequency_parataxi(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='parataxis':
                sum = sum + 1
                       
    return sum    


# In[187]:


dataset['frequency_parataxis'] = dataset.text.str.split(",").apply(lambda x: frequency_parataxi(x))


# In[188]:


def frequency_prepositional_comp(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='pcomp':
                sum = sum + 1
                       
    return sum    


# In[189]:


dataset['frequency_prepositional_complement'] = dataset.text.str.split(",").apply(lambda x: frequency_prepositional_comp(x))


# In[190]:


def frequency_possession_mod(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='poss':
                sum = sum + 1
                       
    return sum    


# In[191]:


dataset['frequency_possession_modifier'] = dataset.text.str.split(",").apply(lambda x: frequency_possession_mod(x))


# In[192]:


def frequency_possessive_mod(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='possessive':
                sum = sum + 1
                       
    return sum    


# In[193]:


dataset['frequency_possessive_modifier'] = dataset.text.str.split(",").apply(lambda x: frequency_possessive_mod(x))


# In[194]:


def frequency_preconjunc(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='preconj':
                sum = sum + 1
                       
    return sum    


# In[195]:


dataset['frequency_preconjunct'] = dataset.text.str.split(",").apply(lambda x: frequency_preconjunc(x))


# In[196]:


def frequency_pre_determin(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='predet':
                sum = sum + 1
                       
    return sum    


# In[197]:


dataset['frequency_pre_determiner'] = dataset.text.str.split(",").apply(lambda x: frequency_pre_determin(x))


# In[198]:


def frequency_prepositional_mod(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='prep':
                sum = sum + 1
                       
    return sum    


# In[199]:


dataset['frequency_prepositional_modifier'] = dataset.text.str.split(",").apply(lambda x:frequency_prepositional_mod(x))


# In[200]:


def frequency_prepositional_clausal_mod(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='prepc':
                sum = sum + 1
                       
    return sum    


# In[201]:


dataset['frequency_prepositional_clausal_modifier'] = dataset.text.str.split(",").apply(lambda x: frequency_prepositional_clausal_mod(x))


# In[202]:


def frequency_phrasal_verb_part(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='prt':
                sum = sum + 1
                       
    return sum    


# In[203]:


dataset['frequency_phrasal_verb_particle'] = dataset.text.str.split(",").apply(lambda x: frequency_phrasal_verb_part(x))


# In[204]:


def frequency_punctuat(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='punct':
                sum = sum + 1
                       
    return sum    


# In[205]:


dataset['frequency_punctuation'] = dataset.text.str.split(",").apply(lambda x: frequency_punctuat(x))


# In[206]:


def frequency_quantifier_phrase_mod(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='quantmod':
                sum = sum + 1
                       
    return sum    


# In[207]:


dataset['frequency_quantifier_phrase_modifier'] = dataset.text.str.split(",").apply(lambda x: frequency_quantifier_phrase_mod(x))


# In[208]:


def frequency_relative_clause_mod(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='rcmod':
                sum = sum + 1
                       
    return sum    


# In[209]:


dataset['frequency_relative_clause_modifier'] = dataset.text.str.split(",").apply(lambda x: frequency_relative_clause_mod(x))


# In[210]:


def frequency_referent(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='ref':
                sum = sum + 1
                       
    return sum    


# In[211]:


dataset['frequency_referen'] = dataset.text.str.split(",").apply(lambda x: frequency_referent(x))


# In[212]:


def frequency_roo(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='root':
                sum = sum + 1
                       
    return sum    


# In[213]:


dataset['frequency_root'] = dataset.text.str.split(",").apply(lambda x: frequency_roo(x))


# In[214]:


def frequency_temporal_mod(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='tmod':
                sum = sum + 1
                       
    return sum    


# In[215]:


dataset['frequency_temporal_moifier'] = dataset.text.str.split(",").apply(lambda x: frequency_temporal_mod(x))


# In[216]:


def frequency_reduced_non_finite_verbal_mod(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='vmod':
                sum = sum + 1
                       
    return sum    


# In[217]:


dataset['frequency_reduced_non_finite_verbal_modifier'] = dataset.text.str.split(",").apply(lambda x: frequency_reduced_non_finite_verbal_mod(x))


# In[218]:


def frequency_open_clausal_complemen(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='xcomp':
                sum = sum + 1
                       
    return sum    


# In[219]:


dataset['frequency_open_clausal_complement'] = dataset.text.str.split(",").apply(lambda x: frequency_open_clausal_complemen(x))


# In[220]:


def frequency_controlling_subjec(t):
    sum = 0
    for sen in t:
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='xsubj':
                sum = sum + 1
                       
    return sum    


# In[221]:


dataset['frequency_controlling_subject'] = dataset.text.str.split(",").apply(lambda x: frequency_controlling_subjec(x))


# In[222]:


dataset.head()


# In[223]:


def fun_active_voice(t):
    sum = 0
    for sen in t:
        a=0
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='csubj' or token.dep_=='nsubj':
                a=1
        if a==0:
            sum = sum + 1
            
                       
    return sum         


# In[224]:


dataset['frequency_active_voice'] = dataset.text.str.split(",").apply(lambda x: fun_active_voice(x))


# In[225]:


def fun_passive_voice(t):
    sum = 0
    for sen in t:
        a=0
        doc =nlp(sen)
        for token in doc:
             
            if token.dep_=='csubjpass' or token.dep_=='nsubjpass':
                a=1
        if a==1:
            sum = sum + 1
            
                       
    return sum     


# In[226]:


dataset['frequency_passive_voice'] = dataset.text.str.split(",").apply(lambda x: fun_passive_voice(x))


# In[227]:


dataset['frequency_passive_voice']


# In[228]:


def freq_present_simple(text):
    a = 0
    
    for t in text:
        doc = nlp(t)
        for token in doc:
            p = [ ]
            if token.tag_ == 'VBZ'  or token.tag_ == 'VBP':
                a = a + 1
                
            if token.tag_ == 'VB':
                for child in token.children:
                    if child.dep_ == 'aux':
                        p.append(child.text)
                        
                s = " "
                s = s.join(p) 
                if s != "shall" and s != "will":
                    a = a + 1
                
    return a          


# In[229]:


dataset['present_simple'] =  dataset.text.str.split(",").apply(lambda x: freq_present_simple(x))


# In[230]:


def freq_future_simple(text):
    
    b = 0
    for t in text:
        doc = nlp(t)
        for token in doc:
            p = [ ]
            
                
            if token.tag_ == 'VB':
                for child in token.children:
                    if child.dep_ == 'aux':
                        p.append(child.text)
                        
                s = " "
                s = s.join(p) 
                if s == "shall" or s == "will":
                    b = b + 1
                
            
                
    return b           


# In[231]:


dataset['future_simple'] =  dataset.text.str.split(",").apply(lambda x: freq_future_simple(x))


# In[232]:


def freq_past_simple(text):
    a = 0
    for t in text:
        doc = nlp(t)
        
        for token in doc:
            if token.tag_ == 'VBD':
                a = a + 1
            
    return a         


# In[233]:


dataset['frequency_past_simple'] =  dataset.text.str.split(",").apply(lambda x: freq_past_simple(x))
dataset.head()


# In[234]:


def freq_present_perfect(text):
    
    a = 0

    for t in text:
        doc = nlp(t)
        for token in doc:
            p = [ ]
            if token.tag_ == 'VBN':
                for child in token.children:
                    if child.dep_ == 'aux':
                        p.append(child.text)
                        
                s = " "
                s = s.join(p)
                if s == "has" or s == "have":
                     a = a +  1
                        
    return a       
            


# In[235]:


dataset[ 'present_perfect' ]= dataset.text.str.split(",").apply(lambda x: freq_present_perfect(x))


# In[236]:


def freq_past_perfect(text):
    
    
    b = 0
    
    for t in text:
        doc = nlp(t)
        for token in doc:
            p = [ ]
            if token.tag_ == 'VBN':
                for child in token.children:
                    if child.dep_ == 'aux':
                        p.append(child.text)
                        
                s = " "
                s = s.join(p)

                if s == "had":
                     b = b + 1
            
    return b       


# In[237]:


dataset[ 'past_perfect' ]= dataset.text.str.split(",").apply(lambda x: freq_past_perfect(x))


# In[238]:


def freq_future_perfect(text):
    

    c = 0
    for t in text:
        doc = nlp(t)
        for token in doc:
            p = [ ]
            if token.tag_ == 'VBN':
                for child in token.children:
                    if child.dep_ == 'aux':
                        p.append(child.text)
                        
                s = " "
                s = s.join(p)
                
                if s == "shall have" or s == "will have":
                    c = c + 1
    return c      


# In[239]:


dataset[ 'future_perfect' ]= dataset.text.str.split(",").apply(lambda x: freq_future_perfect(x))


# In[240]:


def freq_present_continious(text):
    
    a = 0
    
    for t in text:
     
            doc = nlp(t)
            
            for token in doc:
                p = [ ]
                    
                if token.tag_ == 'VBG':
                        
                    for child in token.children:
                             if child.dep_ == 'aux':
                                  p.append( child.text )
                                  
                        
                            
                    s = " "
                    s = s.join(p)
                    if s == "is" or s == "am" or s == "are":
                         a = a + 1
                    
    return a


# In[241]:


dataset[ 'present_continious' ]= dataset.text.str.split(",").apply(lambda x: freq_present_continious(x))


# In[242]:


def freq_present_perfect_continious(text):
    
    
    b = 0
    
    for t in text:
     
            doc = nlp(t)
            
            for token in doc:
                p = [ ]
                    
                if token.tag_ == 'VBG':
                        
                    for child in token.children:
                             if child.dep_ == 'aux':
                                  p.append( child.text )
                                  
                        
                            
                    s = " "
                    s = s.join(p)
                    
                    if s == "has been" or s == "have been":
                          b = b + 1
                    
    return b


# In[243]:


dataset['frequency_present_perf_continious'] =  dataset.text.str.split(",").apply(lambda x: freq_present_perfect_continious(x))
dataset.head()


# In[244]:


def freq_past_continious(text):
    

    c = 0
    
    for t in text:
     
            doc = nlp(t)
            
            for token in doc:
                p = [ ]
                    
                if token.tag_ == 'VBG':
                        
                    for child in token.children:
                             if child.dep_ == 'aux':
                                  p.append( child.text )
                                  
                        
                            
                    s = " "
                    s = s.join(p)
                    
                    if s == "was"  or s == "were":
                          c = c + 1
                
    return c


# In[245]:


dataset['frequency_past_continious'] =  dataset.text.str.split(",").apply(lambda x: freq_past_continious(x))
dataset.head()


# In[246]:


def freq_past_perfect_continious(text):
    
    
    d = 0
    for t in text:
     
            doc = nlp(t)
            
            for token in doc:
                p = [ ]
                    
                if token.tag_ == 'VBG':
                        
                    for child in token.children:
                             if child.dep_ == 'aux':
                                  p.append( child.text )
                                  
                        
                            
                    s = " "
                    s = s.join(p)
                    
                    if s == "had been":
                          d = d + 1
                    
    return d


# In[247]:


dataset['frequency_past_perfect_continious'] =  dataset.text.str.split(",").apply(lambda x: freq_past_perfect_continious(x))
dataset.head()


# In[248]:


def freq_future_continious(text):
    
    
    e = 0
    
    for t in text:
     
            doc = nlp(t)
            
            for token in doc:
                p = [ ]
                    
                if token.tag_ == 'VBG':
                        
                    for child in token.children:
                             if child.dep_ == 'aux':
                                  p.append( child.text )
                                  
                        
                            
                    s = " "
                    s = s.join(p)
                    
                    if s == "shall be" or s == "will be":
                          e = e + 1
                    
    return e


# In[249]:


dataset['frequency_future_continious'] =  dataset.text.str.split(",").apply(lambda x: freq_future_continious(x))
dataset.head()


# In[250]:


def freq_future_perfect_continious(text):
    

    f = 0
    for t in text:
     
            doc = nlp(t)
            
            for token in doc:
                p = [ ]
                    
                if token.tag_ == 'VBG':
                        
                    for child in token.children:
                             if child.dep_ == 'aux':
                                  p.append( child.text )
                                  
                        
                            
                    s = " "
                    s = s.join(p)
                    
                    if s == "shall have been" or s == "will have been":
                           f = f + 1
                            
    return f


# In[251]:


dataset['frequency_future_perfect_continious'] =  dataset.text.str.split(",").apply(lambda x: freq_future_perfect_continious(x))
dataset.head()


# In[252]:


dataset = dataset.drop('id',1)
dataset = dataset.drop('text',1)

X = np.array(dataset.drop('author',1))
Y = np.array(dataset['author'])


# In[254]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test =train_test_split(X, Y, test_size=0.2, random_state=0)  


# In[255]:


sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)  


# In[256]:


from sklearn.decomposition import PCA

pca = PCA(n_components=3)  
X_train_pca = pca.fit_transform(X_train)  
X_test_pca = pca.transform(X_test) 


# In[257]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=10)  
X_train_lda = lda.fit_transform(X_train, Y_train)  
X_test_lda = lda.transform(X_test)  


# In[258]:


seed = 8
scoring = 'accuracy'


# In[259]:


models = []
models.append(('KNN',KNeighborsClassifier(n_neighbors = 1)))
models.append(('SVM',SVC(gamma=1e-1, decision_function_shape='ovo')))
models.append(('lda',LDA()))
models.append(('Decision_tree',DecisionTreeClassifier(random_state=0)))
models.append(('lregression',LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')))
models.append(('randomforest',RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)))
results = []
names = []

for name,model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train_lda, Y_train, cv=kfold,scoring=scoring)
    results.append(name)
    msg = "%s: %f(%f)" %(name, cv_results.mean(), cv_results.std())
    print( msg )
    


# In[262]:


for name, model in models:
    model.fit(X_train_lda, Y_train)
    predictions = model.predict(X_test_lda)
    print(name)
    print(accuracy_score(Y_test,predictions))
    print(classification_report(Y_test,predictions))


# In[ ]:




