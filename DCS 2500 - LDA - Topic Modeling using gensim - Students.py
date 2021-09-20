#!/usr/bin/env python
# coding: utf-8

# # This a generic code to run LDA on a corpus

# ## 0. Imports and Auxiliary functions

# In[1]:


from sklearn.feature_extraction.text import CountVectorizer

import os
from sklearn.feature_extraction import text 

import pyLDAvis
import pyLDAvis.gensim

import gensim
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

import nltk
from nltk.tag import pos_tag
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
nltk.download('stopwords')

import matplotlib.pyplot as plt

context_stop_words = ['man', 'things', 'thing', 'also','a', 'b','c', 'd','e', 'f', 'g','said', 'mr', 'one']

lexical_tags = ["NN","NNS", "NNP", "NNPS",
                "JJ","JJR","JJS",
                "VB","VBD","VBG","VBN","VBP","VBZ",
                "RB","RBR","RBS",
                "PRP","PRP$",
                "WP","WP$","WRB"]


## Helper function to print the words of each topic
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i]+ ' ' + str(round(topic[i], 2))
              +' \n ' for i in topic.argsort()[:-no_top_words - 1:-1]]))

        
## Helper Function to break a string into chunks        
def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def remove_stopwords(tokens):
    #stopwords = nltk.corpus.stopwords.words('english')
    
    stop_words = set(nltk.corpus.stopwords.words('english'))

    new_stopwords_list = stop_words.union(context_stop_words)
    
    return [word for word in tokens if word not in new_stopwords_list]


# Takes tokenized text and removes all proper nouns with nltk tags
def remove_functionwords(token):
    # Break down each word into their category for tags
    tags = pos_tag(token)
    
    # Remove all the words with the proper noun tags or possessives or proper noun purals
    # You can alter this to remove more tags (search for nltk tag for more options)
    lexical_tokens = [word for word,pos in tags if pos in lexical_tags]
    
    return lexical_tokens

def stemming(tokens): 
    stemmer = SnowballStemmer("english", ignore_stopwords=False)
    stems = [stemmer.stem(t) for t in tokens]
    return stems
        


# ## 1. Load corpus

# In[2]:


# Change current folder to the location where your files are stored
folder = './HP2'

documents = []
titles = []

# Lets create documents with 1000 words
chunk_size = 1000

for file in os.listdir(folder):
    if file.endswith(".txt"):
        
        filename = os.path.join(folder, file)
        
        print("Parsing: ", filename)
        
        ## Open and read the file
        file = open(filename, "r")         
        text = file.read()

        ## Normalize to lower case
        text = text.lower()

        #Use tokenizer to split the file text into words
        regex_tokenizer = RegexpTokenizer(r'\w+')
        file_words = regex_tokenizer.tokenize(text)

        # Now we will partion the file into documents of same size (chunk_size)
        words_chunks = list(chunks(file_words,chunk_size)) 
        
        # and append documents chunks into the global list
        for i in range(len(words_chunks)):
            documents.append(remove_functionwords(remove_stopwords(words_chunks[i])))   

## At this point it seems documents is ready to be parsed. 
print ("Done loading corpus...")   


# ## 2. Vectorize documents using gensim 

# In[3]:


print ("Documents: ", len(documents) )
id2word = Dictionary(documents)
corp = [id2word.doc2bow(text) for text in documents]


# ## 3. Explore best number of topics using coherence metric 

# In[4]:


coherence_values = []
model_list = []

## Define test range of k values
start_k = 10 
max_k   = 25
step    = 3

## Run multiple models to test best coherence values

for num_of_topics in range(start_k, max_k, step):
    
    print("Checking model coherence for k = ", num_of_topics)
    
    gensim_model = gensim.models.LdaModel(corpus=corp, id2word=id2word, num_topics = num_of_topics, 
                                      alpha='auto',eta='auto', iterations=400, eval_every=5, random_state=2019)
    model_list.append(gensim_model)
    coherencemodel = CoherenceModel(model=gensim_model, texts=documents, dictionary=id2word, coherence='c_v')
    coherence_values.append(coherencemodel.get_coherence())
    
# Show graph
import matplotlib.pyplot as plt

x = range(start_k, max_k, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


# ## 4. Run the topic model

# In[5]:


# using gensim implementation
gensim_model = gensim.models.LdaModel(corpus=corp, id2word=id2word, num_topics = 10, 
                                      alpha='auto',eta='auto', iterations=1000, eval_every=5, random_state=2019)


# ## 5. Create the visualization with LDAvis

# In[6]:


vis = pyLDAvis.gensim.prepare(gensim_model, corp, id2word)

# save visualization
pyLDAvis.save_html(vis, folder + "topic_model.html")

print("Done...")


# <table>
#   <tr>
#     <td style="width:30%"> 
#         <!-- Practice Time Image -->
#         <img src="https://drive.google.com/uc?export=view&id=1SdW8qEHRqtL5HW5i6NwNxsAwdNE7Bw5F">
#     </td>
#     <td style="text-align:left">
#         <h3>
#         Please, create another Topic Model for the U.S. section NYT times articles from January 2016 (on Blackboard) .
#         </h3>
#     </td>
#   </tr>
# </table> 
# 
# 
# 

# In[ ]:




