# Importing Packages
import warnings
warnings.filterwarnings("ignore") 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
import urllib.request
import re
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn import svm

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow import keras
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import SMOTE
#import pickel
# Reading Data
df = pd.read_csv('Twitter_Data.csv')

# Data Sample
a = df.sample(5)
print(a)

# Checking for NA Values
b = df.isnull().sum()
print(b)

# Distribution of different classes in sentiment
def count_values_in_column(data,feature):
    total=data.loc[:,feature].value_counts(dropna=False)
    percentage=round(data.loc[:,feature].value_counts(dropna=False,normalize=True)*100,2)
    return pd.concat([total,percentage],axis=1,keys=["Total","Percentage"])
c = count_values_in_column(df,"category")

print(c)

# Segrating based on different sentiments
df_negative = df[df["category"]==2]
df_positive = df[df["category"]==1]
df_neutral = df[df["category"]==0]

# create data for Pie Chart
plt.figure(figsize=(13, 8), dpi=80)
pichart = count_values_in_column(df,"category")
names= ["Positive","Neutral","Negative","Nan"]
size=pichart["Percentage"]
 
# Create a circle for the center of the plot
# my_circle=plt.Circle( (0,0), 0.5, color='white')
# plt.pie(size, labels=names, colors=['green','blue','red',"yellow"])
# p=plt.gcf()
# p.gca().add_artist(my_circle)
# plt.show()

# Function to Create Wordcloud
def create_wordcloud(text,path):
    stopwords = set(STOPWORDS)
    wc = WordCloud(background_color="white",
    max_words=3000,
    stopwords=stopwords,
    random_state=42,
    width=900, height=500,
    repeat=True)
    wc.generate(str(text))
    wc.to_file(path)
    print("file Saved Successfully")
    # path=path
    # display(Image.open(path))


# Wordcloud for all tweets
plt.figure(figsize=(15, 8), dpi=80)
create_wordcloud(df['clean_text'].values,"all.png")

# Wordcloud for only positive tweets
plt.figure(figsize=(15, 8), dpi=80)
create_wordcloud(df_positive['clean_text'].values,"desktop/final year project/50%code/positive.png")

# Wordcloud for only negative tweets
plt.figure(figsize=(15, 8), dpi=80)
create_wordcloud(df_negative['clean_text'].values,"negative.png")

# Wordcloud for only neutral tweets
plt.figure(figsize=(15, 8), dpi=80)
create_wordcloud(df_neutral['clean_text'].values,"neutral.png")













