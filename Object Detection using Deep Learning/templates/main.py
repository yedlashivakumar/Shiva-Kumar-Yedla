from gensim.models import Word2vec
from nltk.tokenize import word_tokenize
text=[
    'Machine learning is facinating',
    'Natural processing is important in AI'
]
tokenized_text=[word_tokenize(sentence.lower()) for sentence in text]
model=Word2vec(sentences=tokenized_text,vector_size=100,window=5,min_count=1,sg=0)
word='language'
vector=model.wv[word]
similar=model.wv.most_similar(word)

print(f"{word},{vector}")