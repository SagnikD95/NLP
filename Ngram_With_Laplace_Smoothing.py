import argparse
import math
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import List
from typing import Tuple
from typing import Generator

# Generator for all n-grams in text
# n is a (non-negative) int
# text is a list of strings
# Yields n-gram tuples of the form (string, context), where context is a tuple of strings
def get_ngrams(n: int, text: List[str]) -> Generator[Tuple[str, Tuple[str, ...]], None, None]:
    context = []
    text1 = text
    for i in range(n-1):
        text1.insert(0,str("<s>"))
    text1.append("</s>")
    
    for i in range(n-1):
            context.append("<s>")
    context = tuple(context)
    
    #print(text)
    
    for i in range(n-1,len(text1)):
        
        context = list(context)
        context.append(text1[i-1])
        context.pop(0)
        context = tuple(context)
        ngram_tuple = ((text1[i]),context)
        #print(ngram_tuple)
        yield ngram_tuple
       


# Loads and tokenizes a corpus
# corpus_path is a string
# Returns a list of sentences, where each sentence is a list of strings
def load_corpus(corpus_path: str) -> List[List[str]]:
    corpus = []
    f = open(corpus_path, 'r')
    content = f.read()
    f.close()
    paragraphs =  content.split('\n\n')
    
    for paragraph in paragraphs:
        sentences = sent_tokenize(paragraph)
        for sentence in sentences:
            words = word_tokenize(sentence)
            corpus.append(words)
    
    return corpus


# Builds an n-gram model from a corpus
# n is a (non-negative) int
# corpus_path is a string
# Returns an NGramLM
def create_ngram_lm(n: int, corpus_path: str) -> 'NGramLM':
    corpus = load_corpus(corpus_path)

    ngramlm = NGramLM(n)
    for sentence in corpus:
        ngramlm.update(sentence)
    return ngramlm


# An n-gram language model
class NGramLM:
    def __init__(self, n: int):
        self.n = n
        self.ngram_counts = {}
        self.context_counts = {}
        self.vocabulary = set()

    # Updates internal counts based on the n-grams in text
    # text is a list of strings
    # No return value
    def update(self, text: List[str]) -> None:
        
        self.vocabulary.add("</s>")
        for word in text:
            self.vocabulary.add(word)
        
        for ngram in (get_ngrams(self.n, text)):
            
            count_ngrams = self.ngram_counts.get(ngram,0)
            count_ngrams = count_ngrams + 1
            self.ngram_counts[ngram] =  count_ngrams
            
            count_context = self.context_counts.get(ngram[1],0)
            count_context = count_context + 1
            self.context_counts[ngram[1]] = count_context
        

    # Calculates the MLE probability of an n-gram
    # word is a string
    # context is a tuple of strings
    # delta is an float
    # Returns a float
    def get_ngram_prob(self, word: str, context: Tuple[str, ...], delta= .0) -> float:
        n = (word,context)
        ngram_freq = self.ngram_counts.get(n,0)
        context_freq = self.context_counts.get(context,0)
        
        if delta==0:
            if (context_freq==0):
                prob = 1/len(self.vocabulary)
            else:
                prob = (ngram_freq)/(context_freq)
        else:
    
            prob = (ngram_freq+delta)/(context_freq+(delta*len(self.vocabulary)))
        
        return prob

    # Calculates the log probability of a sentence
    # sent is a list of strings
    # delta is a float
    # Returns a float
    def get_sent_log_prob(self, sent: List[str], delta=.0) -> float:
        sum_val = 0
        
        for n in (get_ngrams(self.n,sent)):
            p = self.get_ngram_prob(n[0], n[1],delta)
            if (p!=0):
                logp = math.log2(p)
                sum_val = sum_val + logp
            else:
                return float('-inf')
                 
        return sum_val

    # Calculates the perplexity of a language model on a test corpus
    # corpus is a list of lists of strings
    # Returns a float
    def get_perplexity(self, corpus: List[List[str]]) -> float:
        token = 0
        prob = 0.0
                
        for sentence in corpus:  
            for ngram in (get_ngrams(self.n,sentence)):
                token = token + len(ngram)
            p = self.get_sent_log_prob(sentence)
            prob = prob + p
            
        prob_avg = float(prob/token)
        perp = float(math.pow(2,-prob_avg))
        return perp

    # Samples a word from the probability distribution for a given context
    # context is a tuple of strings
    # delta is an float
    # Returns a string
    def generate_random_word(self, context: Tuple[str, ...], delta=.0) -> str:
        vocab = sorted(self.vocabulary)
        r = random.random()
        #print(r)
        rang_low = 0.0
        rang_high = 0.0
        for word in vocab:
            #print(word)
            prob = float(self.get_ngram_prob(word,context,delta))
            rang_high = float(rang_high) + prob
            
            #print(rang_low,rang_high)
            if r>=rang_low and r<rang_high:
                return word
            rang_low = rang_high

    # Generates a random sentence
    # max_length is an int
    # delta is a float
    # Returns a string
    def generate_random_text(self, max_length: int, delta=.0) -> str:
        text =[]
        word = None
        word_c = 0
        context = []
        
        for i in range(self.n-1):
            context.append("<s>")
        context = tuple(context)

        while (max_length > word_c and word!="</s>"):
        
            word = self.generate_random_word(context,delta)
            text.append(str(word))
            context = list(context)
            context.append(word)
            context.pop(0)
            context = tuple(context)
            word_c = word_c + 1
           
        text = " ".join(text)
        
        return text


def main(corpus_path: str, delta: float, seed: int):
    trigram_lm = create_ngram_lm(5, corpus_path)
    
    return trigram_lm
    
    

trigram_lm = main('warpeace.txt', 0, 82761904)
s1 = 'God has given it to me, let him who touches it beware!'
s2 = 'Where is the prince, my Dauphin?'

print(trigram_lm.get_sent_log_prob(word_tokenize(s1),.5))
print(trigram_lm.get_sent_log_prob(word_tokenize(s2),.5))
    
print(trigram_lm.generate_random_text(10,delta=0))
