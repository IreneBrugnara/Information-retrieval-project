
from functools import total_ordering
from collections import defaultdict
from math import log
import re
import pickle
from nltk.stem import PorterStemmer
import os



@total_ordering
class Posting:

    def __init__(self, docID, tf):
        self._docID = docID
        self._tf = tf  # term frequency

    def get_from_corpus(self, corpus):
        return corpus[self._docID]

    def __eq__(self, other):
        """Perform the comparison between this posting and another one.
        Since the ordering of the postings is only given by their docID,
        they are equal when their docIDs are equal.
        """
        return self._docID == other._docID

    def __gt__(self, other):
        """As in the case of __eq__, the ordering of postings is given
        by the ordering of their docIDs
        """
        return self._docID > other._docID

    def __repr__(self):
        return str(self._docID) + " [" + str(self._tf) + "]"




class PostingList:

    def __init__(self):
        self._postings = []

    @classmethod
    def from_docID(cls, docID, tf):
        """ A posting list can be constructed starting from
        a single docID.
        """
        plist = cls()
        plist._postings = [Posting(docID, tf)]
        return plist

    @classmethod
    def from_posting_list(cls, postingList):
        """ A posting list can also be constructed by using another
        """
        plist = cls()
        plist._postings = postingList
        return plist

    def merge(self, other):
        """Merge the other posting list to this one in a desctructive
        way, i.e., modifying the current posting list. This method assume
        that all the docIDs of the second list are higher than the ones
        in this list. It assumes the two posting lists to be ordered
        and non-empty. Under those assumptions duplicate docIDs are
        discarded
        """
        i = 0
        last = self._postings[-1]
        while (i < len(other._postings) and last == other._postings[i]):
            last._tf += other._postings[i]._tf
            i += 1
        self._postings += other._postings[i:]

    def get_from_corpus(self, corpus):
        return list(map(lambda x: x.get_from_corpus(corpus), self._postings))

    def __len__(self):
        return len(self._postings)

    def __iter__(self):
        return iter(self._postings)

    def __repr__(self):
        return ", ".join(map(str, self._postings))




class ImpossibleMergeError(Exception):
    pass

@total_ordering
class Term:

    def __init__(self, term, docID, tf):
        self.term = term
        self.posting_list = PostingList.from_docID(docID, tf)

    def merge(self, other):
        """Merge (destructively) this term and the corresponding posting list
        with another equal term and its corrsponding posting list.
        """
        if (self.term == other.term):
            self.posting_list.merge(other.posting_list)
        else:
            raise ImpossibleMergeError()

    def __eq__(self, other):
        return self.term == other.term

    def __gt__(self, other):
        return self.term > other.term

    def __repr__(self):
        return self.term + ": " + repr(self.posting_list)



def normalize(text):
    """ A simple function to normalize a text.
    It removes everything that is not a word, a space or an hyphen
    and downcases all the text.
    """
    no_punctuation = re.sub(r'[^\w^\s^-]', '', text)
    downcase = no_punctuation.lower()
    return downcase


def tokenize(text):
    """ From a (document or query) text returns a list of all
    tokens it contains after performing normalization, stop words
    removal and stemming.
    """
    norm_text = normalize(text)
    tokens = norm_text.split()
    # custom list of stop words obtained by taking the first 20 most frequent words in the corpus (in terms of document frequency)
    stop_list = ["the", "of", "and", "in", "to", "a", "for", "library", "on", "is", "are",
                 "by", "with", "information", "libraries", "an", "as", "from", "which", "at"]  # HARD CODED
    terms = filter(lambda t: not t in stop_list, tokens)
    stemmer = PorterStemmer()
    stemmed_tokens = map(stemmer.stem, terms)    
    return list(stemmed_tokens)


class InvertedIndex:

    def __init__(self):
        self._dictionary = []     # a list of terms
        self._N = 0               # number of documents in the collection
        self._lengths = {}        # lengths of documents (a dictionary from docIDs to lengths)

    @classmethod
    def from_corpus(cls, corpus):
        """ Create an inverted index from a corpus; the corpus must be
        a dictionary of Articles, and the keys of the dictionary are
        used as docIDs for the inverted index"""
        intermediate_dict = {}
        doc_len = {}
        for docID, document in corpus.items():
            tokens = tokenize(document.title+" "+document.abstract)  # simply merge together title and abstract
            doc_len[docID] = len(tokens)
            for token in tokens:
                term = Term(token, docID, 1)
                try:
                    intermediate_dict[token].merge(term)
                except KeyError:
                    intermediate_dict[token] = term
            # To observe the progress of our indexing:
            if (docID % 1000 == 0):
                print(str(docID), end='...')
        idx = InvertedIndex()
        idx._N = len(corpus)
        idx._dictionary = sorted(intermediate_dict.values())
        idx._lengths = doc_len
        return idx

    @classmethod
    def create_pickle(cls, corpus, filename="index.pickle"):
        """ Create a pickle file to save the inverted index
        """
        idx = cls.from_corpus(corpus)
        with open(filename, "wb") as write_file:
            pickle.dump(idx, write_file)
        print("finished pickling")
        
    @classmethod
    def from_pickle(cls, corpus, filename="index.pickle"):
        """ Read an inverted index from a previously created
        pickle file """
        with open(filename, "rb") as read_file:
            idx = pickle.load(read_file)
        print("finished reading")
        return idx
        
        
    def __getitem__(self, key):   # binary search
        start = 0
        end = len(self._dictionary)

        while (start <= end):
            mid = (start + end) // 2
            if key == self._dictionary[mid].term:
                return self._dictionary[mid].posting_list
            if key < self._dictionary[mid].term:
                end = mid - 1
            else:
                start = mid + 1
        raise KeyError

    
    def __repr__(self):
        return "A dictionary with " + str(len(self._dictionary)) + " terms"
    



class Article:

    def __init__(self, title, abstract):
        self.title = title
        self.abstract = abstract

    def __repr__(self):
        return self.title


    

# !! please see file "Issues_with_dataset.txt" !!


def read_LISA_corpus():
    corpus = {}    # a dictionary from docIDs to Articles
    for filename in sorted(os.listdir('LISA_corpus/documents')):   # the documents are spread in various files
        with open('LISA_corpus/documents/'+filename, 'r') as reader:
            tmp = ""
            first_empty_line = True   # this variable is needed because there are two documents which have an empty line
                                      # before the "***", but I use the empty line to detect the separation between title and abstract
            for i, line in enumerate(reader):
                # Correct the error in the file "LISA1.501": skip lines from 5541 to 5548 (see "Issues_with_dataset.txt")
                if filename=='LISA1.501' and i>=5541 and i<=5548:
                    continue
                if line.startswith("Document "):
                    docID = int(line.split()[1])
                elif line.startswith("***"):  # we have reached the end of the abstract and the end of the document
                    abstract = tmp
                    tmp = ""
                    title, abstract = fix_split(title, abstract)
                    doc = Article(title, abstract)
                    corpus[docID] = doc
                    first_empty_line = True
                elif (line=="\n" or line=="     \n") and first_empty_line:  # we have reached the end of the title
                    title = tmp
                    tmp = ""
                    first_empty_line = False
                else:   # append line to title/abstract
                    tmp += line
    return corpus
    

def fix_split(title, abstract):   # see "Issues_with_dataset.txt"
    pattern = "\.\w{1,5}$"
    match = re.search(pattern, title)  # find titles which end with a point followed by one to five letters
    if match:
        letters = match.group()[1:]
        title = re.sub(pattern, "", title)
        abstract = letters + abstract    # move the letters from the end of the title to the beginning of the abstract
    return title, abstract




class IRsystem:

    def __init__(self, corpus, index):
        self._corpus = corpus
        self._index = index
        
    @classmethod
    def from_pickle(cls, corpus):
        index = InvertedIndex.from_pickle(corpus)
        return cls(corpus, index)

    @classmethod
    def from_corpus(cls, corpus):
        index = InvertedIndex.from_corpus(corpus)
        return cls(corpus, index)
    
    @classmethod
    def create_pickle(cls, corpus):
        InvertedIndex.create_pickle(corpus)




class Query:
    """ A class used to answer queries.
    
    Attributes
    ----------
    _corpus   : corpus of the documents which may answer the query
    text      : original text of the query
    _words    : set of terms contained both in the query and the dictionary, without duplicates
    _bm25     : object of class BM25
    _feedback : object of class RelevanceFeedback
    _ranking  : object of class Ranking
    
    """
    
    def __init__(self, text, ir, k1_param=1.5, b_param=0.75, k3_param=1.5):  # to set the default params I followed the book's advice (An introduction to information retrieval, Schutze-Manning-Raghavan)
        """
        Parameters
        ----------
        text: text of the query
        ir: IRsystem used to answer the query
        k1_param: parameter of the BM25 model which weights document term frequency (default 1.5)
        b_param: parameter of the BM25 model which weights document length normalisation (default 0.75)
        k3_param: parameter of the BM25 model which weights query term frequency (default 1.5)
        
        """
        self._corpus = ir._corpus
        self.text = text
        self._bm25, self._words = BM25.from_index(ir._index, text, k1_param, b_param, k3_param)
        # the initialisation of self._words is equivalent to:
        # self._words = set(filter(lambda w: w in ir._index, tokenize(text)))
        self._feedback = RelevanceFeedback(ir._index, self._words)  # ORDER DEPENDENCY
        self._ranking = Ranking(self._bm25.doc_list())  # ORDER DEPENDENCY
                
    def iterative_pseudo_relevance(self, top_k=5):
        """ Iterates pseudo relevance feedback until convergence
        
        Parameter
        ---------
        top_k : number of documents at the top of the rank to be considered relevant at each iteration (default 5)
        
        """
        self._feedback.reset_likes()  # needed just in case the user performs user-relevance feedback, then pseudo-relevance, then user-relevance again
        self.update_results()
        not_converged = True
        maxit = 20   # stop after 20 iterations even if has not converged
        i = 0   # iteration count
        while(not_converged and i<maxit):
            top_rank = self._ranking.top_k_docs(top_k)
            self.give_feedback(top_rank, pseudo=True)
            self.update_results()
            not_converged = not self.converged(self._ranking.top_k_docs(top_k), top_rank)
            # if the top_k docs in the rank do not change in two consecutive iterations, they won't
            # change anymore and so we have converged
            i += 1
        self._feedback.reset_VR_and_VRt()
                
    def converged(self, previous_rank, current_rank):
        """ Returns true when the ranking has converged
        and so we can stop pseudo-relevance feedback iterations
        """
        return previous_rank == current_rank
        
    def update_results(self):
        """ Updates the ranking, should be run after instantiating the
        query or after give_feedback() before calling display_results()
        """
        # compute total scores by combining Okapi weights with relevance feedback weights
        relevance_weights = self._feedback.compute_relevance_weights()
        for placing in self._ranking:
            doc = placing.docID
            tot_score = 0
            for term in self._words:   # for all query terms
                tf_score = self._bm25[doc][term]
                relevance_score = relevance_weights[term]
                tot_score += tf_score * relevance_score
            placing.RSV = tot_score
        self._ranking.sort()
                
    def display_results(self, page=1, how_many=10):
        """ Print the results of the query, remember to run update_results() first
        
        Parameters
        ----------
        page     : page number (default: first page)
        how_many : number of documents per page (default: 10)
        """
        plist = self._ranking.top_k_placings(page, how_many)
        answer = [(x.docID, self._corpus[x.docID], x.RSV) for x in plist]
        for doc in answer:
            print("[" + str(doc[0]) + "]\t" + str(doc[1]) + " (" + str(doc[2]) + ")")
        

    def get_results(self):
        """ Returns the docIDs of all documents in the ranking
        (needed for evaluation of the performance of IR system
        to compute precision and recall)
        """
        return self._ranking.all_ranked_docs()
    
        
    def give_feedback(self, relevant_docs, pseudo=False): 
        """ Allows the user to input some documents judged relevant
        
        Parameters
        ---------
        relevant_docs : a list of docIDs of documents that the user judges relevant
        pseudo        : a boolean for pseudo-relevance feedback (the user should not
        								use this, should call iterative_pseudo_relevance() instead)
        """
        # check that relevant_docs are actual docIDs
        if any(map(lambda d: not d in self._corpus, relevant_docs)):
            raise ValueError("Document not in corpus")
        
        relevant_docs = self._feedback.initialize_feedback_round(relevant_docs, pseudo)
        
        # update VR and VRt
        self._feedback.increment_VR(len(relevant_docs))
        for doc in relevant_docs:
            for term in self._words:                                                                  
                if self._bm25.contains(doc, term):   # if the document contains the term
                    self._feedback.increment_VRt(term)                      

    def reset_feedback(self):
        """ Delete the whole set of judged relevant documents given so far
        """
        self._feedback.reset_VR_and_VRt()
        self._feedback.reset_likes()
		
    
class BM25:
    """ A class for storing the part of the RSV that does not depend on relevance feedback,
    can be computed when the query is instantiated and does not change afterwards
    
    Attribute
    --------
    _bm25 : a dictionary from docIDs to TfWeights
    
    """
    def __init__(self):
        self._bm25 = defaultdict(TfWeights)
        
    @classmethod    # custom constructor
    def from_index(cls, index, query_text, k1, b, k3):   # Warning: accesses private attributes of class InvertedIndex
        """ Construct a BM25 model
        
        Parameters
        index      : an InvertedIndex from which to build the BM25
        query_text : original text of the query        
        k1         : parameter of the BM25 model which weights document term frequency
        b          : parameter of the BM25 model which weights document length normalisation
        k3         : parameter of the BM25 model which weights query term frequency
        
        """
        
        # compute query term frequency
        tf_q = defaultdict(lambda: 0)
        for w in tokenize(query_text):
            tf_q[w] += 1
        words = set(tf_q.keys())    # set of query terms without repetitions
        
        Lavg = sum(index._lengths.values())/index._N    # average length of a document in the corpus
        bm25 = cls()

        for w in tf_q.keys():   # for all unique terms in the query
            try:
                res = index[w]     # "res" is a posting list
                tf_wq = tf_q[w]     # frequency of term t in query q
                query_weight = (k3+1)*tf_wq/(k3+tf_wq)
                for posting in res:
                    tf_wd = posting._tf     # frequency of term t in document d
                    Ld = index._lengths[posting._docID]     # length of document d
                    okapi_weight = (k1+1)*tf_wd / (k1*((1-b)+b*Ld/Lavg) + tf_wd)
                    bm25[posting._docID][w] = okapi_weight * query_weight
            except KeyError:
                words.remove(w)    # remove words that are not present in the corpus at all
                
        return bm25, words


    def doc_list(self):
        """ List of the documents which received a positive TfWeight,
        to be put in the rank
        """
        return self._bm25.keys()

    def contains(self, doc, term):
        """ Returns true if the document contains the term
        
        Parameters
        ----------
        doc  : a docID
        term : a query term
        
        """
        return self[doc][term] > 0
    
    def __getitem__(self, key):
        return self._bm25[key]
                
        
class TfWeights:
    """ Stores the part of the weights of a document
    product of the Okapi weight and the query term frequency weight
    
    Parameter
    ---------
    _tfw: a dictionary from query terms to real numbers (weights)
    
    """
    def __init__(self):
        self._tfw = defaultdict(lambda: 0)  # if a document is in the BM25 dictionary but does not contain a query term it will get a weight=0
        
    def __getitem__(self, key):
        return self._tfw[key]
    
    def __setitem__(self,key,value):
        self._tfw[key] = value
        
        
class RelevanceFeedback:
    """ A class containing everything related to relevance feedback
    
    Attributes
    ----------
    _likes : set of docIDs of the documents which the user marked as relevant ("liked") - not used in pseudo-relevance feedback
    _VR    : number of liked documents - equal to the cardinality of _likes
    _VRt   : number of liked documents containing each query term: a dictionary from query terms to integer numbers
    _dft   : number of documents in the corpus containing term t, for each query term t (document frequency of term t)
    _N     : total number of documents in the corpus
    """
    def __init__(self, index, words):
        self._likes = set()
        self._VR = 0
        self._VRt = defaultdict(lambda: 0)
        self._dft = {t: len(index[t]) for t in words}
        self._N = index._N

    def compute_relevance_weights(self):
        """ Compute Robertson-Sparck-Jones weights from VR and VRt;
        at the very beginning, before the user has given any feedback, this score will
        approximately be the idf (inverse document frequency)"""
        relevance_weights = {}
        for term in self._dft:
            df_t = self._dft[term]
            vr_t = self._VRt[term]
            p_t = (vr_t+0.5)/(self._VR+1)   # probability of a document containing term t given that the document is relevant
            u_t = (df_t-vr_t+0.5)/(self._N-self._VR+1)   # probability of a document containing term t given that the document is non-relevant
            relevance_weights[term] = log(p_t/(1-p_t) * (1-u_t)/u_t)    # log odds ratio
        return relevance_weights
           
    def increment_VR(self, how_many):
        self._VR += how_many
                
    def increment_VRt(self, term):
        self._VRt[term] += 1
        
    def reset_VR_and_VRt(self):
        for t in self._VRt:
            self._VRt[t]=0
        self._VR = 0
        
    def reset_likes(self):
        self._likes.clear()
        
    def initialize_feedback_round(self, relevant_docs, pseudo):
        """ To be run at the beginning of a round of relevance feedback;
        in the case of pseudo-relevance, cleans from the previous iteration;
        in the case of user-relevance, checks that the documents were not already judged relevant in previous iterations
        """
        if pseudo: # in the case of pseudo-relevance, discard previous relevant documents
            self.reset_VR_and_VRt()
        else:      # compute the set of new relevant documents
            relevant_docs = list(filter(lambda x: x not in self._likes, relevant_docs))   # filter the documents which the user already marked as relevant
            self._likes.update(relevant_docs)  # add the new relevant docs
        return relevant_docs

class Ranking:
    """ A class to represent the ranking of documents according to their RSV to be returned to the user
    
    Attribute
    ---------
    _rank : a list of Placings
    
    """

    def __init__(self, docs):  
        self._rank = list(map(Placing, docs))
        # the documents in the ranking will remain those in the first iteration,
        # because they are all and only those which contain at least one term of the query
        # (they are those saved in the BM25 dictionary)
        
    def __iter__(self):
        return iter(self._rank)
    
    def sort(self):
        """ Sort the documents in order of decreasing RSV """
        self._rank.sort(key=lambda x: -x.RSV)
    
    def top_k_docs(self, k):
        """ Returns the docIDs of the documents in the top k positions of the rank"""
        return list(map(lambda x: x.docID, self._rank[:k]))
    
    def all_ranked_docs(self):
        """ Returns the docIDs of all documents in the ranking"""
        return self.top_k_docs(None)
    
    def top_k_placings(self, page, how_many):
        """ If we imagine the ranking to be divided in pages of 'how_many' documents each,
        this method returns the Placings of the documents in a given page """
        start = (page-1)*how_many
        stop = page*how_many
        return self._rank[start:stop]

    
class Placing:
    """ A class to represent the placing of a document inside the ranking """
    def __init__(self, docID):
        self.docID = docID
        self.RSV = 0   # Retrieval Status Value (total score)
    
    


