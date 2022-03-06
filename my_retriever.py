from collections import defaultdict
from math import log,sqrt
class Retrieve:
    def __init__(self,index, term_weighting):
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()
        self.doc_word_counts = self.compute_number_of_doc_words(index)
        self.word_doc_freq = self.compute_word_doc_freq(index)

    def compute_number_of_documents(self):
        self.doc_ids = set()
        for term in self.index:
            self.doc_ids.update(self.index[term])
        return len(self.doc_ids)

    def compute_number_of_doc_words(self,index):
        """Calculate the number of words in each document
        """
        doc_word_counts = defaultdict(int)
        for term in index.keys():
            for doc_id, doc_tf in index[term].items():
                doc_word_counts[doc_id] += doc_tf
        return doc_word_counts

    def compute_query_tokens(self,query_tokens):
        """Compute frequency of query_tokens
        """
        counts = defaultdict(int)
        for token in query_tokens:
            counts[token] += 1
        return list(counts.items())

    def compute_word_doc_freq(self,index):
        """Calculate the number of documents that contain certain word
        """
        result = {}
        for key in index.keys():
            result[key] = len(index[key].keys())
        return result

    def compute_tfidf(self,term,doc_word_count,doc_tf,num_docs,word_doc_freq):
        # calculate term frequency
        TF = doc_tf / doc_word_count
        # calculate inverse document frequency
        IDF = log(num_docs / (1 + word_doc_freq[term]))
        return TF * IDF

    def compute_tf(self,doc_tf,doc_word_count):
        # calculate term frequency
        TF = doc_tf / doc_word_count
        return TF

    def doc_to_norm(self,index,word_doc_freq,num_docs,term_weighting):
        """Calculate the vector representation of the document
        """
        doc_norm = defaultdict(int)
        for term in index.keys():
            for doc_id, doc_tf in index[term].items():
                scores = None
                if term_weighting == "binary":
                    scores = 1
                elif term_weighting == "tf":
                    scores = self.compute_tf(doc_tf,self.doc_word_counts[doc_id])
                elif term_weighting == "tfidf":
                    scores = self.compute_tfidf(term, self.doc_word_counts[doc_id], doc_tf, num_docs, word_doc_freq)

                doc_norm[doc_id] += (scores) ** 2

        # take square root squared norms
        for doc_id in doc_norm.keys():
            doc_norm[doc_id] = sqrt(doc_norm[doc_id])
            # [modulus of the doc vector ,the number of words contained in the doc]
            doc_norm[doc_id] = [doc_norm[doc_id], self.doc_word_counts[doc_id]]

        return doc_norm

    def run_query(self,query_token_counts, index, doc_freq, doc_norm, num_docs, term_weighting):
        """Calculate the vector representation of query, and then calculate it together with the
           vector representation of document to get the similarity between query and document,
           and get the recall result
        """
        query_norm = 0
        for query_term, query_tf in query_token_counts:
            if query_term not in index.keys():
                continue
            scores = None
            if term_weighting == "binary":
                scores = 1
            elif term_weighting == "tf":
                scores = self.compute_tf(query_tf, len(query_token_counts))
            elif term_weighting == "tfidf":
                scores = self.compute_tfidf(query_term, len(query_token_counts), query_tf, num_docs, doc_freq)
            query_norm += (scores) ** 2

        query_norm = sqrt(query_norm)

        # calculate cosine similarity for all relevant documents
        doc_to_score = defaultdict(float)
        for query_term, query_tf in query_token_counts:
            # ignore query terms not in the index
            if query_term not in index:
                continue
            # add to similarity for documents that contain current query word
            for doc_id, doc_tf in index[query_term].items():
                query_scores = None
                doc_scores = None
                if term_weighting == "binary":
                    query_scores = 1
                    doc_scores = 1
                elif term_weighting == "tf":
                    query_scores = self.compute_tf(query_tf, len(query_token_counts))
                    doc_scores = self.compute_tf(doc_tf,doc_norm[doc_id][1])
                elif term_weighting == "tfidf":
                    query_scores = self.compute_tfidf(query_term, len(query_token_counts), query_tf, num_docs, doc_freq)
                    doc_scores = self.compute_tfidf(query_term, doc_norm[doc_id][1], doc_tf, num_docs, doc_freq)

                doc_to_score[doc_id] += query_scores * doc_scores / (doc_norm[doc_id][0] * query_norm)

        sorted_docs = sorted(doc_to_score.items(), key=lambda x: -x[1])
        doc_ids = [value[0] for value in sorted_docs]
        return doc_ids

    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms). Returns list 
    # of doc ids for relevant docs (in rank order).
    def for_query(self, query):
        doc_norms = self.doc_to_norm(self.index, self.word_doc_freq,self.num_docs,self.term_weighting)
        query_token_counts = self.compute_query_tokens(query)
        result = self.run_query(query_token_counts, self.index, self.word_doc_freq, doc_norms, self.num_docs,
                                self.term_weighting)
        return result




