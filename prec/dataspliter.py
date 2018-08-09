#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Script for preprocessing raw data from Amazon reviews of Electronics with Core 5.
The raw data can be found in the following link:
http://jmcauley.ucsd.edu/data/amazon/

Data Spliter essentially splits the raw data (single json file) into ratings and
reviews accordingly, which have structure (user_id, item_id, rating) and
(item_id, text) respectively. In particular, the text is a BoW representation
(i.e a fixed-sized numerical vector).
'''

import json

def data_generator(
    filename,
    rating_text_flag=True):
    '''
    A data generator for reading raw data from `Amazon Product Data` provided by
    Julian McAuley (UCSD), and yielding specific content of each tuple. By
    indicating `rating_text_flag` as True, the generator would return tuples of
    reviews consist of (user_id, item_id, rating, helpfulness), otherwise return
    the text only of each of the reviews.
    '''
    with open(filename, 'r') as fr:
        for json_str in fr:
            json_obj = json.loads(json_str)
            user_id  = json_obj['reviewerID']
            item_id  = json_obj['asin']
            rating   = json_obj['overall']
            text     = json_obj['reviewText']
            n_helpful, n_total = json_obj['helpful'] \
                if len(json_obj['helpful']) == 2 else [0, 0]
            if rating_text_flag:
                yield([user_id, item_id, rating, n_helpful, n_total])
            else:
                yield(text)

if __name__ == '__main__':
    from gensim.matutils import corpus2dense
    from gensim import corpora, models
    from textutils import vocabulary, corpus, merge_documents
    from itertools import tee

    # TODO: replace the sample data with complete data. Sample data here is only
    #       for testing purpose.
    # filename = 'data/amazon_reviews_electronics_5.json'
    data_filename    = 'data/sample_data.json'
    corpus_filename  = 'resource/sample.corpus'
    vocab_filename   = 'resource/sample.vocab'
    ratings_filename = 'resource/ratings.txt'
    reviews_filename = 'resource/reviews.txt'

    # Build vocabulary and corpus
    # TODO: when n is larger than 1. The nltk will raise exception to stop the
    #       the iteration. The reason caused the issue might be the existance of
    #       some unexpected characters in the text. This issue should be fixed
    #       in the future.
    # - Initiate data generator for yielding the text of reviews iteratively
    texts = data_generator(data_filename, rating_text_flag=False)
    #   duplicate the generator for the use of building vocabulary and corpus
    #   respectively.
    texts_for_vocab, texts_for_corpus = tee(texts)
    # - Building vocabulary with Uni-gram terms (n=1)
    vocab = vocabulary(texts_for_vocab, min_term_freq=1, n=1)
    vocab.save(vocab_filename)
    # - Building corpus with Uni-gram terms (n=1)
    corpus = corpus(texts_for_corpus, vocab, n=1)
    corpora.MmCorpus.serialize(corpus_filename, corpus)
    # dense_corpus = corpus2dense(corpus, num_terms=len(vocab)).transpose()

    # Split raw data into `reviews` and `ratings`
    # - Initiate data generator for yielding the review tuple iteratively.
    #   The review tuple consists of (`user_id`, `item_id`, `rating`,
    #   `n_helpful`, `n_total`)
    ratings = data_generator(data_filename, rating_text_flag=True)
    #   duplicate the generator for the use of collecting ratings and extracting
    #   item ids for the text of reviews.
    ratings, reviews = tee(ratings)
    # - Merge reviews by item ids
    item_ids = [ review[1] for review in reviews ]
    merged_item_ids, merged_corpus = merge_documents(corpus, vocab, item_ids)
    #   calculate tfidf for merged corpus
    tfidf_model        = models.TfidfModel(merged_corpus)
    merged_tfidf       = tfidf_model[merged_corpus]
    merged_dense_tfidf = corpus2dense(merged_tfidf, num_terms=len(vocab)).transpose()
    # - Write ratings and reviews into text files delimited by `\t`
    with open(ratings_filename, 'w') as ratings_fw, \
         open(reviews_filename, 'w') as reviews_fw:
        for rating in ratings:
            ratings_fw.write('\t'.join(map(str, rating)) + '\n')
        for review in zip(merged_item_ids, merged_dense_tfidf):
            item_id = review[0]
            bow_vec = ','.join(map(str, review[1].tolist()))
            reviews_fw.write('\t'.join((item_id, bow_vec)) + '\n')
