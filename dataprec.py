#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for preprocessing raw data from Amazon reviews of Electronics with Core 5.
The raw data can be found in the following link:
http://jmcauley.ucsd.edu/data/amazon/
"""

import json

i = 0
with open('data/amazon_reviews_electronics_5.json', 'r') as fr:
    for json_str in fr:
        json_obj = json.loads(json_str)
        user_id  = json_obj['reviewerID']
        item_id  = json_obj['asin']
        rating   = json_obj['overall']
        text     = json_obj['reviewText']

        rating_tuple = [user_id, item_id, rating]

        if i == 10:
            break

        i += 1
