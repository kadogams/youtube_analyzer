#!/usr/bin/env python
# coding: utf-8
# %%

# %%


"""
Funtions to analyse text via Microsoft Azure's Text Analytics REST API.

more info at:
https://docs.microsoft.com/en-in/azure/cognitive-services/text-analytics/quickstarts/python
"""


# %%


import requests


# %%


def get_languages(documents, subscription_key, text_analytics_base_url):
    """ Detect the languages from a set of documents (max. 1000).
    
    :param dict documents: Dictionary with a documents key that consists of a list of documents
    :param str subscription_key: MS Azure subscription key
    :param str text_analytics_base_url: endpoint for the Text Analytics REST API
    :return: Output to the API call
    :rtype: dict
    """
    language_api_url = text_analytics_base_url + "languages"
    headers   = {"Ocp-Apim-Subscription-Key": subscription_key}
    response  = requests.post(language_api_url,
                              headers=headers,
                              json=documents)
    return response.json()


def get_key_phrases(documents, subscription_key, text_analytics_base_url):
    """ Extract the key phrases from a set of documents (max. 1000).
    
    :param dict documents: Dictionary with a documents key that consists of a list of documents
    :param str subscription_key: MS Azure subscription key
    :param str text_analytics_base_url: endpoint for the Text Analytics REST API
    :return: Output to the API call
    :rtype: dict
    """
    keyPhrase_url = text_analytics_base_url + "keyPhrases"
    headers   = {"Ocp-Apim-Subscription-Key": subscription_key}
    response  = requests.post(keyPhrase_url,
                              headers=headers,
                              json=documents)
    return response.json()
    
    
def get_sentiments(documents, subscription_key, text_analytics_base_url):
    """ Detect the sentiment (which ranges between positive or negative)
    of a set of documents (max. 1000).
    
    :param dict documents: Dictionary with a documents key that consists of a list of documents
    :param str subscription_key: MS Azure subscription key
    :param str text_analytics_base_url: endpoint for the Text Analytics REST API
    :return: Output to the API call
    :rtype: dict
    """
    sentiments = {}
    sentiment_url = text_analytics_base_url + "sentiment"
    headers   = {"Ocp-Apim-Subscription-Key": subscription_key}
    response  = requests.post(sentiment_url,
                              headers=headers,
                              json=documents)
    return response.json()

