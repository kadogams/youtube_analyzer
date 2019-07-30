#!/usr/bin/env python
# coding: utf-8
# %%

# %%


"""
Funtions to translate text via IBM Watson's Language Translator API and to analyse text via it's Natural Language Understanding API.

requirements: pyjwt (conda install pyjwt)

clone/download/fork repos from:
 - https://github.com/watson-developer-cloud/python-sdk
 - https://github.com/IBM/python-sdk-core

in python-sdk/ibm_watson/__init__.py: comment everything but:
 - ibm_cloud_sdk_core (line 16)
 - language_translator_v3 (line 21)
 - natural_language_understanding_v1 (line 23)

more info at:
https://cloud.ibm.com/apidocs/natural-language-understanding?code=python
"""


# %%


import os
import sys

from tqdm import tqdm

# IBM_MASTER = os.path.dirname(os.path.abspath(__file__)) + '/python-sdk-master'
# if not IBM_MASTER in sys.path:
#     sys.path.append(IBM_MASTER)

# IBM_CORE_MASTER = os.path.dirname(os.path.abspath(__file__)) + '/python-sdk-core-master'
# if not IBM_CORE_MASTER in sys.path:
#     sys.path.append(IBM_CORE_MASTER)

from ibm_watson import LanguageTranslatorV3, NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, EmotionOptions


# %%


def get_emotions(df, api_key, nlu_base_url):
    """ Detects anger, disgust, fear, joy, or sadness that is conveyed in the contents
    from a Pandas DataFrame's 'text' column.
    
    :param DataFrame df: DataFrame containing an 'id', 'text' and 'text_en' column
    :param str subscription_key: MS Azure subscription key
    :param str text_analytics_base_url: endpoint for the Text Analytics REST API
    :return: Values of the outputs to the API calls (list of tuples)
    :rtype: list
    """
    natural_language_understanding = NaturalLanguageUnderstandingV1(
        version='2018-11-16',
        iam_apikey=api_key,
        url=nlu_base_url
    )
    
    response_list = []
    for row in tqdm(df.itertuples(), total=df.shape[0]):
        try:
            response = natural_language_understanding.analyze(
                text=row.text,
                features=Features(
                    emotion=EmotionOptions()
                )
            ).get_result()
            emotions = response['emotion']['document']['emotion']
            values = (emotions['anger'],
                      emotions['disgust'],
                      emotions['fear'],
                      emotions['joy'],
                      emotions['sadness'],
                      row.id)
            response_list.append(values)
        except:
            values = ('N/A', 'N/A', 'N/A', 'N/A', 'N/A', row.id)
            response_list.append(values)
    return response_list


def get_translations(df, api_key, lt_base_url):
    """ Translate text from a Pandas DataFrame's 'text column to English.
    
    :param DataFrame df: DataFrame containing an 'id', 'text', and 'language' column
    :param str subscription_key: MS Azure subscription key
    :param str text_analytics_base_url: endpoint for the Text Analytics REST API
    :return: Values of the outputs to the API calls (list of tuples)
    :rtype: list
    """
    language_translator = LanguageTranslatorV3(
        version='2018-05-01',
        iam_apikey=api_key,
        url=lt_base_url
    )
    
    response_list = []
    for row in tqdm(df.itertuples(), total=df.shape[0]):
        try:
            response = language_translator.translate(
                text=row.text,
                model_id=f'{row.language}-en'
            ).get_result()
            values = (response['translations'][0]['translation'],
                      row.id)
            response_list.append(values)
        except:
            values = ('N/A', row.id)
            response_list.append(values)
    return response_list


# %%
