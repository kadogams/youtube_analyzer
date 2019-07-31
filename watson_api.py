#!/usr/bin/env python
# coding: utf-8
# %%

# %%


"""
Funtions to translate text via IBM Watson's Language Translator API
and to analyse text via it's Natural Language Understanding API.

requirements:
 - ibm-watson
 - tqdm

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


def get_emotions(df, nlu_api_key, nlu_base_url):
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
        iam_apikey=nlu_api_key,
        url=nlu_base_url
    )
    
    value_list = []
    progress_bar = tqdm(df.itertuples(), total=df.shape[0])
    progress_bar.set_description('Updating emotions')
    for row in progress_bar:
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
            value_list.append(values)
        except:
            values = ('N/A', 'N/A', 'N/A', 'N/A', 'N/A', row.id)
            value_list.append(values)
    return value_list


def get_translations(df, nlu_api_key, lt_base_url):
    """ Translate text from a Pandas DataFrame's 'text column to English.
    
    :param DataFrame df: DataFrame containing an 'id', 'text', and 'language' column
    :param str subscription_key: MS Azure subscription key
    :param str text_analytics_base_url: endpoint for the Text Analytics REST API
    :return: Values of the outputs to the API calls (list of tuples)
    :rtype: list
    """
    language_translator = LanguageTranslatorV3(
        version='2018-05-01',
        iam_apikey=nlu_api_key,
        url=lt_base_url
    )
    
    value_list = []
    progress_bar = tqdm(df.itertuples(), total=df.shape[0])
    progress_bar.set_description('Updating emotions')
    for row in progress_bar:
        try:
            response = language_translator.translate(
                text=row.text,
                model_id=f'{row.language}-en'
            ).get_result()
            values = (response['translations'][0]['translation'],
                      row.id)
            value_list.append(values)
        except:
            values = ('N/A', row.id)
            value_list.append(values)
    return value_list


# %%
