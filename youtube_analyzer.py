#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
A class to make a sentiment and emotion analysis on Youtube comments.
The analysis are saved locally in an SQLite database.
Database schema available at 'data/sqlite_schema/database_diagram.png'

requirements:
 - google-api-python-client
 - ibm-watson
 - pandas
 - tqdm

API keys required for the following methods:
 - self.search(): Google API
 - self.run_analysis(): Microsoft Azure Text Analytics API
                        IBM Watson Natural Language Understanding API
"""


# In[ ]:


import json
import os
import sys
from time import sleep

import pandas as pd
from tqdm import tqdm

from googleapiclient.discovery import build
from google.cloud import translate

from .sqlite3_wrapper.database import Database
from .azure_api import get_key_phrases, get_languages, get_sentiments
from .watson_api import get_emotions


# In[ ]:


# SQLite
CONFLIT_RESOLUTION_ALGORITHMS = ['ROLLBACK', 'ABORT', 'FAIL', 'IGNORE', 'REPLACE']

# Google API Client
MAX_SEARCH = 50
MAX_COMMENT_THREADS = 100
RESULT_ORDERS = ['date', 'rating', 'relevance', 'title', 'videoCount', 'viewCount']
COMMENT_ORDERS = ['time', 'relevance']

# MS Azure
AZURE_MAX_DOCUMENTS = 1000
# supported languages for both sentiment analysis and key phrases extraction:
AZURE_SUPPORTED_LANG = ['da', 'nl', 'en', 'fi', 'fr', 'de', 'it', 'no', 'pl', 'pt', 'ru', 'es', 'sv']

# IBM Watson (supported languages for emotion analysis)
WATSON_SUPPORTED_LANG = ['en']


# In[ ]:


class youtubeAnalyzer(Database):
    """ A class to manage the YouTube SQLite database.
    Inherits from base class 'Database' in database.py (a wrapper around the sqlite3 python library)
    Database schema available at 'data/sqlite_schema/database_diagram.png'
    """

    #################
    # Magic Methods #
    #################

    def __init__(self, google_api_key=None, azure_api_key=None, azure_text_analytics_base_url=None,
                 watson_nlu_api_key=None, watson_nlu_base_url=None,
                 conflict_resolution='IGNORE', sqlite_file='youtube.sqlite'):
        """
        :param str google_api_key: Developper key for Google API
        :param str azure_api_key: Subscription key for Azure Text Analytics API
        :param str watson_nlu_api_key: API key for IBM Watson's Natural Language Understanding API
        :param str azure_text_analytics_base_url : Base url for Azure Text Analytics API
        :param str watson_nlu_base_url_nlu: Base url for IBM Watson's Natural Language Understanding API
        :param str conflict_resolution: ON CONFLICT clause for the SQLite queries. Warning: 'REPLACE' will delete the all Azure and Watson analysis.
        :param str sqlite_file: SQLite file name
        """
        if conflict_resolution not in CONFLIT_RESOLUTION_ALGORITHMS:
            raise ValueError("Valid values for the `conflict_resolution` param are '{}', "
                             "the given value is invalid: '{}'"
                             .format("', '".join(CONFLIT_RESOLUTION_ALGORITHMS), conflict_resolution))

        path = os.path.dirname(os.path.abspath(__file__)) + '/data'
        if not os.path.exists(path):
            os.makedirs(path)
        self.dir = path + '/' + sqlite_file
        self.conn = None
        self.cursor = None
        self.conflict_resolution = conflict_resolution
        
        self.__google_api_key = google_api_key
        self.__azure_api_key = azure_api_key
        self.__azure_text_analytics_base_url = azure_text_analytics_base_url 
        self.__watson_nlu_api_key = watson_nlu_api_key
        self.__watson_nlu_base_url = watson_nlu_base_url

        try:
            self._init_youtube()
        except Exception as e:
            print('Could not connect to the Google API Client:', e)
        
        Database.__init__(self, name=self.dir) # init self.conn and self.cursor
        if self.conn is not None:
            print('***** YouTube database directory: {} *****'.format(self.dir))
    
   
    
    ##################
    # Public Methods #
    ##################
    
    def create_structure(self):
        """ Create the wap SQLite database structure (cf. data/sqlite_schema/database_diagram.png).
        SQL 'CREATE TABLE' statements available in 'data/sqlite_schema/*_schema.txt'
        """

        def _create_table(create_table_sql):
            """ Create a table from the create_table_sql statement
            
            :param str create_table_sql: a CREATE TABLE statement
            """
            try:
                self.query(create_table_sql)
            except Exception as e:
                print(e)

        path = os.path.dirname(self.dir) + '/sqlite_schema'
        with open(path + '/channels_schema.txt','r') as f:
            sql_create_channels_table = f.read()
        with open(path + '/comments_schema.txt','r') as f:
            sql_create_comments_table = f.read()
        with open(path + '/videos_schema.txt','r') as f:
            sql_create_videos_table = f.read()
        if self.conn is not None:
            _create_table(sql_create_channels_table)
            _create_table(sql_create_comments_table)
            _create_table(sql_create_videos_table)
            self.conn.commit()
        else:
            print("Error: Cannot create the database connection.")
    
    
    def display_schema(self):
        """ Print the database schemas
        """
        if not self.conn:
            return
        sql = "SELECT SQL FROM sqlite_master WHERE TYPE = 'table'"
        self.query(sql)
        tables = self.cursor.fetchall()
        for table in tables:
            print(*table, '\n')
    
    
    def get_comments_df(self, video_search=None, video_separator='OR',
                        channel_search=None, channel_separator='OR',
                        from_date=None, to_date=None):
        """ Return a DataFrame of comments to the contents with the specified criterias
        in the SQLite database

        :param str video_search: Comma separated words for a keyword search in videos.title and videos.description
        :param str video_separator: Choose between 'AND' (match every words) or 'OR' (match any word) for the `video_search` param
        :param str channel_search: Comma separated words for a keyword search in channels.title and channels.description
        :param str channel_separator: Choose between 'AND' (match every words) or 'OR' (match any word) for the `channel_search` param
        :param datetime-like from_date: From specified comment published date
        :param datetime-like to_date: To specified comment published date
        """
        for separator in (video_separator, channel_separator):
            if separator != 'AND' and separator !='OR':
                raise ValueError("Valid values for the `_separator` params are 'AND' or 'OR', "
                                 "the given value is invalid: '{}'".format(separator))

        condition_list = []
        condition_list.append(self._format_datetime_condition(from_date, to_date))
        condition_list.append(self._format_search_condition(video_search, video_separator,
                                                            channel_search, channel_separator))
        condition_list = list(filter(None, condition_list))
        if condition_list:
            conditions = 'WHERE' + ' AND '.join(condition_list)
        else:
            conditions = ''

        sql = f"""
            SELECT DISTINCT
                comments.*,
                channels.country
            FROM
                comments
            INNER JOIN
                videos ON videos.id = comments.videoId
            INNER JOIN
                channels ON channels.id = comments.authorChannelId
            {conditions}
        """
        df = pd.read_sql_query(sql, self.conn)
        df['publishedAt'] = pd.to_datetime(df['publishedAt'])
        return df
    
    
    def run_analysis(self):
        """ Run a sentiment analysis on the comments of the SQLite database
        via MS Azure Text Analytics and an emotion analysis via
        IBM Watson NLU if their detected language is supported by the APIs.
        The analysis is stored in the SQLite database.
        """
        if self.youtube == None:
            try:
                self._init_youtube()
            except Exception as e:
                print('Could not connect to the Google API Client:', e)
                return
        self._update_languages()
        self._update_sentiments()
        self._update_keywords()
        self._update_emotions()
        self.conn.commit()
        print('Analysis complete.')
    
    
    def search(self, query, n_results=20, result_order='relevance',
               n_comments=100, comment_order='relevance', include_replies=False):
        """ Update the local SQLite database according 
        to the specific search query on YouTube API.
        
        For more information about the search options, please refer
        to the documentation at:
        https://developers.google.com/youtube/v3/docs/search/list
        
        :param str query: Query term to search for
        :param int n_results: Number of search results desired
        :param str result_order: Order of the search results in the API response
        :param int n_comments: Number of comment threads per video desired
        :param str comment_order: Order of the comment threads in the API response
        :param bool include_replies: Include replies to the comments, if any
        """
        if self.youtube == None:
            try:
                self._init_youtube()
            except Exception as e:
                print('Could not connect to the Google API Client:', e)
                return
        
        if result_order not in RESULT_ORDERS:
            raise ValueError("Valid values for the `result_order` param are '{}', "
                             "the given value is invalid: '{}'"
                             .format("', '".join(RESULT_ORDERS), result_order))

        if comment_order not in COMMENT_ORDERS:
            raise ValueError("Valid values for the `comment_order` param are '{}', "
                             "the given value is invalid: '{}'"
                             .format("', '".join(COMMENT_ORDERS), comment_order))
        
        response_list = []
        page_token = None
        for n in range(0, n_results, MAX_SEARCH):
            # Maximum number of search results per page: 50
            max_results = MAX_SEARCH if n_results - n > MAX_SEARCH else n_results - n
            try:
                search_response = self.youtube.search().list(
                    part='snippet',
                    maxResults=max_results,
                    order=result_order, # You may consider using 'viewCount'
                    pageToken=page_token,
                    q=query,
                    safeSearch='none',
                    type='video', # Channels might appear in search results
                ).execute()
            except Exception as e:
                print(f"An error occured during a search request on YouTube API:", e)
                continue
            response_list.append(search_response)
            
            if not 'nextPageToken' in search_response:
                break
            page_token = search_response['nextPageToken']
        
        videoId_list = self._insert_videos(response_list, n_results)
        self._get_comments(videoId_list, n_comments, comment_order, include_replies)
        self.conn.commit()
        print(f"Search results stored in '{self.dir}'")
    
    
    ###################
    # Private Methods #
    ###################
        
    def _format_comment_resource(self, comment_resource):
        """ Format the comment resource into a list of tuples
        for the SQLite query.
        
        :param dict comment_resource: Information about a single YouTube comment
        :return: Specific values of the comment resource
        :rtype: list
        """
        if 'authorChannelId' in comment_resource['snippet'] \
          and 'value' in comment_resource['snippet']['authorChannelId']:
            authorChannelId = comment_resource['snippet']['authorChannelId']['value']
            self._insert_channel(authorChannelId)
        else:
            authorChannelId = None
        
        values = (
            comment_resource['id'],
            comment_resource['snippet']['videoId'],
            authorChannelId,
            str(pd.to_datetime(comment_resource['snippet']['publishedAt'])),
            comment_resource['snippet']['likeCount'],
            comment_resource['snippet']['parentId'] if 'parentId' in comment_resource['snippet'] else None,
            comment_resource['snippet']['textDisplay']
        )
        return values
    
    
    def _format_datetime_condition(self, from_date, to_date):
        """ Return a condition for the SQLite 'WHERE' clause with the specified datetime range

        :param datetime-like from_date: From specified comment published date
        :param datetime-like to_date: To specified comment published date
        :return: A condition for the SQLite 'WHERE' clause
        :rtype: str
        """
        if not from_date and not to_date:
            return None
        elif not from_date:
            from_date = '1900-01-01 00:00:00'
        elif not to_date:
            to_date = 'now'
        return "(contents.published BETWEEN '{}' AND '{}')".format(str(pd.to_datetime(from_date)),
                                                                   str(pd.to_datetime(to_date)))
    

    def _format_search_condition(self, video_search, video_separator, channel_search, channel_separator):
        """ Return a condition for the SQLite 'WHERE' clause with the specified search query.

        :param str video_search: Comma separated words for a keyword search in videos.title and videos.description
        :param str video_separator: Choose between 'AND' (match every words) or 'OR' (match any word) for the `video_search` param
        :param str channel_search: Comma separated words for a keyword search in channels.title and channels.description
        :param str channel_separator: Choose between 'AND' (match every words) or 'OR' (match any word) for the `channel_search` param
        :return: A condition for the SQLite 'WHERE' clause
        :rtype: str
        """
        if not video_search and not channel_search:
            return None
        conditions = []
        if video_search:
            word_list = ["'%{}%'".format(word.strip()) for word in video_search.split(',') if word.strip()]
            video_conditions = [f'videos.title LIKE {word} OR videos.description LIKE {word}'
                                for word in word_list]
            if video_conditions:
                separator = f' {video_separator} '
                conditions.append('({})'.format(separator.join(video_conditions)))
        if channel_search:
            word_list = ["'%{}%'".format(word.strip()) for word in channel_search.split(',') if word.strip()]
            channel_conditions = [f'channels.title LIKE {word} OR channels.description LIKE {word}'
                                  for word in word_list]
            if channel_conditions:
                separator = f' {video_separator} '
                conditions.append('({})'.format(separator.join(channel_conditions)))
        if len(conditions) == 1:
            return conditions.pop()
        return '({})'.format(' OR '.join(conditions))

    
    def _get_comments(self, videoId_list, n_comments, comment_order, include_replies):
        """ Get the comment threads to the videos found with the search request.
        
        :paran list videoId_list: Ids of the comments' parent video
        :param int n_comments: Number of comment threads per video desired
        :param str order: Order of the resources in the API response
        :param bool include_replies: Include replies to the comments, if any
        """
        if include_replies == False:
            part = 'snippet'
        else:
            part = 'snippet,replies'
        
        progress_bar = tqdm(videoId_list)
        progress_bar.set_description('Fetching comment threads by video')
        for videoId in progress_bar:
            
            response_list = []
            page_token = None
            for n in range(0, n_comments, MAX_COMMENT_THREADS):
                # Maximum number of comment threads per page: 100
                max_comments = MAX_COMMENT_THREADS if n_comments - n > MAX_COMMENT_THREADS else n_comments - n
                try:
                    comment_response = self.youtube.commentThreads().list(
                        part=part,
                        maxResults=max_comments,
                        order=comment_order,
                        pageToken=page_token,
                        textFormat = 'plainText',
                        videoId=videoId,
                    ).execute()
                except Exception as e:
                    print(f"An error occured during the commmentThreads request of videoId '{videoId}':", e)
                    continue
                response_list.append(comment_response)
                
                if not 'nextPageToken' in comment_response:
                    break
                page_token = comment_response['nextPageToken']
            
            self._insert_comments(response_list, n_comments)

    
    def _init_youtube(self):
        # Disable OAuthlib's HTTPS verification when running locally.
        # *DO NOT* leave this option enabled in production.
        os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
        
        api_service_name = "youtube"
        api_version = "v3"
        self.youtube = build(
            api_service_name,
            api_version,
            developerKey=self.__google_api_key
        )
    

    def _insert_channel(self, channelId):
        """ Insert the specified channel's info into the SQLite 'channels' table.

        :param str channelId: Id of the channel
        """
        try:
            channel_response = self.youtube.channels().list(
                part='snippet',
                id=channelId
            ).execute()
        except Exception as e:
            print(f"An error occured during the channels request of channelId '{channelId}':", e)
            return
        
        if not 'items' in channel_response:
            return
        # The response should contain one item.
        for channel_resource in channel_response['items']:
            values = (
                channel_resource['id'],
                channel_resource['snippet']['title'],
                channel_resource['snippet']['description'],
                channel_resource['snippet']['country'] if 'country' in channel_resource['snippet'] else None
            )
            sql = f'INSERT OR {self.conflict_resolution} INTO channels VALUES(?,?,?,?)'
            if not values:
                return
            self.cursor.execute(sql, values)
    
    
    def _insert_comments(self, response_list, n_comments):
        """ Insert the collection of comment threads into the SQLite 'comments' table.
        
        :param list response_list: Responses to the commentThreads request
        :param int n_comments: Number of comment threads per video desired
        """
        value_list = []
        progress_bar = tqdm(total=n_comments)
        progress_bar.set_description('Processing comments')
        for comment_response in response_list:
            if not 'items' in comment_response:
                continue

            for item in comment_response['items']:
                values = self._format_comment_resource(item['snippet']['topLevelComment'])
                value_list.append(values)
                if 'replies' in item:
                    for comment in item['replies']['comments']:
                        values = self._format_comment_resource(comment)
                        value_list.append(values)
                progress_bar.update()

        cols = 'id,videoId,authorChannelId,publishedAt,likeCount,parentId,text'
        sql = f'INSERT OR {self.conflict_resolution} INTO comments({cols}) VALUES(?,?,?,?,?,?,?)'
        if value_list:
            self.cursor.executemany(sql, value_list)

            
    
    def _insert_videos(self, response_list, n_results):
        """ Insert the collection of search results into the SQLite 'videos' table.
        
        :param list response_list: Responses to the search request
        :param int n_results: Number of search results desired
        :return: Ids of the videos inserted
        :rtype: list
        """
        value_list = []
        videoId_list = []
        progress_bar = tqdm(total=n_results)
        progress_bar.set_description('Searching videos')
        for search_response in response_list:
            if not 'items' in search_response:
                continue

            for item in search_response['items']:
                self._insert_channel(item['snippet']['channelId'])
                values = (
                    item['id']['videoId'],
                    item['snippet']['channelId'],
                    str(pd.to_datetime(item['snippet']['publishedAt'])),
                    item['snippet']['title'],
                    item['snippet']['description']
                )
                value_list.append(values)
                videoId_list.append(item['id']['videoId'])
                progress_bar.update()
        
        sql = f'INSERT OR {self.conflict_resolution} INTO videos VALUES(?,?,?,?,?)'
        if value_list:
            self.cursor.executemany(sql, value_list)
        return videoId_list
   

    def _update_emotions(self):
        """ Update the 5 emotion columns of the 'comments' table
        via IBM Watson's Natural Language Understanding API.
        """
        sql_select = """
            SELECT id, text, language
            FROM comments
            WHERE anger IS NULL AND text IS NOT NULL
        """
        df = pd.read_sql_query(sql_select, self.conn)
        df = df[df['language'].isin(WATSON_SUPPORTED_LANG)]
        
        if not df.empty:
            sql_update = """
                UPDATE comments
                SET anger = ?, disgust = ?, fear = ?, joy = ?, sadness = ?
                WHERE id = ?
            """
            values = get_emotions(df, self.__watson_nlu_api_key, self.__watson_nlu_base_url)
            if values:
                self.cursor.executemany(sql_update, values)
                
        # Set 'N/A' to comments not supported by the API
        df = pd.read_sql_query(sql_select, self.conn)
        sql_update = """
            UPDATE comments
            SET anger = 'N/A', disgust = 'N/A', fear = 'N/A', joy = 'N/A', sadness = 'N/A'
            WHERE id = ?
        """
        values = [(row.id,) for row in df.itertuples()]
        if values:
            self.cursor.executemany(sql_update, values)


    def _update_keywords(self):
        """ Update the 'keywords' column of the 'comments' table
        via MS Azure's Text Analytics API if the detected
        language is supported by the API.
        """
        sql_select = """
            SELECT id, language, text
            FROM comments
            WHERE keywords IS NULL AND text IS NOT NULL
        """
        df = pd.read_sql_query(sql_select, self.conn)
        df = df[df['language'].isin(AZURE_SUPPORTED_LANG)]

        if not df.empty:
            key_phrases = []
            progress_bar = tqdm(total=df.shape[0])
            progress_bar.set_description('Updating keywords')
            for i in range(0, df.shape[0], AZURE_MAX_DOCUMENTS):
                # maximum number of documents in a request: 1000
                n_documents = AZURE_MAX_DOCUMENTS if df.shape[0] - i > AZURE_MAX_DOCUMENTS else df.shape[0] - i
                documents = {
                    'documents': df.iloc[i:i + n_documents].to_dict('records')
                }
                response = get_key_phrases(documents, self.__azure_api_key, self.__azure_text_analytics_base_url  )
                if 'documents' in response:
                    key_phrases.extend(response['documents'])
                # time sleep not to exceed the API requests limit
                if i + AZURE_MAX_DOCUMENTS < df.shape[0]:
                    sleep(1)
                progress_bar.update(n_documents)

            sql_update = 'UPDATE comments SET keywords = ? WHERE id = ?'
            values = [(','.join(elem['keyPhrases']), elem['id']) for elem in key_phrases]
            if values:
                self.cursor.executemany(sql_update, values)
        
        # Set 'N/A' to comments not supported by the API
        df = pd.read_sql_query(sql_select, self.conn)
        sql_update = "UPDATE comments SET keywords = 'N/A' WHERE id = ?"
        values = [(row.id,) for row in df.itertuples()]
        if values:
            self.cursor.executemany(sql_update, values)


    def _update_languages(self):
        """ Update the 'language' column of the 'comments' table.
        """        
        sql_select = 'SELECT id, text FROM comments WHERE language IS NULL AND text IS NOT NULL'
        df = pd.read_sql_query(sql_select, self.conn)
        if df.empty:
            return
        
        if not df.empty:
            sentiments = []
            progress_bar = tqdm(total=df.shape[0])
            progress_bar.set_description('Updating languages')
            for i in range(0, df.shape[0], AZURE_MAX_DOCUMENTS):
                # maximum number of documents in a request: 1000
                n_documents = AZURE_MAX_DOCUMENTS if df.shape[0] - i > AZURE_MAX_DOCUMENTS else df.shape[0] - i
                documents = {
                    'documents': df.iloc[i:i + n_documents].to_dict('records')
                }
                response = get_languages(documents, self.__azure_api_key, self.__azure_text_analytics_base_url  )
                if 'documents' in response:
                    sentiments.extend(response['documents'])
                # time sleep not to exceed the API requests limit
                if i + AZURE_MAX_DOCUMENTS < df.shape[0]:
                    sleep(1)
                progress_bar.update(n_documents)
            
            sql_update = 'UPDATE comments SET language = ? WHERE id = ?'
            values = [(elem['detectedLanguages'][0]['iso6391Name'], elem['id'])
                      for elem in sentiments]
            if values:
                self.cursor.executemany(sql_update, values)

        # Set '(Unknown)' to comments not supported by the API
        df = pd.read_sql_query(sql_select, self.conn)
        sql_update = "UPDATE comments SET language = '(Unknown)' WHERE id = ?"
        values = [(row.id,) for row in df.itertuples()]
        if values:
            self.cursor.executemany(sql_update, values)
        

    def _update_sentiments(self):
        """ Update the 'sentimentScore' and 'sentimentLabel' columns of the 'comments' table
        via MS Azure's Text Analytics API.
        """
        sql_select = """
            SELECT id, language, text
            FROM comments
            WHERE sentimentScore IS NULL AND text IS NOT NULL
        """
        df = pd.read_sql_query(sql_select, self.conn)
        df = df[df['language'].isin(AZURE_SUPPORTED_LANG)]

        if not df.empty:
            sentiments = []
            progress_bar = tqdm(total=df.shape[0])
            progress_bar.set_description('Updating sentiments')
            for i in range(0, df.shape[0], AZURE_MAX_DOCUMENTS):
                # maximum number of documents in a request: 1000
                n_documents = AZURE_MAX_DOCUMENTS if df.shape[0] - i > AZURE_MAX_DOCUMENTS else df.shape[0] - i
                documents = {
                    'documents': df.iloc[i:i + n_documents].to_dict('records')
                }
                response = get_sentiments(documents, self.__azure_api_key, self.__azure_text_analytics_base_url  )
                if 'documents' in response:
                    sentiments.extend(response['documents'])
                # time sleep not to exceed the API requests limit
                if i + AZURE_MAX_DOCUMENTS < df.shape[0]:
                    sleep(1)
                progress_bar.update(n_documents)

            for elem in sentiments:
                if elem['score'] < 0.4:
                    elem['label'] = 'negative'
                elif elem['score'] < 0.7:
                    elem['label'] = 'neutral'
                else:
                    elem['label'] = 'positive'

            sql_update = """
                UPDATE comments
                SET sentimentLabel = ?, sentimentScore = ?
                WHERE id = ?
            """
            values = [(elem['label'], elem['score'], elem['id']) for elem in sentiments]
            if values:
                self.cursor.executemany(sql_update, values)
        
        # Set 'N/A' to comments not supported by the API
        df = pd.read_sql_query(sql_select, self.conn)
        sql_update = """
            UPDATE comments
            SET sentimentLabel = 'N/A', sentimentScore = 'N/A'
            WHERE id = ?
        """
        values = [(row.id,) for row in df.itertuples()]
        if values:
            self.cursor.executemany(sql_update, values)
