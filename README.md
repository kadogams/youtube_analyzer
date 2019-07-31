# youtube_analyzer.youtubeAnalyzer

A class to play around with Google, IBM and Microsoft APIs.

## What is it?

It allows to:
- scrape the comments to videos of a search request on YouTube Data API
- run a sentiment analysis and key phrases extraction on them using Microsoft Azure Text Analytics API
- run an emotion analysis using IBM Watson Natural Language Understanding API
- store everything in a local SQLite database
- retrieve specific data in a Panda's DataFrame

## Requirements

Some API keys are required if you want to build your own database.
The current repo includes an SQLite database containing about 2K video comments to a YouTube search of the word 'brexit'.


### Packages

The follwing packages need to be installed:
- google-api-python-client
- ibm-watson
- pandas
- tqdm

### API keys (optional)

API keys required for the following methods:
- self.search():
  - Google API
- self.run_analysis():
  - Microsoft Azure Text Analytics API
  - IBM Watson Natural Language Understanding API

## Installation

Clone the repository at the root of your project using the following command:

```bash
git clone --recurse-submodules https://github.com/kadogams/youtube_analyzer.git
```
## Usage

```python
from youtube_analyzer import youtubeAnalyzer

with open('youtube_analyzer/YOUR_CREDENTIALS.json', 'r') as f:
    credentials = json.load(f)

ya = youtubeAnalyzer(
    google_api_key=credentials['google_developer_key'],
    azure_api_key=credentials['azure_subscription_key'],
    azure_text_analytics_base_url=credentials['azure_text_analytics_base_url'],
    watson_nlu_api_key=credentials['watson_nlu_api_key'],
    watson_nlu_base_url=credentials['watson_nlu_base_url']
)

#### OR ####

ya = youtubeAnalyzer()
```

to be continued...
