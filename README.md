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

The follwing packages need to be installed:
- google-api-python-client
- ibm-watson
- pandas
- tqdm

Also, API keys required for the following methods:
- self.search(): Google API
- self.run_analysis(): Microsoft Azure Text Analytics API
                        IBM Watson Natural Language Understanding API

## Installation

Clone the repository at the root of your project using the following command:

```bash
git clone --recurse-submodules https://github.com/kadogams/youtube_analyzer.git
```

## Usage

```python
from youtube_analyzer import youtubeAnalyzer
```

to be continued...
