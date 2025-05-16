# JobFilter
Search job offers with advanced filter

## Installation

1. Create env

2. Install poetry
```bash
pip install poetry
```

3. Install dependencies
```bash
poetry install
```

## Behaviour

Input:
- CV

Processing:
- search phase: search from job websites leaveraging their filters
- filtering phase: indexing + ai filter

Output:
- list of relevant job offers


<!-- Things to check in the job offer:
- english language
- no internship
- no startup / company size > 50
- no senior role (except for entry senior)
- check if it's truly fully remote -->

## Tools

Interesting tools to use:
- https://github.com/topics/jobsearch
- https://github.com/AIHawk-co/Auto_Jobs_Applier (automatic job aplly)
- https://github.com/Bunsly/JobSpy (search job offers)
- https://github.com/langchain-ai/langchain (llm pipelines)
- https://github.com/deepset-ai/haystack (llm pipelines)
- https://github.com/Nv7-GitHub/googlesearch (google search)
- https://github.com/joeyism/linkedin_scraper (linkedin person or company scraper, it requires chrome and an email-password)

Simple NLP libraries:
- re
- spacy
- nltk / gensim
- RapidFuzz
- langdetect

Things done:
- DONE add try except for scrapers since they don't work all the time
- DONE company info: linkedin scraper opensource (hard!), linkedin scraper API (fee), crunchbase scraper, google scraper "<company_name> number of emplyees", ...?
- DONE run llm locally (done with huggiface slow and ollama fast since it uses 4bit llama3.2-3b)

Next things to do (priority order):
- Move search query creation before scraper
- Execute seript and see if scrapers work propery
- Add search: search query manual or add a file dexcription of what the user is searching
- Use a more powerful model (explore APIs, on demand is better)
- Cutting cost option: llm to create rule based filtering and execute rules only
- explore more search libraries
- add a searching company tools on the internet (like AI companies in Germany)

Additional things (maybe out of scope):
- store job offers or applied jobs or companies better?
- cover letter creation?
- autoapply?

