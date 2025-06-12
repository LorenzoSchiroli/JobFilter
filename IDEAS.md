## Tools and Libraries

Interesting links:
- https://github.com/AIHawk-co/Auto_Jobs_Applier (automatic job aplly)
- https://github.com/topics/jobsearch

LLMs:
- https://github.com/langchain-ai/langchain (llm pipelines)
- https://github.com/deepset-ai/haystack (llm pipelines)
- Also: llamaindex, txtai

Scrapers:
- https://github.com/Bunsly/JobSpy (search job offers)
- https://github.com/Nv7-GitHub/googlesearch (google search)
- https://github.com/alirezamika/autoscraper
- https://github.com/joeyism/linkedin_scraper (linkedin person or company scraper, it requires chrome and an email-password)
- selenium
- Other: Crawl4AI, FireCrawl, ScrapeGraphAI

Database:
- PostgreSQL (better)
- MongoDB
- airflow

Vector indexing:
- qdrant
- elasticsearch
- Also: Chroma, Pinecone, Weaviate, Milvus

Simple NLP libraries:
- re
- spacy
- nltk / gensim
- RapidFuzz
- langdetect

Text embeddings: NOMIC, SBERT, BGE, Ollama

Evaluation: Giskard, ragas, trulens

Document extraction: MegaParser, Docling, Llama Parse, ExtractThinker

## Ideas

Things done:
- DONE add try except for scrapers since they don't work all the time
- DONE company info: linkedin scraper opensource (hard!), linkedin scraper API (fee), crunchbase scraper, google scraper "<company_name> number of emplyees", ...?
- DONE run llm locally (done with huggiface slow and ollama fast since it uses 4bit llama3.2-3b)

Next things to do (priority order):
- Move search query creation before the scraper
- Execute script and see if scrapers work propery
- Add search: search query manual or add a file description of what the user is searching
- Use a more powerful model (explore APIs, on demand is better)
- split collecting from searching: first collect job offers and companies in a database (like postgres) and then search them through a RAG
- advanced scraper with a LLM agent? is it possible?
- explore more search libraries
- add a searching company tools on the internet (like AI companies in Germany)

Ideas on LLMs:
- knowledge distillation (from gpt4o to small model specific for job offers data extraction)
- embedding LLM finetuning with feedback implementation (good/bad suggestion)
- using efficient training (q)lora
- cutting cost option: use llm to create rule based filtering and execute rules only
- instruction fine-tuning: train llm to extract structured information (eg. salary range, remote, ...)
- estimate of salary when missing (like llm to extract common data and xgboost to predict salary?)
- to extract structured data: prompting (returning json), semantic / classification tagging (similar to NER) with finetuning or fewshot or zeroshot
- to extract user input: ...

Ideas on storage and retrieval:
- store the plain text (cleaned)
- extract and store specific data (remote?, company name, ...): postgres jsonb
- extract and store the embedding
- what to use: embedding or rules? both to see if we can use hybrid. How? First use sql / rules to do a first filtering then use embedding to rank filterd documents

Additional things (maybe out of scope):
- store job offers or applied jobs or companies better?
- cover letter creation?
- autoapply?