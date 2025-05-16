

#%% search phase

import csv
from jobspy import scrape_jobs

import pandas as pd
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack import Pipeline
from haystack import Document
from haystack.components.generators import HuggingFaceLocalGenerator, HuggingFaceAPIGenerator
from haystack.components.generators.chat import HuggingFaceAPIChatGenerator, HuggingFaceLocalChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.components.builders import PromptBuilder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.generators.ollama import OllamaChatGenerator

import PyPDF2

import time

from huggingface_hub import login

from haystack.utils import Secret
from tqdm import tqdm
import copy
import argparse
from scrapers.company_scraper import get_company_size

from langdetect import detect

import json

from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

huggugface_token = os.environ.get("HUGGINGFACE_TOKEN")

# login(token=huggugface_token)

def get_jobs(generator):
    jobs = scrape_jobs(
        site_name=["indeed", "linkedin", "zip_recruiter", "glassdoor", "google"],
        search_term="software engineer",
        google_search_term="software engineer jobs near San Francisco, CA since yesterday",
        location="San Francisco, CA",
        results_wanted=2,
        hours_old=72,
        country_indeed='Germany',
        
        # linkedin_fetch_description=True # gets more info such as description, direct job url (slower)
        # proxies=["208.195.175.46:65095", "208.195.175.45:65095", "localhost"],
    )
    print(f"Found {len(jobs)} jobs")
    # print(jobs.head())
    # jobs.to_csv("jobs.csv", quoting=csv.QUOTE_NONNUMERIC, escapechar="\\", index=False) # to_excel

    # add company size information
    unique_companies = jobs['company'].unique()
    print("Number of companies:", len(unique_companies))
    mapping = {company: get_company_size(company, generator) for company in tqdm(unique_companies)}
    jobs['company_num_employees'] = jobs['company'].map(mapping)

    return jobs

#%% indexing + ai phase

def row2text(row, columns):
    return '\n\n'.join([f"{col.upper()}\n{row[col]}" for col in columns])

def df2text(jobs):
    # convert to plain text
    jobs_strings = []
    for index, row in jobs.iterrows():
        row_string = row2text(row, jobs.columns)
        jobs_strings.append(row_string)
    return jobs_strings

def pdf_to_text(path):
    """
    Open pdf file and return plain text.
    """

    with open(path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text

def get_generator():
    # LOCAL INFERENCE
    # generator = HuggingFaceLocalGenerator(model="meta-llama/Llama-3.2-1B-Instruct",
    #     task="text-generation",
    #     generation_kwargs={
    #     "max_new_tokens": 100,
    #     "temperature": 0.3,
    # })
    # generator.warm_up()
    
    # ONLINE INFERENCE
    # generator = HuggingFaceAPIGenerator(api_type="serverless_inference_api",
    #     api_params={"model": "meta-llama/Llama-3.2-3B-Instruct"},
    #     token=Secret.from_token(huggugface_token))
    # index_query = generator.run(search_query_request)
    # print(f"Inference time: {time.time() - start}")
    # print(f"Total tokens {len(index_query['replies'][0])}")

    # generator = HuggingFaceAPIChatGenerator(api_type="serverless_inference_api",
    #     api_params={"model": "meta-llama/Llama-3.2-3B-Instruct"},
    #     token=Secret.from_token(huggugface_token))
    
    # generator = HuggingFaceAPIChatGenerator(api_type="serverless_inference_api",
    #     api_params={"model": "meta-llama/Llama-3.1-8B-Instruct"},
    #     token=Secret.from_token(huggugface_token))
    
    generator = OllamaChatGenerator(model="llama3.2",
        url = "http://localhost:11434",
        generation_kwargs={
            "num_predict": 200,
            "temperature": 0.5,
            })
    # generator = HuggingFaceLocalChatGenerator(
    #     model="microsoft/Phi-3.5-mini-instruct",
    #     task="text-generation",
    # )
    # generator.warm_up()
    return generator

def get_search_query(cv, generator):
    search_query_request = f"Curriculum vitae: \n<text-begin>\n {cv} \n<text-end>\n Generate a search query to find a job for this worker (10 words maximum). Don't add comments."
    # Candidate's CV and the initial query format

    start = time.time()
    messages = [ChatMessage.from_system("\\nYou are a helpful, respectful and honest assistant"),
                ChatMessage.from_user(search_query_request)]
    response = generator.run(messages)
    reply_query = response["replies"][0]
    index_query = reply_query.content

    # force manual
    index_query = "data scientist remote german company"

    print(f"Inference time: {time.time() - start}")
    print(f"Total words {index_query.count(' ') + 1}")

    print("Reply:", index_query)

def filter_jobs(jobs_info, cv, generator):

    # rearch query was here

    jobs_text = df2text(jobs_info)

    documents_all = [Document(content=jo, id=i+1) for i, jo in enumerate(jobs_text)]
    document_store = InMemoryDocumentStore()
    # document_writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE)
    document_store.write_documents(documents_all)
    # document_writer.run(documents=documents)
    print("Documents count:", document_store.count_documents())
    retriever = InMemoryBM25Retriever(document_store=document_store)

    messages = [ChatMessage.from_system("\\nYou are a helpful, respectful and honest assistant")]
    messages.append(ChatMessage.from_user("Extract query form the CV: \n" + cv))
    response = generator.run(messages)
    index_query = response["replies"][0].content

    documents_candidate = retriever.run(query=index_query)["documents"]

    matched_jobs = []
    # messages.append(reply_query)
    for document in tqdm(documents_candidate):
        is_match = job_match(generator, messages, jobs_info.iloc[document.id-1], document.content)
        if is_match:
            matched_jobs.append(document.content)

    return matched_jobs


import re

def is_full_time(job_offer_text):
    full_time_keywords = ['full-time', 'full time', 'permanent', 'regular', 'fulltime']
    part_time_keywords = ['part-time', 'part time', 'parttime','freelance', 'hourly', "internship"]
    job_offer_text = job_offer_text.lower()
    found_full_time = any([full_time_keyword in job_offer_text for full_time_keyword in full_time_keywords])
    found_part_time = any([part_time_keyword in job_offer_text for part_time_keyword in part_time_keywords])
    if found_full_time or (not found_full_time and not found_part_time):
        return True
    else:
        return False
    

def rule_based_filtering(job_info):

    checks = {}
    job_text = job_text.lower()
    checks["junior_mid"] = "senior" not in job_text # risky
    checks["english_text"] = True if detect(job_info["description"]) == "en" else False
    checks["company_established"] = "startup" not in job_text
    checks["full_time"] = is_full_time(job_text)
    checks["remote"] = any([keyword in job_text for keyword in ["remote", "home", "teleworking"]])
    checks["company_size"] = True if pd.isna(job_info["company_num_employees"]) or job_info["company_num_employees"] > 100 else False
    checks["permament_position"] = not any([keyword in job_text for keyword in ["internship", "trainee", "fellowship"]])
    checks["mlops"] = any([keyword in job_text for keyword in ["mlops", "AWS", "cloud", "mlflow", "CI/CD", "deploy"]])

    return all([value for key, value in checks.items()])


def llm_based_filtering(generator, messages, job_text):
    match_request = f"Job Offer: \n\n<text-begin>\n {job_text} \n<text-end>\n\n" \
"""
The following list are job offer conditions, check each condition if it is True or False. 
If a condition is unclear or not specified consider it as None.
Job offer conditions:
- "junior_mid": check if the job offer's role is a junior or mid-level role and not an advanced senor role.
- "english_text": check if the description is written in english.
- "company_established": check if the job offer's company is an established company (not a startup).
- "full_time": check if the job is a full-time job or has the option to work full-time.
- "remote": check if the job is truly fully remote without specifications regarding office returns.
- "company_size": check if the job offer's company has more than 100 employees.
- "permament_position": check if the job is a normal job and not an internship.
""" \
        "\nMoreover, check if the curriculum vitae role ambitions match with the the job offer role. Answer True or False or None. \n \n" \
        """
Don't explain details, write output as a json file.
Output example:

{
    "junior_mid": False,
    "english_text": False,
    "company_established": False,
    "full_time": False,
    "reomte": False,
    "company_size": False,
    "permament_position": False,
    "match": False
}
"""
    # "If all conditions are NOT FALSE and the job offer is a MATCH, answer just PASSED, otherwise FAILED." \
        
    messages_side = copy.deepcopy(messages)
    messages_side.append(ChatMessage.from_user(match_request))
    response = generator.run(messages_side)
    reply_match = response["replies"][0].content
    try:
        reply_match = reply_match.replace("\n", "").replace("true", "True").replace("false", "False")
        match = re.search(r'(\{.*?\})', reply_match)
        json_text = match.group(1) if match else {}
        reply_match = json.loads(json_text)
    except Exception as e:
        reply_match = {}

    print(reply_match)

    return all([value for key, value in reply_match.items()])


def job_match(generator, messages, job_info, job_text):

    if pd.isna(job_info["description"]):
        return False
    
    check_rules = rule_based_filtering(job_info)

    if not check_rules:
        return False
    
    check_llm = llm_based_filtering(generator, messages, job_text)

    return check_llm


def get_example():
    jobs = [
    """
Job Offer 1: Remote Full Stack Developer at Tech Innovators Inc.
Location: Remote
Salary: $90,000 - $110,000 per year
Description:
Tech Innovators Inc. is seeking a talented Full Stack Developer to join our dynamic team. The ideal candidate will have experience with JavaScript frameworks (React preferred), Node.js for backend development, and familiarity with AWS services. You will work on exciting projects that involve building scalable applications and collaborating with product managers and designers.
Requirements:
Proficiency in JavaScript (React) and Node.js.
Experience with RESTful APIs.
Strong understanding of database management (MySQL or MongoDB).
Excellent communication skills.
        """,
        """
Job Offer 2: Remote Data Analyst at Data Insights Corp.
Location: Remote
Salary: $70,000 - $85,000 per year
Description:
Data Insights Corp. is looking for a Remote Data Analyst to help us interpret data trends and provide actionable insights. The role requires proficiency in data analysis tools but does not require extensive programming knowledge. You will work closely with marketing teams to optimize campaigns based on data findings.
Requirements:
Experience with data visualization tools (Tableau or Power BI).
Basic knowledge of SQL.
Strong analytical skills.
Familiarity with programming languages is a plus but not essential.
        """,
        """
Job Offer 3: Remote Graphic Designer at Creative Minds Studio
Location: Remote
Salary: $50,000 - $65,000 per year
Description:
Creative Minds Studio is hiring a Remote Graphic Designer to create visually appealing designs for our clients. The ideal candidate should have experience in graphic design software such as Adobe Photoshop and Illustrator. This role does not involve any coding or software development tasks.
Requirements:
Proficiency in graphic design software (Photoshop, Illustrator).
Strong portfolio showcasing design work.
Ability to work independently and meet deadlines.
Excellent communication skills.
        """,
    ]
    jobs = pd.DataFrame(jobs, columns=["description"])
    # jobs = pd.DataFrame({"description": jobs})
    cv = """
Name: John Doe
Email: johndoe@example.com
Phone: +1 (555) 123-4567
Location: Remote (Based in San Francisco, CA)
LinkedIn: linkedin.com/in/johndoe
Summary
Detail-oriented and innovative Software Engineer with over 5 years of experience in full-stack development and a strong background in designing scalable web applications. Proficient in JavaScript, Python, and Java, with a passion for developing efficient code and optimizing user experiences. Seeking to leverage expertise in a challenging remote position.
Technical Skills
Languages: JavaScript, Python, Java, C++
Frameworks: React, Node.js, Django, Spring Boot
Databases: MySQL, MongoDB
Tools & Technologies: Git, Docker, AWS, RESTful APIs
Methodologies: Agile, Scrum
Professional Experience
Software Engineer
ABC Tech Solutions | Remote
June 2021 – Present
Developed and maintained scalable web applications using React and Node.js.
Collaborated with cross-functional teams to define project requirements and deliver high-quality software solutions.
Implemented RESTful APIs to enhance application functionality and performance.
Junior Software Developer
XYZ Innovations | San Francisco, CA
May 2019 – May 2021
Assisted in the development of e-commerce platforms using Django and PostgreSQL.
Participated in code reviews and contributed to the team’s knowledge base.
Wrote unit tests to ensure code quality and reliability.
Education
Bachelor of Science in Computer Science
University of California, Berkeley | Graduated: May 2019
Certifications
Certified AWS Solutions Architect – Associate
Full Stack Web Development Certificate (Coursera)
Projects
Personal Portfolio Website: Designed a responsive portfolio showcasing projects using HTML, CSS, JavaScript.
Task Management App: Developed a task management application with user authentication using Django and React.
    """
    return jobs, cv

def find_jobs(cv_path):
    generator = get_generator()
    # jobs, cv = get_example()
    cv = pdf_to_text(cv_path)
    jobs = get_jobs(generator)
    

    filtered_jobs = filter_jobs(jobs, cv, generator)

    save_path = "data/filtered_jobs/"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    for i, jo in enumerate(filtered_jobs):
        with open(Path(save_path) / f"job_offer_{i}.txt", "w+") as f:
            f.write(jo)

    return filtered_jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input a Curriculum Vitae file.")
    parser.add_argument(
        '-f', '--file', 
        type=str, 
        required=False, 
        help='Path to the Curriculum Vitae file (pdf)'
    )
    args = parser.parse_args()

    find_jobs(args.file)


