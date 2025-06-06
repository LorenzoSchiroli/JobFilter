from jobspy import scrape_jobs

import pandas as pd
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack import Document
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.ollama import OllamaChatGenerator

import re
import PyPDF2

import time

from tqdm import tqdm
import copy
import argparse
from jobfilter.scrapers.company_scraper import get_company_size

from langdetect import detect

import json

from pathlib import Path

from dotenv import load_dotenv
import os

from typing import List

load_dotenv()

huggugface_token = os.environ.get("HUGGINGFACE_TOKEN")

# login(token=huggugface_token)

#%% search phase

def get_jobs(generator: OllamaChatGenerator) -> pd.DataFrame:
    """
    Scrape job postings from various platforms, enrich them with company size
    information, and return the resulting DataFrame.
    """
    jobs = scrape_jobs(
        site_name=["indeed", "linkedin", "zip_recruiter", "glassdoor", "google"],
        search_term="software engineer",
        google_search_term=(
            "software engineer jobs near Munich, BY since yesterday"
        ),
        location="Munich, germany",
        results_wanted=10,
        hours_old=72,
        country_indeed="Germany",
    )
    print(f"Found {len(jobs)} jobs")
    unique_companies = jobs["company"].unique()
    print("Number of companies:", len(unique_companies))
    mapping = {
        company: get_company_size(company, generator)
        for company in tqdm(unique_companies)
    }
    jobs["company_num_employees"] = jobs["company"].map(mapping)
    return jobs

#%% indexing + ai phase

def row2text(row: pd.Series, columns: List[str]) -> str:
    """
    Convert a DataFrame row into a formatted string using specified columns.
    """
    return "\n\n".join([f"{col.upper()}\n{row[col]}" for col in columns])

def df2text(jobs: pd.DataFrame) -> List[str]:
    """
    Convert a DataFrame of job postings into a list of formatted strings.
    """
    # convert to plain text
    jobs_strings = []
    for index, row in jobs.iterrows():
        row_string = row2text(row, jobs.columns)
        jobs_strings.append(row_string)
    return jobs_strings

def pdf_to_text(path: str) -> str:
    """
    Open a PDF file and return its plain text content.
    """

    with open(path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text

def get_generator() -> OllamaChatGenerator:
    """
    Initialize and return an instance of the OllamaChatGenerator.
    """
    generator = OllamaChatGenerator(
        model="llama3.2",
        url="http://localhost:11434",
        generation_kwargs={
            "num_predict": 200,
            "temperature": 0.5,
        },
    )
    return generator

def get_search_query(cv: str, generator: OllamaChatGenerator) -> None:
    """
    Generate a search query based on the provided CV using the generator.
    """
    search_query_request = (
        "Curriculum vitae: \n<text-begin>\n {cv} \n<text-end>\n "
        "Generate a search query to find a job for this worker "
        "(10 words maximum). Don't add comments."
    )
    start = time.time()
    messages = [
        ChatMessage.from_system(
            "\\nYou are a helpful, respectful and honest assistant"
        ),
        ChatMessage.from_user(search_query_request),
    ]
    response = generator.run(messages)
    reply_query = response["replies"][0]
    index_query = reply_query.text

    # force manual
    index_query = "data scientist remote german company"

    print(f"Inference time: {time.time() - start}")
    print(f"Total words {index_query.count(' ') + 1}")
    print("Reply:", index_query)

def filter_jobs(
    jobs_info: pd.DataFrame, cv: str, generator: OllamaChatGenerator
) -> List[str]:
    """
    Filter job postings based on the CV and generator, returning a list of matched
    job descriptions.
    """
    jobs_text = df2text(jobs_info)

    documents_all = [
        Document(content=jo, id=i + 1) for i, jo in enumerate(jobs_text)
    ]
    document_store = InMemoryDocumentStore()
    document_store.write_documents(documents_all)
    print("Documents count:", document_store.count_documents())
    retriever = InMemoryBM25Retriever(document_store=document_store)

    messages = [
        ChatMessage.from_system("\\nYou are a helpful, respectful and honest assistant")
    ]
    messages.append(
        ChatMessage.from_user(
            "Extract a search query form the CV to find the best suited job, use few "
            "keywords (10 maximum). Just write the query without comments. The CV: \n"
            + cv
        )
    )
    response = generator.run(messages)
    index_query = response["replies"][0].text

    documents_candidate = retriever.run(query=index_query)["documents"]

    matched_jobs = []
    for document in tqdm(documents_candidate):
        is_match = job_match(
            generator, messages, jobs_info.iloc[document.id - 1], document.content
        )
        if is_match:
            matched_jobs.append(document.content)

    return matched_jobs

def is_full_time(job_offer_text: str) -> bool:
    """
    Determine if a job offer is for a full-time position based on its text.
    """
    full_time_keywords = ["full-time", "full time", "permanent", "regular", "fulltime"]
    part_time_keywords = [
        "part-time", "part time", "parttime", "freelance", "hourly", "internship"
    ]
    job_offer_text = job_offer_text.lower()
    found_full_time = any(
        full_time_keyword in job_offer_text for full_time_keyword in full_time_keywords
    )
    found_part_time = any(
        part_time_keyword in job_offer_text for part_time_keyword in part_time_keywords
    )
    return found_full_time or (not found_full_time and not found_part_time)

def rule_based_filtering(job_info: pd.Series) -> bool:
    """
    Apply rule-based filtering to a job posting to determine if it meets predefined
    criteria.
    """
    checks = {}
    job_text = job_info["description"].lower()
    checks["junior_mid"] = "senior" not in job_text
    checks["english_text"] = detect(job_info["description"]) == "en"
    checks["company_established"] = "startup" not in job_text
    checks["full_time"] = is_full_time(job_text)
    checks["remote"] = any(
        keyword in job_text for keyword in ["remote", "home", "teleworking"]
    )
    checks["company_size"] = (
        pd.isna(job_info["company_num_employees"])
        or job_info["company_num_employees"] > 100
    )
    checks["permament_position"] = not any(
        keyword in job_text for keyword in ["internship", "trainee", "fellowship"]
    )
    checks["mlops"] = any(
        keyword in job_text
        for keyword in ["mlops", "AWS", "cloud", "mlflow", "CI/CD", "deploy"]
    )
    return all(checks.values())

def llm_based_filtering(
    generator: OllamaChatGenerator, messages: List[ChatMessage], job_text: str
) -> bool:
    """
    Use an LLM-based approach to filter job postings based on predefined conditions.
    """
    match_request = (
        f"Job Offer: \n\n<text-begin>\n {job_text} \n<text-end>\n\n"
        "The following list are job offer conditions, check each condition if it is "
        "True or False. If a condition is unclear or not specified consider it as "
        "None.\nJob offer conditions:\n"
        '- "junior_mid": check if the job offer\'s role is a junior or mid-level role '
        "and not an advanced senior role.\n"
        '- "english_text": check if the description is written in english.\n'
        '- "company_established": check if the job offer\'s company is an established '
        "company (not a startup).\n"
        '- "full_time": check if the job is a full-time job or has the option to work '
        "full-time.\n"
        '- "remote": check if the job is truly fully remote without specifications '
        "regarding office returns.\n"
        '- "company_size": check if the job offer\'s company has more than 100 '
        "employees.\n"
        '- "permament_position": check if the job is a normal job and not an '
        "internship.\n"
        "Moreover, check if the curriculum vitae role ambitions match with the the "
        "job offer role. Answer true or false or null.\n\n"
        "Don't explain details, write output as a json file.\nOutput example:\n\n"
        "{\n"
        '    "junior_mid": null,\n'
        '    "english_text": null,\n'
        '    "company_established": null,\n'
        '    "full_time": null,\n'
        '    "remote": null,\n'
        '    "company_size": null,\n'
        '    "permament_position": null,\n'
        '    "match": null\n'
        "}"
    )

    messages_side = copy.deepcopy(messages)
    messages_side.append(ChatMessage.from_user(match_request))
    response = generator.run(messages_side)
    reply_match = response["replies"][0].text
    try:
        reply_match = reply_match.replace("\n", "")
        match = re.search(r"(\{.*?\})", reply_match)
        json_text = str(match.group(1) if match else {})
        reply_match = json.loads(json_text)
    except Exception:
        reply_match = {}

    print(reply_match)

    return all(reply_match.values())

def job_match(
    generator: OllamaChatGenerator,
    messages: List[ChatMessage],
    job_info: pd.Series,
    job_text: str,
) -> bool:
    """
    Determine if a job posting matches the candidate's profile using both rule-based
    and LLM-based filtering.
    """
    if pd.isna(job_info["description"]):
        return False

    check_rules = rule_based_filtering(job_info)

    if not check_rules:
        return False

    check_llm = llm_based_filtering(generator, messages, job_text)

    return check_llm


def get_example() -> tuple[pd.DataFrame, str]:
    """
    Return example job postings and a sample CV for testing purposes.
    """
    jobs = [
    """
Job Offer 1: Remote Full Stack Developer at Tech Innovators Inc.
Location: Remote
Salary: $90,000 - $110,000 per year
Description:
Tech Innovators Inc. is seeking a talented Full Stack Developer to
join our dynamic team. The ideal candidate will have experience
with JavaScript frameworks (React preferred), Node.js for backend
development, and familiarity with AWS services. You will work on
exciting projects that involve building scalable applications and
collaborating with product managers and designers.
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
Data Insights Corp. is looking for a Remote Data Analyst to help
 us interpret data trends and provide actionable insights.
 The role requires proficiency in data analysis tools but does
 not require extensive programming knowledge. You will work closely
 with marketing teams to optimize campaigns based on data findings.
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
Creative Minds Studio is hiring a Remote Graphic Designer to create
visually appealing designs for our clients. The ideal candidate
should have experience in graphic design software such as Adobe
Photoshop and Illustrator. This role does not involve any coding
or software development tasks.
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
Detail-oriented and innovative Software Engineer with over 5 years of experience
in full-stack development and a strong background in designing scalable web
applications. Proficient in JavaScript, Python, and Java, with a passion for
developing efficient code and optimizing user experiences. Seeking to leverage
expertise in a challenging remote position.
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
Collaborated with cross-functional teams to define project requirements and
deliver high-quality software solutions.
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
Personal Portfolio Website: Designed a responsive portfolio showcasing projects
using HTML, CSS, JavaScript.
Task Management App: Developed a task management application with user
authentication using Django and React.
    """
    return jobs, cv

def find_jobs(cv_path: str) -> List[str]:
    """
    Find and filter job postings based on the provided CV file path.
    """
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
        "-f", "--file",
        type=str,
        required=False,
        help="Path to the Curriculum Vitae file (pdf)"
    )
    args = parser.parse_args()

    find_jobs(args.file)


