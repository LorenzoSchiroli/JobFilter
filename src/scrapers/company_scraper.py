"""
Company size scraper thrugh google search description.
"""

from googlesearch import search
import re
import time

from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.ollama import OllamaChatGenerator

def extract_first_number(s):
    digits = re.findall(r'\d', s)  # Find all individual digits
    return int(''.join(digits)) if digits else None

def description2employees_llm(text, generator):
    prompt = " \nExtract the number of employees (only one) and if it is present write it in the standard numeric format otherwise write None. Do not add comments."
    query = f'"{text}"' + prompt
    messages = [ChatMessage.from_user(query)]
    response = generator.run(messages)
    reply = response["replies"][0]
    number_text = reply.text
    number = extract_first_number(number_text)
    return number


def description2employees_regex(text):
    pattern = r'(\d+(-\d+)?)\s*(?=\bemployees\b)'
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1) if match else None

def get_company_size(company_name, generator):
    search_query = company_name + " number of emplyees company size"
    try:
        # time.sleep(3)  # try reduce search query frequecy
        results = search(search_query, advanced=True, num_results=1)
        description = next(results).description
    except Exception as e:
        print("Googlesearch not working")
        return None
    # description = "Nov 5, 2024 — As of FY 2024, the total number of employees had reached around 164 thousand (only counting full-time equivalent), up from 161 thousand recorded in the ..."
    # print(description)
    employees_number = description2employees_llm(description, generator)
    return employees_number


if __name__ == "__main__":
    generator = OllamaChatGenerator(model="llama3.2",
        url = "http://localhost:11434",
        generation_kwargs={
            "num_predict": 100,
            "temperature": 0.0,
        }
    )
    # Simple list of famous companies
    famous_companies = [
        "Apple",
        "Microsoft",
        "Nvidia",
        "Alphabet",
        "Amazon",
        "Saudi Aramco",
        "Exxon Mobil",
        "Tesla",
        "Berkshire Hathaway",
        "JPMorgan Chase",
        "Bank of America",
        "Eli Lilly",
        "UnitedHealth Group",
        "Meta Platforms",
        "Walmart"
    ]

    for company in famous_companies:
        print(f"{company}: {get_company_size(company, generator)}")


    """
    Nov 5, 2024 — As of FY 2024, the total number of employees had reached around 164 thousand (only counting full-time equivalent), up from 161 thousand recorded in the ...\nExtract the number of employees (only one) and if it is present write it in the standard numeric format otherwise write None. Do not add comments.
    """