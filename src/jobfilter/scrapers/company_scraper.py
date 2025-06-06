"""
Company size scraper through Google search description.
"""

from googlesearch import search
import re

from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from typing import Optional


def extract_first_number(s: str) -> Optional[int]:
    """
    Extract the first number from a string. If no number is found, return None.
    """
    digits = re.findall(r"\d", s)  # Find all individual digits
    return int("".join(digits)) if digits else None


def description2employees_llm(
    text: str, generator: OllamaChatGenerator
) -> Optional[int]:
    """
    Use an LLM to extract the number of employees from a text description.
    """
    prompt = (
        " \nExtract the number of employees (only one) and if it is present write it "
        "in the standard numeric format otherwise write None. Do not add comments."
    )
    query = f'"{text}"' + prompt
    messages = [ChatMessage.from_user(query)]
    response = generator.run(messages)
    reply = response["replies"][0]
    number_text = reply.text
    number = extract_first_number(number_text)
    return number


def description2employees_regex(text: str) -> Optional[str]:
    """
    Use a regex pattern to extract the number of employees from a text description.
    """
    pattern = r"(\d+(-\d+)?)\s*(?=\bemployees\b)"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1) if match else None


def get_company_size(
    company_name: str, generator: OllamaChatGenerator
) -> Optional[int]:
    """
    Retrieve the company size (number of employees) by performing a Google search
    and extracting the relevant information using an LLM.
    """
    search_query = company_name + " number of employees company size"
    try:
        results = search(search_query, advanced=True, num_results=1)
        description = next(results).description
    except Exception:
        print("Googlesearch not working")
        return None
    employees_number = description2employees_llm(description, generator)
    return employees_number


if __name__ == "__main__":
    """
    Test the company size scraper with a list of famous companies.
    """
    generator = OllamaChatGenerator(
        model="llama3.2",
        url="http://localhost:11434",
        generation_kwargs={
            "num_predict": 100,
            "temperature": 0.0,
        },
    )
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
        "Walmart",
    ]

    for company in famous_companies:
        print(f"{company}: {get_company_size(company, generator)}")
