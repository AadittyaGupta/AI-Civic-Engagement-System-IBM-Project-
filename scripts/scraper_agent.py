
import os
import json
import time
import requests 
from bs4 import BeautifulSoup
import google.generativeai as genai
import pandas as pd

try:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        GOOGLE_API_KEY = input("Please enter your Google Gemini API Key: ")
    genai.configure(api_key=GOOGLE_API_KEY)
    llm = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    print(f"Error configuring Google AI: {e}")
    exit()

# Static Web Scrapper Function 
def get_page_data_static(url: str) -> dict:
    """Fetches text AND title from a static URL."""
    try:
        # 
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract the page title
        page_title = soup.title.string if soup.title else "No Title Found"
        
        # Remove scripts, styles, and irrelevant sections
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
        
        # Get the visible text
        text = soup.get_text(separator='\n', strip=True)

        # Collapse multiple newlines into one
        print(f"STATIC SCRAPE OK: {url}. Title: '{page_title}'. Length: {len(text)}.")
        return {"text": text, "title": page_title}
        
    # Handle HTTP and request errors    
    except requests.exceptions.RequestException as e:
        print(f"Error during static scrape of {url}: {e}")
        return None


EXTRACTION_PROMPT_TEMPLATE = """
You are an expert data extraction AI. Your task is to read the unstructured text from a government scheme webpage and extract the specified information
into a clean JSON object.

Follow these rules strictly:
1.  Extract the information for the fields listed in the desired JSON format below.
2.  If you cannot find an official scheme name in the text, use a concise version of the webpage's title for the `scheme_name`.
3.  For text fields like `description`, `eligibility_criteria`, etc., if you cannot find the information, you MUST use the value "Information not found".
4.  **CRITICAL RULE FOR NUMBERS:** For numeric fields (`min_age`, `max_age`, `min_income`, `max_income`), if the information is not found, you MUST use a 
default integer value.
    - For `min_age` and `min_income`, use `0`.
    - For `max_age`, use `100`.
    - For `max_income`, use `5000000`.
    - DO NOT write text like 'Information not found' in these number fields.
5.  Your entire response MUST be only the JSON object, with no other text before or after it.

**Desired JSON Format:**
{{
  "scheme_name": "string",
  "description": "string",
  "category": "string",
  "target_state": "string",
  "min_age": "integer",
  "max_age": "integer",
  "min_income": "integer",
  "max_income": "integer",
  "target_gender": "string",
  "eligibility_criteria": "string",
  "documents_required": "string",
  "application_steps": "string"
}}

---
**Unstructured Web Page Text to Analyze:**
{web_page_text}
---
"""


def main():
    """Main function to run the scraping and extraction pipeline."""
    
    target_urls = [
      
        "https://www.india.gov.in/spotlight/ayushman-bharat-pradhan-mantri-jan-arogya-yojana",
        
        
        "https://pib.gov.in/PressReleasePage.aspx?PRID=1983842",
        
        
        "https://pib.gov.in/PressReleaseIframePage.aspx?PRID=1914233",
        
       
        "https://pib.gov.in/PressReleaseIframePage.aspx?PRID=1997395",
        
      
        "https://pib.gov.in/PressReleaseIframePage.aspx?PRID=1900133"
    ]
    
    all_extracted_schemes = []
    
    for url in target_urls:
        print("-" * 50)
        time.sleep(5) #
        page_data = get_page_data_static(url)
        
        if page_data and len(page_data["text"]) > 100:
            prompt = EXTRACTION_PROMPT_TEMPLATE.format(
                page_title=page_data["title"],
                url=url,
                web_page_text=page_data["text"]
            )
            
            print("Sending enhanced context to AI for information extraction...")
            try:
                response = llm.generate_content(prompt)
                json_string = response.text.strip().replace("```json", "").replace("```", "").strip()
                scheme_data = json.loads(json_string)
                scheme_data['official_link'] = url
                
                all_extracted_schemes.append(scheme_data)
                print(f"Successfully extracted data for: {scheme_data.get('scheme_name')}")
                
            except (json.JSONDecodeError, Exception) as e:
                print(f"Could not parse AI response for URL {url}. Error: {e}")
        else:
            print(f"Skipping AI extraction for {url} due to insufficient content.")

    if not all_extracted_schemes:
        print("No new schemes were extracted. Exiting.")
        return

    output_filename = 'schemes.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_extracted_schemes, f, indent=4, ensure_ascii=False)
    
    print("-" * 50)
    print(f"Successfully saved {len(all_extracted_schemes)} schemes to {output_filename}")
    
    print("\nLoading the new data into a Pandas DataFrame:")
    df = pd.read_json(output_filename)
    print(df.info())

if __name__ == "__main__":
    main()