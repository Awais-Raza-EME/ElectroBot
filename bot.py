import requests
import re
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os

# Initialize models
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
t5_model_name = "t5-small"
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)

# Fetch articles from arXiv
def fetch_arxiv_articles(query, max_results=10):
    url = f'http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.content.decode('utf-8')
        articles = []
        for entry in data.split('<entry>')[1:]:  # Extract each article entry
            title_start = entry.find('<title>') + len('<title>')
            title_end = entry.find('</title>')
            title = entry[title_start:title_end].strip()
            
            summary_start = entry.find('<summary>') + len('<summary>')
            summary_end = entry.find('</summary>')
            summary = entry[summary_start:summary_end].strip()
            
            pdf_url_start = entry.find('<link title="pdf" href="') + len('<link title="pdf" href="')
            pdf_url_end = entry.find('"', pdf_url_start)
            pdf_url = entry[pdf_url_start:pdf_url_end]
            
            articles.append({"title": title, "summary": summary, "pdf_url": pdf_url})
        return articles
    else:
        print("Failed to fetch data from arXiv.")
        return []

# Download and extract text from PDF
def download_and_extract_pdf(pdf_url):
    try:
        response = requests.get(pdf_url, stream=True)
        if response.status_code == 200:
            local_path = "temp.pdf"
            with open(local_path, 'wb') as f:
                f.write(response.content)
            
            text = extract_text_from_pdf(local_path)
            os.remove(local_path)  # Clean up the temporary file
            return text
        else:
            print(f"Failed to download PDF from {pdf_url}.")
            return None
    except Exception as e:
        print(f"Error downloading or extracting PDF: {e}")
        return None

# Clean text by removing unnecessary spaces, URLs, and patterns
def clean_text(text):
    text = " ".join(text.split())
    lines = text.split('\n')
    filtered_lines = [line for line in lines if not line.startswith("Page") and not line.startswith("http")]
    return " ".join(filtered_lines)

# Extract text from a local PDF
def extract_text_from_pdf(pdf_url):
    try:
        doc = fitz.open(pdf_url)
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            text += page_text

        cleaned_text = clean_text(text)
        return cleaned_text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

# Summarize article text using T5 model
# Adjusted Summarize function with balanced length and better truncation handling
def summarize_text_with_t5(text):
    input_text = f"summarize: {text}"
    inputs = t5_tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
    summary_ids = t5_model.generate(inputs, max_length=200, min_length=80, length_penalty=1.5, num_beams=4, early_stopping=True)
    summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Generate solution by matching the query with the most relevant article
def generate_solution(query, articles):
    texts = [f"{a['title']} - {a['summary']}" for a in articles]
    if not texts:
        return "No relevant articles found."

    # Encode query and articles
    query_embedding = sentence_model.encode(query, convert_to_tensor=True)
    article_embeddings = sentence_model.encode(texts, convert_to_tensor=True)

    # Find the most relevant article
    similarities = util.cos_sim(query_embedding, article_embeddings)
    best_match_index = similarities.argmax().item()
    best_article = articles[best_match_index]

    # Download and process the PDF content
    if "pdf_url" in best_article and best_article["pdf_url"]:
        print(f"\nDownloading and extracting text from: {best_article['pdf_url']}")
        full_text = download_and_extract_pdf(best_article["pdf_url"])
    else:
        full_text = best_article["summary"]

    if not full_text:
        return "Unable to retrieve the full text of the selected article."

    # Summarize the full text of the article
    summary = summarize_text_with_t5(full_text)
    return f"Based on the article titled '{best_article['title']}', here’s what you can do:\n\n{summary}"

# Main program to interact with the user and provide solutions
def main():
    print("Welcome to the Intelligent Assistant!")
    query = input("Enter your query or problem statement: ")
    
    print("\nFetching articles from arXiv...")
    articles = fetch_arxiv_articles(query)

    if not articles:
        print("No articles found for your query.")
        return

    print("\nProcessing retrieved data to generate a solution...")
    solution = generate_solution(query, articles)
    print("\n=== Solution ===\n")
    print(solution)

if __name__ == "__main__":
    main()
