# import sympy as sp
# class chatbot:
#     def __init__(self,name="ElectroBot"):
#         self.name=name
#         self.intro_message()

#     def intro_message(self):
#         print(f"Hi!I am {self.name},your electrical engineering assistant. Let's chat and solve problem together")
#     def respond(self,user_input):
#         if "ohms" in user_input.lower():
#             return self.ohms_law_help()
#         elif "power" in user_input.lower():
#             return self.power_calculation_help()
#         elif "inductance" in user_input.lower():
#             return self.inductance_formula_help()
#         elif "help" in user_input.lower():
#             return self.show_help()
#         else:
#             return "I am your personal chat bot and I am here to help you solve basic electrical engineering problems."
#     def ohms_law_help(self):
#         print("Ohm's Law: V = I * R (Voltage = Current × Resistance)")
#         V = input("Enter Voltage (V) or press Enter if unknown: ")
#         I = input("Enter Current (I) or press Enter if unknown: ")
#         R = input("Enter Resistance (R) or press Enter if unknown: ")

#         # Solve for missing value
#         try:
#             if V == "":
#                 V = float(I) * float(R)
#                 return f"Calculated Voltage (V) is: {V} volts."
#             elif I == "":
#                 I = float(V) / float(R)
#                 return f"Calculated Current (I) is: {I} amperes."
#             elif R == "":
#                 R = float(V) / float(I)
#                 return f"Calculated Resistance (R) is: {R} ohms."
#             else:
#                 return "All values are provided. No calculation needed."
#         except ValueError:
#             return "Please enter valid numerical values."

#     def power_calculation_help(self):
#         print("Power Calculation: P = V * I (Power = Voltage × Current)")
#         V = input("Enter Voltage (V): ")
#         I = input("Enter Current (I): ")
        
#         try:
#             P = float(V) * float(I)
#             return f"Calculated Power (P) is: {P} watts."
#         except ValueError:
#             return "Please enter valid numerical values."

#     def show_help(self):
#         return """
#         Here are some things I can help with:
#         - Ohm's Law calculations
#         - Power calculations
#         - Inductance formula guidance
#         Type a related question to get started!
#         """

# # Start the chatbot
# def main():
#     bot = chatbot()
#     print("Type 'exit' to end the chat And type article to search for articles")
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == "exit":
#             print(f"{bot.name}: Goodbye! Have a great day!")
#             break
#         response = bot.respond(user_input)
#         print(f"{bot.name}: {response}")

# if __name__ == "__main__":
#     main()

# from sentence_transformers import SentenceTransformer, util

# # Initialize the SentenceTransformer model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Sample articles
# sample_articles = [
#     "Ohm's Law states that Voltage equals Current times Resistance.",
#     "Inductance measures the property of an electrical conductor to induce an electromotive force.",
#     "Transformer cores are made of ferromagnetic materials for better efficiency."
# ]

# def find_relevant_article(query, articles):
#     # Encode the query and articles
#     query_embedding = model.encode(query, convert_to_tensor=True)
#     article_embeddings = model.encode(articles, convert_to_tensor=True)
    
#     # Compute cosine similarities
#     similarities = util.cos_sim(query_embedding, article_embeddings)
    
#     # Find the index of the most similar article
#     best_match_index = similarities.argmax().item()
#     return articles[best_match_index]

# # Main program
# def main():
#     print("Welcome to the PyTorch Article Finder!")
#     query = input("Enter your query: ")
#     best_article = find_relevant_article(query, sample_articles)
#     print("\n=== Most Relevant Article ===\n")
#     print(best_article)

# if __name__ == "__main__":
#     main()

import requests
from sentence_transformers import SentenceTransformer, util
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Initialize the SentenceTransformer model
# model = SentenceTransformer('all-MiniLM-L6-v2')
model_name = "t5-small"  # You can also try "t5-base" or "t5-large" for better quality
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


# IEEE Xplore API configuration
API_KEY = "55anpjues3nqsee5jgjaycwq"
API_URL = "http://ieeexploreapi.ieee.org/api/v1/search/articles"

# Function to fetch articles from IEEE Xplore
def fetch_ieee_articles(query, max_results=10):
    params = {
        "apikey": API_KEY,
        "querytext": query,
        "format": "json",
        "max_records": max_results
    }
    response = requests.get(API_URL, params=params)

    if response.status_code == 200:
        data = response.json()
        articles = [
            {"title": article.get("title", "No title"), "abstract": article.get("abstract", "No abstract available")}
            for article in data.get("articles", [])
        ]
        return articles
    else:
        print(f"Failed to fetch data. HTTP Status Code: {response.status_code}")
        return []

# Function to process articles and find solutions
def generate_solution(query, articles):
    # Combine titles and abstracts for similarity matching
    article_texts = [f"{a['title']} - {a['abstract']}" for a in articles]
    
    if not article_texts:
        return "No relevant articles were found. Please try another query."

    # Encode the query and articles
    query_embedding = model.encode(query, convert_to_tensor=True)
    article_embeddings = model.encode(article_texts, convert_to_tensor=True)

    # Compute cosine similarities
    similarities = util.cos_sim(query_embedding, article_embeddings)

    # Find the most relevant article
    best_match_index = similarities.argmax().item()
    best_article = articles[best_match_index]

    # Generate a solution by summarizing and interpreting the best article
    solution = f"Based on the article '{best_article['title']}', here’s what you can do:\n\n"
    solution += summarize_abstract(best_article['abstract'])
    return solution

def summarize_with_t5(text, max_length=50):
    # Prepare the input
    input_text = f"summarize: {text}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True)

    # Generate the summary
    summary_ids = model.generate(inputs, max_length=max_length, min_length=10, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to summarize the abstract (mock function for now)
def summarize_abstract(abstract):
    try:
        return summarize_with_t5(abstract)
    except Exception as e:
        return f"Error during summarization: {e}"


# Main program
def main():
    print("Welcome to the Intelligent Assistant!")
    query = input("Enter your query or problem statement: ")
    
    print("\nFetching articles from IEEE Xplore...")
    articles = fetch_ieee_articles(query)
    
    if not articles:
        print("No articles found for your query.")
        return

    print("\nProcessing retrieved data to generate a solution...")
    solution = generate_solution(query, articles)
    print("\n=== Solution ===\n")
    print(solution)

if __name__ == "__main__":
    main()
