from backend.pdf_reader import py_reader
from backend.text_splitter import text_splitter
from backend.embeddings import get_embeddings
from backend.search import get_search_result
from backend.llm_explainer import explain_legal_question, reason_about_scenario
from backend.vector_store import store_embeddings,search_embeddings

def main():
    try:
        print("Loading document...")

        # Load and process document
        full_text = py_reader("data/legal/PL.pdf")
        chunks = text_splitter(full_text)
        embeddings = get_embeddings(chunks)
       #store_embeddings(chunks)

        # Get user input
        mode = input("\nChoose mode:\n1. Normal explanation\n2. Scenario reasoning\n> ")
        question = input("\nAsk your legal question: ")
        best_chunk = get_search_result(chunks, embeddings, question)

       #best_chunk =search_embeddings(question)

        
        if mode == "2":
            final_answer = reason_about_scenario(question, best_chunk)
            
        else:
            final_answer = explain_legal_question(question, best_chunk)
        print("AI's Explanation:")
        print(final_answer)
        
        print("\n--- Answer Complete ---")
    except FileNotFoundError:
        print("PDF file not found. Please ensure the file path is correct.")

    except Exception as e:
        print("An error occured:",(e))
     

if __name__ == "__main__":
    main()