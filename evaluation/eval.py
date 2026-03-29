import json
import numpy as np
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric
)
from deepeval.models import OllamaModel

from backend.pdf_reader import py_reader
from backend.text_splitter import text_splitter
from backend.embeddings import get_embeddings
from backend.search import get_search_result
from backend.llm_explainer import explain_legal_question


def run_rag_evaluation():

    ollama_model = OllamaModel(
        model="mistral",
        base_url="http://localhost:11434"
    )

    # async_mode=False forces synchronous metric calls
    faithfulness      = FaithfulnessMetric(threshold=0.7,  model=ollama_model, async_mode=False)
    answer_relevancy  = AnswerRelevancyMetric(threshold=0.7, model=ollama_model, async_mode=False)
    context_precision = ContextualPrecisionMetric(threshold=0.7, model=ollama_model, async_mode=False)

    with open("evaluation/data.json", "r") as f:
        eval_data = json.load(f)

    # Build RAG index once
    text = py_reader("data/legal/PL.pdf")
    chunks = text_splitter(text)
    embeddings = get_embeddings(chunks)

    f_scores, ar_scores, cp_scores = [], [], []

    print(f"\n{'='*60}")
    print(f"Evaluating {len(eval_data)} questions sequentially...")
    print(f"{'='*60}\n")

    for i, item in enumerate(eval_data):
        question     = item["question"]
        ground_truth = item["ground_truth"]

        best_chunk = get_search_result(chunks, embeddings, question)
        answer     = explain_legal_question(question, best_chunk)

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=[best_chunk],
            expected_output=ground_truth
        )

        print(f"Q{i+1}: {question}")

        # Call .measure() directly — avoids DeepEval's async evaluate() runner
        faithfulness.measure(test_case)
        answer_relevancy.measure(test_case)
        context_precision.measure(test_case)

        print(f"  Faithfulness:      {faithfulness.score:.2f}  — {faithfulness.reason}")
        print(f"  Answer Relevancy:  {answer_relevancy.score:.2f}  — {answer_relevancy.reason}")
        print(f"  Context Precision: {context_precision.score:.2f}  — {context_precision.reason}")
        print()

        f_scores.append(faithfulness.score)
        ar_scores.append(answer_relevancy.score)
        cp_scores.append(context_precision.score)

    print(f"{'='*60}")
    print(f"FINAL AVERAGE SCORES ({len(eval_data)} questions)")
    print(f"{'='*60}")
    print(f"  Faithfulness:      {np.mean(f_scores):.3f}")
    print(f"  Answer Relevancy:  {np.mean(ar_scores):.3f}")
    print(f"  Context Precision: {np.mean(cp_scores):.3f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_rag_evaluation()