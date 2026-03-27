from langchain_google_genai import ChatGoogleGenerativeAI
import os
import traceback
from dotenv import load_dotenv
from rag_retriever import rag_retriever

load_dotenv()

gemini_api_key=os.getenv("GEMINI_API_KEY")  

try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=gemini_api_key,
        max_output_tokens=200
    )

except Exception as e:
    print("Error:")
    traceback.print_exc()
    

def rag_simple(query, retriever, llm, top_k=3):

    # Retrieve the context
    results = retriever.retrieve(query, top_k = top_k)

    context = "\n\n".join([doc['content'] for doc in results]) if results else ""

    # ans using gemini

    if context:
        prompt = f"""
            You are an AI assistant. Answer the question using ONLY the given context.
            If the answer is not present, say you don't have enough information.

            Context:
            {context}

            Question:
            {query}

            Answer:
            """
        response = llm.invoke([prompt])
        return f"[From Docs]\n{response.content}"
    
    else:
        # fallback to LLM
        prompt = f"""
            You are an AI assistant. Answer the question using your own knowledge.

            Question:
            {query}

            Answer:
            """
        response = llm.invoke([prompt])
        return f"[LLM Answer - No Docs]\n{response.content}"


def rag_advanced(query, retriever, llm, top_k=5, min_score=0.2, return_context=False):
    
    # Retrieve results (fixed typo + consistent API assumption)
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)

    if not results:
        return {
            'answer': "No relevant context found.",
            'sources': [],
            'confidence': 0.0,
            'context': "" if return_context else None
        }

    # Build context
    context = "\n\n".join([doc['content'] for doc in results])

    # Extract sources
    sources = [{
        'source': doc.get('metadata', {}).get('source_file',
                  doc.get('metadata', {}).get('source', 'unknown')),
        'page': doc.get('metadata', {}).get('page', 'unknown'),
        'score': doc.get('similarity_score', 0.0),
        'preview': doc['content'][:120] + '...'
    } for doc in results]

    # Confidence (simple but consistent)
    confidence = max([doc.get('similarity_score', 0.0) for doc in results])

    # Prompt (fixed formatting issue)
    prompt = f"""
        You are an AI assistant. Answer the question using ONLY the given context.
        If the answer is not present, say you don't have enough information.

        Context:
        {context}

        Question:
        {query}

        Answer:
        """

    response = llm.invoke([prompt])

    # Final output
    output = {
        'answer': response.content,
        'sources': sources,
        'confidence': confidence
    }

    if return_context:
        output['context'] = context

    return output

# Example usage:
# result = rag_advanced("What is Python Programming Language ?", rag_retriever, llm, top_k=3, min_score=0.1, return_context=True)
# print("Answer:", result['answer'])
# print("Sources:", result['sources'])
# print("Confidence:", result['confidence'])
# print("Context Preview:", result['context'][:300])

# answer = rag_simple("Run Python Scripts from the Python IDLE", rag_retriever, llm)
# print(answer)
