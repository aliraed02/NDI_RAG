from langchain.load import dumps, loads
from sentence_transformers import CrossEncoder



cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def get_unique_union(documents: list[list]):
    """
    Get the unique union of retrieved documents.
    
    Args:
        documents (list[list]): List of lists of retrieved documents.
    
    Returns:
        list: Unique documents.
    """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

def get_unique_contents(documents):
    """
    Get the unique contents of retrieved documents.
    
    Args:
        documents (list[list]): List of lists of retrieved documents.
    
    Returns:
        list: Unique contents.
    """
    # Flatten list of lists
    flattened_docs = [doc.page_content for doc in documents]
    # Get unique contents
    unique_contents = list(set(flattened_docs))
    # Return
    return unique_contents


def get_pairs(question, documents):
    pairs= []
    for doc in documents:
        pairs.append([question, doc])
    scores = cross_encoder.predict(pairs)
    scored_docs = zip(scores, documents)
    sorted_docs = sorted(scored_docs, reverse=True)
    best_docs = sorted_docs[:int(len(sorted_docs) * 0.9)]
    reRanked_docs = [doc for _, doc in best_docs]

    return reRanked_docs
    