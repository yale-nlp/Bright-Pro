QUERY_TEMPLATE = """
You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search and get_document tools provided. Please perform reasoning and use the tools step by step, in an interleaved manner. You may use the search and get_document tools multiple times.

Question: {Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer. For this explanation section only, you should cite your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. For example, [20].}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

# QUERY_TEMPLATE_NO_GET_DOCUMENT = """
# You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search tool provided. Please perform reasoning and use the tool step by step, in an interleaved manner. You may use the search tool multiple times.

# Question: {Question}

# Your response should be in the following format:
# Explanation: {{your explanation for your final answer. For this explanation section only, you should cite your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. For example, [20].}}
# Exact Answer: {{your succinct, final answer}}
# Confidence: {{your confidence score between 0% and 100% for your answer}}
# """.strip()

# Deep Research Main Prompt — paper Figure 3 (appendix/prompt.tex). Kept
# verbatim with the figure used in the paper.
QUERY_TEMPLATE_NO_GET_DOCUMENT = """
Question: {Question}

You are a research agent. Your task is to answer the question by actively using the provided Search Tool.

Use the search tool iteratively for many turns. But in each turn, you should only use the search tool once.

Refine your queries based on previous results to broaden coverage and fill knowledge gaps.

Stop searching only once you have gathered a comprehensive and multi-perspective set of evidence.

Your final response must integrate information from different angles, supported by multiple sources. You must base your answer solely on the retrieved evidence documents—do not use any prior or external knowledge.

Your final response should be in the following format:
Answer: {{Your final answer. You should cite your evidence documents inline by enclosing their docids in square brackets at the end of sentences. For example, [20].}}
Confidence: {{Your confidence score between 0% and 100% for your answer}}
""".strip()

QUERY_TEMPLATE_NO_GET_DOCUMENT_NO_CITATION = """
You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search tool provided. Please perform reasoning and use the tool step by step, in an interleaved manner. You may use the search tool multiple times.

Question: {Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

# Fixed Round Response Generation Prompt — paper Figure 4. Used by
# openai_fixed_turn / answers_from_runs to condition the model on retrieved
# evidence for a single answer pass (no interactive search tool calls).
QUERY_TEMPLATE_ORACLE = """
I will give you a question and a set of evidence documents, which contains helpful information to answer the question.

Question: {Question}

Evidence documents: {EvidenceDocuments}

Your final response must integrate information from different angles, supported by multiple sources. You must base your answer solely on the retrieved evidence documents—do not use any prior or external knowledge.

Your final response should be in the following format:
Answer: {{Your final answer. You should cite your evidence documents inline by enclosing their docids in square brackets at the end of sentences. For example, [20].}}
Confidence: {{Your confidence score between 0% and 100% for your answer}}
""".strip()

def format_query(query: str, query_template: str | None = None) -> str:
    """Format the query with the specified template if provided."""
    if query_template is None:
        return query
    elif query_template == "QUERY_TEMPLATE":
        return QUERY_TEMPLATE.format(Question=query)
    elif query_template == "QUERY_TEMPLATE_NO_GET_DOCUMENT":
        return QUERY_TEMPLATE_NO_GET_DOCUMENT.format(Question=query)
    elif query_template == "QUERY_TEMPLATE_NO_GET_DOCUMENT_NO_CITATION":
        return QUERY_TEMPLATE_NO_GET_DOCUMENT_NO_CITATION.format(Question=query)
    elif query_template == "QUERY_TEMPLATE_ORACLE":
        return QUERY_TEMPLATE_ORACLE.format(Question=query)
    else:
        raise ValueError(f"Unknown query template: {query_template}")
