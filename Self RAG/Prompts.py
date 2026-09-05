from langchain_core.prompts import ChatPromptTemplate

decide_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You decide whether retrieval is needed."
            "Return JSON with key: should_retrieve ."
            "Guidelines:"
            "should_retrieve=True if answering requires specific facts from company documents."
            "should_retrieve=False for general explanations/definitions."
        ),
        ("human", "Question: {question}")
    ]
)

direct_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer using only your general knowledge."
        ),
        ("human", "{question}")
    ]
)


is_relevant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are judging document relevance at a TOPIC level."
            "Return JSON matching the schema."
            "A document is relevant if it discusses the same entity or topic area as the question."
            "It does NOT need to contain the exact answer."
            "Examples:"
            "- HR policies are relevant to questions about notice period, probation, termination, benefits."
            "- Pricing documents are relevant to questions about refunds, trials, billing terms."
            "- Company profile is relevant to questions about leadership, culture, size, or strategy."
            "Do NOT decide whether the document fully answers the question."
            "That will be checked later by IsSUP."
            "When unsure, return is_relevant=true."
        ),
        ("human", "Question:\n{question}\nDocument:\n{document}"),
    ]
)


rag_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a business rag chatbot."
            "You will receive a CONTEXT block from internal company documents."
            "Task:"
            "Answer the question based on the context"
            "Dont mention that you are getting a context in your answer"
        ),
        ("human", "Question:\n{question}\nContext:\n{context}"),
    ]
)


issup_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are verifying whether the ANSWER is supported by the CONTEXT."
            "Return JSON with keys: issup, evidence."
            "issup must be one of: fully_supported, partially_supported, no_support."
            "How to decide issup:"
            "fully_supported:"
            "  Every meaningful claim is explicitly supported by CONTEXT, and the ANSWER does NOT introduce"
            "  any qualitative/interpretive words that are not present in CONTEXT."
            "  (Examples of disallowed words unless present in CONTEXT: culture, generous, robust, designed to,"
            "  supports professional development, best-in-class, employee-first, etc.)"
            "partially_supported:"
            "  The core facts are supported, BUT the ANSWER includes ANY abstraction, interpretation, or qualitative"
            "  phrasing not explicitly stated in CONTEXT (e.g., calling policies 'culture', saying leave is 'generous',"
            "  or inferring outcomes like 'supports professional development')."
            "no_support:"
            "  The key claims are not supported by CONTEXT."
            "Rules:"
            "Be strict: if you see ANY unsupported qualitative/interpretive phrasing, choose partially_supported."
            "If the answer is mostly unrelated to the question or unsupported, choose no_support."

        ),
        (
            "human",
            "Question:\n{question}\n"
            "Answer:\n{answer}\n"
            "Context:\n{context}\n"
        ),
    ]
)


revise_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a STRICT reviser."
            "You must output based on the following format:"
            "FORMAT (quote-only answer):"
            "- <direct quote from the CONTEXT>"
            "- <direct quote from the CONTEXT>"
            "Rules:"
            "- Use ONLY the CONTEXT."
            "- Do NOT add any new words besides bullet dashes and the quotes themselves."
            "- Do NOT explain anything."
            "- Do NOT say 'context', 'not mentioned', 'does not mention', 'not provided', etc."
        ),
        (
            "human",
            "Question:\n{question}\n"
            "Current Answer:\n{answer}\n"
            "CONTEXT:\n{context}"
        )
    ]
)


isuse_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are judging USEFULNESS of the ANSWER for the QUESTION."
            
            "Decide if the answer actually addresses what the user asked"
            "Return JSON with keys: isuse, reason."
            "isuse must be one of: useful, not_useful."
            "Rules:"
            "useful: The answer directly answers the question or provides the requested specific info."
            "not_useful: The answer is generic, off-topic, or only gives related background without answering."
            "Do NOT use outside knowledge."
            "Do NOT re-check grounding (IsSUP already did that). Only check: 'Did we answer the question?'"
            "Keep reason to 1 short line."
        ),
        (
            "human",
            "Question:\n{question}\n\nAnswer:\n{answer}"
        ),
    ]
)




rewrite_for_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's QUESTION into a query optimized for vector retrieval over INTERNAL company PDFs."
            "Rules:"
            "Keep it short (6-16 words)."
            "Preserve key entities (e.g., NexaAI, plan names)."
            "Add 2-5 high-signal keywords that likely appear in policy/pricing docs."
            "Remove filler words."
            "Do NOT answer the question."
            " Output JSON with key: retrieval_query"
            "Examples:"
            "Q: 'Do NexaAI plans include a free trial?'"
            " {{'retrieval_query': 'NexaAI free trial duration trial period plans'}}"
            "Q: 'What is NexaAI refund policy?'"
            " {{'retrieval_query': 'NexaAI refund policy cancellation refund timeline charges'}}"
        ),
        (
            "human",
            "QUESTION:\n{question}\n"
            "Previous retrieval query:\n{retrieval_query}\n"
            "Answer :\n{answer}"
        ),
    ]
)








