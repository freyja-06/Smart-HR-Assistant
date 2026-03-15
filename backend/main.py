from typing import List, Dict, Any
from langchain_core.documents import Document

# ===============================
# Mock function format_docs
# ===============================

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join([doc.page_content for doc in docs])


# ===============================
# Function cần test
# ===============================

def get_final_prompt(**kwargs) -> str:
    user_query = kwargs.get("user_query")

    documents: List[Document] = kwargs.get("documents") or []
    filtered_documents: List[Document] = kwargs.get("filtered_documents") or []

    email_draft = kwargs.get("email_draft")
    email_sent = kwargs.get("email_sent")

    interview_brief = kwargs.get("interview_brief")
    interview_pdf_path = kwargs.get("interview_pdf_path")

    module_outputs: Dict[str, Any] = kwargs.get("module_outputs") or {}

    completed_tasks = kwargs.get("completed_tasks") or []
    failed_tasks = kwargs.get("failed_tasks") or []

    context_blocks = []

    context_blocks.append(
        f"USER QUESTION:\n{user_query}"
    )

    context_blocks.append(
        f"""
EXECUTION SUMMARY

Completed tasks: {completed_tasks}
Failed tasks: {failed_tasks}
"""
    )

    if filtered_documents:
        docs_text = format_docs(filtered_documents)
        context_blocks.append(
            f"""
RELEVANT KNOWLEDGE (filtered documents)

{docs_text}
"""
        )

    elif documents:
        docs_text = format_docs(documents)
        context_blocks.append(
            f"""
RETRIEVED KNOWLEDGE

{docs_text}
"""
        )

    if email_draft:
        context_blocks.append(
            f"""
EMAIL DRAFT GENERATED

{email_draft}
"""
        )

    if email_sent is True:
        context_blocks.append(
            "EMAIL STATUS: The email was successfully sent."
        )

    if interview_brief:
        context_blocks.append(
            f"""
INTERVIEW BRIEF GENERATED

{interview_brief[:2000]}
"""
        )

    if interview_pdf_path:
        context_blocks.append(
            f"INTERVIEW PDF PATH: {interview_pdf_path}"
        )

    if module_outputs:
        context_blocks.append(
            f"""
OTHER MODULE OUTPUTS

{module_outputs}
"""
        )

    context_blocks.append(
        """
Using the information above, generate the final response to the user.

Guidelines:
- Answer the user question directly.
- Use the provided knowledge when relevant.
- Summarize generated documents instead of repeating them fully.
- If an email draft exists, inform the user.
- If an interview brief exists, summarize the key points.
- If tasks failed, politely inform the user.

Write a clear, professional HR assistant response.
"""
    )

    final_prompt = "\n\n".join(context_blocks)

    return final_prompt


# ===============================
# TEST DATA
# ===============================

documents = [
    Document(page_content="Company policy: Employees can take 12 days annual leave.")
]

filtered_documents = [
    Document(page_content="Annual leave must be approved by the direct manager.")
]

email_draft = """
Subject: Annual Leave Request

Hi Manager,

I would like to request annual leave from June 10 to June 12.

Best regards
"""

interview_brief = """
Candidate: John Doe
Position: Python Developer
Experience: 3 years Python, Django, APIs
Strengths: Backend architecture, debugging
"""

# ===============================
# RUN TEST
# ===============================

prompt = get_final_prompt(
    user_query="Can you help me request annual leave?",
    documents=documents,
    filtered_documents=filtered_documents,
    email_draft=email_draft,
    email_sent=False,
    interview_brief=interview_brief,
    interview_pdf_path="/tmp/interview_brief.pdf",
    module_outputs={"rag_score": 0.87},
    completed_tasks=["retrieve_docs", "generate_email"],
    failed_tasks=[]
)

print(prompt)