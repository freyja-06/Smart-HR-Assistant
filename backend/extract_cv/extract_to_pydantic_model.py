from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from extract_cv.candidate_profile import CandidateProfile

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature = 0 # Set the temperature to 0 to ensure maximum extraction accuracy.
)

llm_with_structed_output = llm.with_structured_output(CandidateProfile)

instruction = """
    You are an expert HR Data Extractor. 
    Your task is to extract candidate information from the CV text provided.
    Extract data strictly according to the schema provided.
    If a field is missing in the CV, leave it as None or empty list.

"""

def extract_cv_str_data(cv_text: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", instruction),
        ("human", "{cv_text}")
    ])

    chain = prompt | llm_with_structed_output

    return chain.invoke({"cv_text": cv_text})

# Có thể có rủi ro sập tiến trình nạp CV hàng loạt
