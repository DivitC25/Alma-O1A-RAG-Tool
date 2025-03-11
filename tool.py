import tiktoken
import pdfplumber
import docx
import pytesseract
import time
import os
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from pdf2image import convert_from_path
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from pinecone import Pinecone
from pinecone import ServerlessSpec
from uuid import uuid4

app = FastAPI()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

class Response(BaseModel):
    assessment: str

# Function to validate the PDF
def validate_pdf(file: UploadFile) -> bool:
    try:
        # Rewind the file pointer to the beginning
        file.file.seek(0)
        pdf = pdfplumber.open(BytesIO(file.file.read()))
        if len(pdf.pages) > 0:
            return True
        else:
            return False
    except Exception:
        return False

# Function to extract text from PDF
def text_from_pdf(file: UploadFile) -> str:
    try:
        # Ensure the file is valid before extracting text
        if not validate_pdf(file):
            raise HTTPException(status_code=400, detail="Invalid or corrupted PDF file.")
        
        with pdfplumber.open(BytesIO(file.file.read())) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            return text if text else None
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error reading PDF.")

# Function to extract text from DOCX
def text_from_docx(file: UploadFile) -> str:
    doc = docx.Document(BytesIO(file.file.read()))
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Function to extract text from image
def text_from_image(file: UploadFile) -> str:
    image = BytesIO(file.file.read())
    return pytesseract.image_to_string(image)

# Function to extract text from scanned PDFs (OCR)
def text_from_scanned_pdf(file: UploadFile) -> str:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file.file.read())
        temp_file_path = temp_file.name
    images = convert_from_path(temp_file_path, 300)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image)
    os.remove(temp_file_path)
    return text


# Function to calculate tiktoken length
def tiktoken_len(text):
  tokenizer = tiktoken.get_encoding('cl100k_base')
  tokens = tokenizer.encode(
    text,
    disallowed_special=()
  )
  return len(tokens)

# Function to create Pinecone database
def create_database():
  pc = Pinecone(api_key=PINECONE_API_KEY)
  spec = ServerlessSpec(
    cloud="aws", region="us-east-1"
  )
  index_name = 'alma-prod'
  existing_indexes = [
    index_info["name"] for index_info in pc.list_indexes()
  ]

  if index_name not in existing_indexes:
    pc.create_index(
      index_name,
      dimension=1536,
      metric='dotproduct',
      spec=spec
    )

    while not pc.describe_index(index_name).status['ready']:
      time.sleep(1)

  index = pc.Index(index_name)
  return index


# Function to determine response based on extracted text
# Function to determine response based on extracted text
def determine_response(text: str) -> str:
    model_name = 'text-embedding-ada-002'

    # LangChain & OpenAI Text Processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=20,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
    )
    embed = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=OPENAI_API_KEY
    )

    # Database Indexing
    index = create_database()
    batch_limit = 100
    texts = []
    metadatas = []

    # Split the text directly without treating it as an iterable with 'page_content'
    record_texts = text_splitter.split_text(text)
    record_metadatas = [{
        "chunk": j, "text": record_text
    } for j, record_text in enumerate(record_texts)]
    
    texts.extend(record_texts)
    metadatas.extend(record_metadatas)

    if len(texts) >= batch_limit:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embed.embed_documents(texts)
        index.upsert(vectors=zip(ids, embeds, metadatas))
        texts = []
        metadatas = []

    if len(texts) > 0:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embed.embed_documents(texts)
        index.upsert(vectors=zip(ids, embeds, metadatas))

    # Creating Vector Store 
    text_field = "text"
    vectorstore = Pinecone(
        index, embed.embed_query, text_field
    )

    # Query Logic
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name='gpt-3.5-turbo',
        temperature=0.0
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    # Relevant Queries for Generating Response
    query = "Assume you are a strict immigration judge. (a) You are identifying the big awards in this resume that meet the o1-a criteria for big awards i.e they fall within the single field of expertise you have identified, they are nationally or internationally recognised, they are very selective, and they dont fall within another field (b) You are identifying the exclusive professional groups in this resume that meet the o1-a criteria for exclusive professional groups i.e they strictly have a 5% - 10% acceptance rate, have established rules of membership, and are within the persons field of expertise (c) You are identifying the media mentions in this resume i.e articles in mainstream media or in industry-specific publications that mention the candidate. Include all articles made by mainstream media or industry specific-publications in the resume and do not just give a brief summary. (d) You are identifying the judging roles in this resume i.e roles in judging other's work in the candidates field of expertise, at competitions or in journals (e) You are identifying the original contributions in this resume i.e novel and significantly impactful contributions to the candidate's field of expertise. these contributions should not be easy to create, or significantly found within work in the candidate's field of expertise (f) You are identifying the scholarly articles in this resume i.e articles published in research journals or major trade publications in the candidate's field of expertise (g) You are identifying the critical employment in this resume i.e important roles in dsitinguished institutions in the candidates field of expertise where the company cannot possibly function without the candidate like executive roles or highly skilled non-management roles. These are not internships or apprenticeships (h) You are identifying the high compensation in this resume i.e either roles in established companies with a known significant total compensation higher than the average salary in the field, or founder/executive roles in startups with significant funding.list the items that fall within each one of these. Strictly only list the item, and not any further analysis. Based on the number of criteria this individual satisfies, give a low/medium/high rating, structured as \"Rating for O1-A Visa Chances\" followed by the rating. If less than 3 criteria are satisfied, give a rating of low. If exactly three criteria are satisfied, give a rating of medium. If more than three criteria are satisfied, give a rating of high"
    response = qa(query)

    # Clearing Index for User Data Privacy
    try:
        index.delete_all()
    except Exception as e:
        print(f"Error executing privacy command: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to execute data privacy command")
    
    return response

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Alma-O1A-RAG-Tool API!"}

@app.post("/assess_ola")
async def assess_ola(file: UploadFile = File(...)):
    # Text Extraction Logic
    file_extension = file.filename.split('.')[-1].lower()

    if file_extension == 'pdf':
        text = text_from_pdf(file)
        if not text:
            print("No text found in PDF, resorting to OCR...")
            text = text_from_scanned_pdf(file)

    elif file_extension == 'docx':
        text = text_from_docx(file)
    elif file_extension in ['jpg', 'jpeg', 'png']:
        text = text_from_image(file)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    if not text:
        raise HTTPException(status_code=400, detail="Unable to extract text from the file")

    debug_info = {
        "message": "Text extraction complete",
        "text_length": len(text),
        "sample_text": text[:300],  # Show first 300 characters for quick validation
    }
    
    assessment = determine_response(text)
    return JSONResponse(content={"assessment": assessment, "debug_info": debug_info})
