from langchain.document_loaders import PyPDFLoader
import pdfplumber

def extract_text_and_tables(pdf_path):
    # 使用 LangChain 加载文本
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    text = "\n".join([page.page_content for page in pages])

    # 使用 PDFPlumber 提取表格
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables.extend(page.extract_tables())

    return text, tables

def format_tables(tables):
    formatted_tables = []
    for table in tables:
        # 将表格的每一行转换为字符串
        table_str = "\n".join(["\t".join(map(str, row)) for row in table])
        formatted_tables.append(table_str)
    return formatted_tables

from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_content(text, tables, chunk_size=500, chunk_overlap=50):
    # 初始化分片器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # 分片文本
    text_chunks = text_splitter.split_text(text)

    # 分片表格
    table_chunks = []
    for table in tables:
        table_chunks.extend(text_splitter.split_text(table))

    # 合并所有分片
    all_chunks = text_chunks + table_chunks
    return all_chunks

from langchain.embeddings import OpenAIEmbeddings

# 初始化 OpenAI 向量模型
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

def vectorize_chunks(chunks):
    # 将分片内容转换为向量
    vectors = embeddings.embed_documents(chunks)
    return vectors

import chromadb
from chromadb.config import Settings

# 初始化 Chroma 客户端
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db"  # 数据持久化目录
))

# 创建或加载集合
collection = client.get_or_create_collection(name="pdf_content")

def store_vectors(chunks, vectors):
    # 将分片和向量存储到 Chroma
    ids = [f"chunk-{i}" for i in range(len(chunks))]
    collection.add(
        ids=ids,
        embeddings=vectors,
        documents=chunks
    )
from langchain.chains import VectorDBQA
from langchain.llms import OpenAI

def query_chroma(query):
    # 初始化 LangChain 的 QA 链
    qa = VectorDBQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        vectorstore=collection
    )
    # 查询 Chroma
    result = qa.run(query)
    return result


import pdfplumber
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import chromadb
from chromadb.config import Settings

# 初始化 Chroma 客户端
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db"
))
collection = client.get_or_create_collection(name="pdf_content")

# 初始化 OpenAI 向量模型
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")


def process_pdf(pdf_path):
    # 提取文本和表格
    text, tables = extract_text_and_tables(pdf_path)

    # 格式化表格
    formatted_tables = format_tables(tables)

    # 分片
    chunks = chunk_content(text, formatted_tables)

    # 向量化
    vectors = vectorize_chunks(chunks)

    # 存储到 Chroma
    store_vectors(chunks, vectors)


# 示例调用
process_pdf("example.pdf")

# 查询 Chroma
query_result = query_chroma("表格中的内容是什么？")
print(query_result)