import fitz  # PyMuPDF
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. 读取TXT文档
def load_txt_file(file_path):
    """
    使用LangChain的TextLoader加载TXT文件
    """
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    return documents

# 2. 分片处理
def split_documents(documents, chunk_size=500, chunk_overlap=100):
    """
    使用RecursiveCharacterTextSplitter对文档进行分片
    :param documents: 加载的文档对象
    :param chunk_size: 每个分片的最大字符数
    :param chunk_overlap: 分片之间的重叠字符数
    :return: 分片后的文档列表
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(documents)
    return splits


def clean_texts(text):
    """
    清理文本内容，去除多余的空格、空行等
    """
    lines = text.splitlines()
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    cleaned_text = "\n".join(cleaned_lines)
    return cleaned_text





def extract_and_clean_pdf_content(pdf_path):
    """
    读取PDF文件并提取清理后的文本内容
    """
    doc = fitz.open(pdf_path)
    cleaned_content = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        cleanedtext = clean_texts(text)

        # 如果清理后的文本不为空，则添加到内容列表中
        if cleanedtext:
            cleaned_content.append(cleanedtext)

    # 将清理后的内容合并为一个字符串
    final_content = "\n".join(cleaned_content)
    return final_content


def save_cleaned_content(content, output_path):
    """
    将清理后的内容保存到文件中
    """
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(content)




def get_split_pdf(pdf_path):
    #pdf_path = "2018陇南维护招标文件.pdf"  # 替换为你的PDF文件路径
    output_path = "cleaned_content.txt"  # 清理后的文本保存路径

    # 提取并清理PDF内容
    cleaned_content = extract_and_clean_pdf_content(pdf_path)

    # 保存清理后的内容
    save_cleaned_content(cleaned_content, output_path)

    print(f"清理后的内容已保存到: {output_path}")

    documents= load_txt_file(output_path)

    splits = split_documents(documents)

    return splits

def get_split_text(txt_path):
    # 提取并清理PDF内容
    documents= load_txt_file(txt_path)
    splits = split_documents(documents)

    return splits