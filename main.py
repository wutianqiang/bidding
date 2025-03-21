from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


import utils
from utils import *
#load_txt_file,split_documents,clean_text,extract_and_clean_pdf_content,save_cleaned_content
#嵌入式模型
embeddings = OpenAIEmbeddings(model="text-embedding-3-large",base_url="https://api.openai-hk.com/v1")

# 初始化 LLM
llm = ChatOpenAI(
    model="deepseek-chat",  # 使用的模型名称，具体名称以 DeepSeek 官方文档为准
    openai_api_key="sk-bdb95c9420874becab146824db240c90",  # 替换为你的 DeepSeek API Key
    openai_api_base="https://api.deepseek.com/v1",  # DeepSeek 的 API 基础地址
    temperature=0.1,  # 控制生成文本的随机性，可根据需求调整
    max_tokens=1000  # 最大输出 token 数，可根据需求调整
)



# 定义提示词
prompt_template = """
你是一个专业的招标文件解析助手，请从以下招标文件中提取以下关键信息：
1. 招标编号
2. 项目名称
3. 委托单位
4. 代理机构
5. 开标日期
6. 服务内容
7. 服务期限
8. 项目预算
9. 供应商要求
10. 投标文件递交截止时间
11. 投标文件递交地点
12. 开标地点
13. 采购人联系人及联系方式
14. 招标代理机构联系人及联系方式

请确保提取的信息准确无误，并以 JSON 格式返回结果。

招标文件内容：
{text}
"""

prompt = PromptTemplate(
    input_variables=["text"],
    template=prompt_template
)

# 将文档分割成小块
texts = utils.get_split_text("2018陇南维护招标文件.pdf")

# 创建向量数据库并持久化
persist_directory = 'db'  # 持久化存储目录

# 使用已定义的embeddings创建Chroma数据库
vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory=persist_directory
)



# 创建 LLMChain
#chain = prompt | llm