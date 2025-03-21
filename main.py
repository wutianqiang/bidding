from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import InMemoryVectorStore

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
    temperature=0,  # 控制生成文本的随机性，可根据需求调整
    max_tokens=1000  # 最大输出 token 数，可根据需求调整
)

'''
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
'''
# 定义提示词
prompt_template = """
招标文件内容：
{text}
"""

prompt = PromptTemplate(
    input_variables=["text"],
    template=prompt_template
)

# 将文档分割成小块
texts = utils.get_split_text("data.txt")
texts =texts[:4]

# 创建向量数据库并持久化
#persist_directory = 'db'  # 持久化存储目录
vector_store = InMemoryVectorStore(embeddings)
# 使用已定义的embeddings创建Chrom
#
# a数据库
vectordb = vector_store.from_documents(
    texts,
    embedding=embeddings)


#aa = vectordb.similarity_search("请查找项目的招标编号",k=3)

for res in texts:
    print(f"* {res.page_content}")
    print("==========")


# results = vector_store.similarity_search(
#     "编号",
#     k=2
# )
# for res in results:
#     print(f"* {res.page_content} [{res.metadata}]")


results = vectordb.similarity_search_with_score(
    "招标文件获取时间",
    k=2,
)
for res,score in results:
    print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")
    print("========================")


# results = vector_store.similarity_search_by_vector(
#     embedding=embeddings.embed_query("招标编号"), k=3
# )
# for doc in results:
#     print(f"* {doc.page_content} [{doc.metadata}]")




# 创建 LLMChain
chain = prompt | llm

#retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # 检索前 3 个最相似的文档

# 4. 创建检索问答链
qa_chain = RetrievalQA.from_chain_type(
    llm=chain,  # 使用 OpenAI GPT
    chain_type="stuff",  # 使用简单的 "stuff" 方法
    retriever=vectordb.as_retriever(search_kwargs={"k": 2}),  # 使用 FAISS 作为检索器

)

# 5. 提出问题并获取总结
query = "请列出所有的网站地址"
result = qa_chain.invoke(query)

print("问题:", query)
print("总结:", result)
