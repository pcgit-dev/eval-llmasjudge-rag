
##First, we set up a complete RAG system using LangChain. We’ll index content from Lilian
##Weng’s blog posts about LLM agents, prompt engineering, and adversarial attacks:
#Key decisions here: we’re using a chunk size of 250 tokens with no overlap, 
# OpenAI embeddings for the vector store, and retrieving the top 6 most relevant chunks 
# per query.

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

from config import get_settings
from langchain.chat_models import init_chat_model

class raggeneration:
    def __init__(self):
        self.urls = [
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        ]
        self.settings = get_settings()
        self.llm = init_chat_model("openai:gpt-4o-mini")

    def rag_uploader(self):
        docs = [WebBaseLoader(url).load() for url in self.urls]
        docs_list = [item for sublist in docs for item in sublist]

        # Initialize a text splitter with specified chunk size and overlap
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0
        )

        # Split the documents into chunks
        doc_splits = text_splitter.split_documents(docs_list)

        # Add the document chunks to the "vector store" using OpenAIEmbeddings
        vectorstore = InMemoryVectorStore.from_documents(
            documents=doc_splits,
            embedding=OpenAIEmbeddings(api_key=self.settings.openai_api_key),
        )

        # With langchain we can easily turn any vector store into a retrieval component:
        self.retriever = vectorstore.as_retriever(k=6)
    
    def invoke_retriever(self, query: str):
        return self.retriever.invoke(query)
    
    ## The RAG bot retrieves relevant documents, injects them into the system prompt, 
    # and generates an answer. The @traceable() decorator from LangSmith ensures every step is logged:
    @traceable()
    def rag_bot(self,question:str)->dict:
        ## Relevant context
        docs=self.retriever.invoke(question)
        docs_string = " ".join(doc.page_content for doc in docs)

        instructions = f"""You are a helpful assistant who is good at analyzing source information and answering questions.       Use the following source documents to answer the user's questions.       If you don't know the answer, just say that you don't know.       Use three sentences maximum and keep the answer concise.

        Documents:
        {docs_string}"""
            
        ## llm invoke
        ai_msg=self.llm.invoke([
                {"role": "system", "content": instructions},
                {"role": "user", "content": question},

            ])
        return {"answer":ai_msg.content,"documents":docs}