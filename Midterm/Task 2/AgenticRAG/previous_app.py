import os
from typing import List
from chainlit.types import AskFileResponse
from aimakerspace.text_utils import CharacterTextSplitter, TextFileLoader
from aimakerspace.openai_utils.prompts import (
    UserRolePrompt,
    SystemRolePrompt,
    AssistantRolePrompt,
)
from aimakerspace.openai_utils.embedding import EmbeddingModel
from aimakerspace.vectordatabase import VectorDatabase
from aimakerspace.openai_utils.chatmodel import ChatOpenAI
import chainlit as cl
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_experimental.text_splitter import SemanticChunker
# from langchain_openai.embeddings import OpenAIEmbeddings

system_template = """\
Use the following context to answer a users question. If you cannot find the answer in the context, say you don't know the answer."""
system_role_prompt = SystemRolePrompt(system_template)

user_prompt_template = """\
Context:
{context}
Question:
{question}
"""
user_role_prompt = UserRolePrompt(user_prompt_template)

class RetrievalAugmentedQAPipeline:
    def __init__(self, llm: ChatOpenAI(), vector_db_retriever: VectorDatabase) -> None:
        self.llm = llm
        self.vector_db_retriever = vector_db_retriever

    async def arun_pipeline(self, user_query: str):
        context_list = self.vector_db_retriever.search_by_text(user_query, k=4)

        context_prompt = ""
        for context in context_list:
            context_prompt += context[0] + "\n"

        formatted_system_prompt = system_role_prompt.create_message()

        formatted_user_prompt = user_role_prompt.create_message(question=user_query, context=context_prompt)

        async def generate_response():
            async for chunk in self.llm.astream([formatted_system_prompt, formatted_user_prompt]):
                yield chunk

        return {"response": generate_response(), "context": context_list}

text_splitter = RecursiveCharacterTextSplitter()
# try:
#     api_key = os.environ["OPENAI_API_KEY"]
# except KeyError:
#     print("Environment variable OPENAI_API_KEY not found")
# text_splitter = SemanticChunker(OpenAIEmbeddings(api_key=api_key), breakpoint_threshold_type="standard_deviation")

def process_text_file(file: AskFileResponse):
    import tempfile
    from langchain_community.document_loaders.pdf import PyPDFLoader

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=file.name) as temp_file:
        temp_file_path = temp_file.name

    with open(temp_file_path, "wb") as f:
        f.write(file.content)

    if file.type == 'text/plain':
        text_loader = TextFileLoader(temp_file_path)
        documents = text_loader.load_documents()
    elif file.type == 'application/pdf':
        pdf_loader = PyPDFLoader(temp_file_path)
        documents = pdf_loader.load()
    else:
        raise ValueError("Provide a .txt or .pdf file")
    texts = [x.page_content for x in text_splitter.transform_documents(documents)]
    # texts = [x.page_content for x in text_splitter.split_documents(documents)]
    return texts



@cl.on_chat_start
async def on_chat_start():
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a Text file or a PDF to begin!",
            accept=["text/plain", "application/pdf"],
            max_size_mb=12,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(
        content=f"Processing `{file.name}`...", disable_human_feedback=True
    )
    await msg.send()

    # load the file
    texts = process_text_file(file)

    print(f"Processing {len(texts)} text chunks")

    # Create a dict vector store
    vector_db = VectorDatabase()
    vector_db = await vector_db.abuild_from_list(texts)
    
    chat_openai = ChatOpenAI()

    # Create a chain
    retrieval_augmented_qa_pipeline = RetrievalAugmentedQAPipeline(
        vector_db_retriever=vector_db,
        llm=chat_openai
    )
    
    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", retrieval_augmented_qa_pipeline)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")

    msg = cl.Message(content="")
    result = await chain.arun_pipeline(message.content)

    async for stream_resp in result["response"]:
        await msg.stream_token(stream_resp)

    await msg.send()