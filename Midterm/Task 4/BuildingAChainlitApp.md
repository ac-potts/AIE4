# Building a Chainlit App

What if we want to take our Week 1 Day 2 assignment - [Pythonic RAG](https://github.com/AI-Maker-Space/AIE4/tree/main/Week%201/Day%202) - and bring it out of the notebook?

Well - we'll cover exactly that here!

## Anatomy of a Chainlit Application

[Chainlit](https://docs.chainlit.io/get-started/overview) is a Python package similar to Streamlit that lets users write a backend and a front end in a single (or multiple) Python file(s). It is mainly used for prototyping LLM-based Chat Style Applications - though it is used in production in some settings with 1,000,000s of MAUs (Monthly Active Users).

The primary method of customizing and interacting with the Chainlit UI is through a few critical [decorators](https://blog.hubspot.com/website/decorators-in-python).

> NOTE: Simply put, the decorators (in Chainlit) are just ways we can "plug-in" to the functionality in Chainlit. 

We'll be concerning ourselves with three main scopes:

1. On application start - when we start the Chainlit application with a command like `chainlit run app.py`
2. On chat start - when a chat session starts (a user opens the web browser to the address hosting the application)
3. On message - when the users sends a message through the input text box in the Chainlit UI

Let's dig into each scope and see what we're doing!

## On Application Start:

The first thing you'll notice is that we have the traditional "wall of imports" this is to ensure we have everything we need to run our application. 

```python
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
```

Next up, we have some prompt templates. As all sessions will use the same prompt templates without modification, and we don't need these templates to be specific per template - we can set them up here - at the application scope. 

```python
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
```

> NOTE: You'll notice that these are the exact same prompt templates we used from the Pythonic RAG Notebook in Week 1 Day 2!

Following that - we can create the Python Class definition for our RAG pipeline - or *chain*, as we'll refer to it in the rest of this walkthrough. 

Let's look at the definition first:

```python
class RetrievalAugmentedQAPipeline:
    def __init__(self, llm: ChatOpenAI(), vector_db_retriever: VectorDatabase) -> None:
        self.llm = llm
        self.vector_db_retriever = vector_db_retriever

    async def arun_pipeline(self, user_query: str):
        ### RETRIEVAL
        context_list = self.vector_db_retriever.search_by_text(user_query, k=4)

        context_prompt = ""
        for context in context_list:
            context_prompt += context[0] + "\n"

        ### AUGMENTED
        formatted_system_prompt = system_role_prompt.create_message()

        formatted_user_prompt = user_role_prompt.create_message(question=user_query, context=context_prompt)


        ### GENERATION
        async def generate_response():
            async for chunk in self.llm.astream([formatted_system_prompt, formatted_user_prompt]):
                yield chunk

        return {"response": generate_response(), "context": context_list}
```

Notice a few things:

1. We have modified this `RetrievalAugmentedQAPipeline` from the initial notebook to support streaming. 
2. In essence, our pipeline is *chaining* a few events together:
    1. We take our user query, and chain it into our Vector Database to collect related chunks
    2. We take those contexts and our user's questions and chain them into the prompt templates
    3. We take that prompt template and chain it into our LLM call
    4. We chain the response of the LLM call to the user
3. We are using a lot of `async` again!

Now, we're going to create a helper function for processing uploaded text files.

First, we'll instantiate a shared `CharacterTextSplitter`.

```python
text_splitter = CharacterTextSplitter()
```

Now we can define our helper.

```python
def process_text_file(file: AskFileResponse):
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as temp_file:
        temp_file_path = temp_file.name

    with open(temp_file_path, "wb") as f:
        f.write(file.content)

    text_loader = TextFileLoader(temp_file_path)
    documents = text_loader.load_documents()
    texts = text_splitter.split_texts(documents)
    return texts
```

Simply put, this downloads the file as a temp file, we load it in with `TextFileLoader` and then split it with our `TextSplitter`, and returns that list of strings!

#### QUESTION #1:

Why do we want to support streaming? What about streaming is important, or useful?

## On Chat Start:

The next scope is where "the magic happens". On Chat Start is when a user begins a chat session. This will happen whenever a user opens a new chat window, or refreshes an existing chat window.

You'll see that our code is set-up to immediately show the user a chat box requesting them to upload a file. 

```python
while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a Text File file to begin!",
            accept=["text/plain"],
            max_size_mb=2,
            timeout=180,
        ).send()
```

Once we've obtained the text file - we'll use our processing helper function to process our text!

After we have processed our text file - we'll need to create a `VectorDatabase` and populate it with our processed chunks and their related embeddings!

```python
vector_db = VectorDatabase()
vector_db = await vector_db.abuild_from_list(texts)
```

Once we have that piece completed - we can create the chain we'll be using to respond to user queries!

```python
retrieval_augmented_qa_pipeline = RetrievalAugmentedQAPipeline(
        vector_db_retriever=vector_db,
        llm=chat_openai
    )
```

Now, we'll save that into our user session!

> NOTE: Chainlit has some great documentation about [User Session](https://docs.chainlit.io/concepts/user-session). 

### QUESTION #2: 

Why are we using User Session here? What about Python makes us need to use this? Why not just store everything in a global variable?

## On Message

First, we load our chain from the user session:

```python
chain = cl.user_session.get("chain")
```

Then, we run the chain on the content of the message - and stream it to the front end - that's it!

```python
msg = cl.Message(content="")
result = await chain.arun_pipeline(message.content)

async for stream_resp in result["response"]:
    await msg.stream_token(stream_resp)
```

## 🎉

With that - you've created a Chainlit application that moves our Pythonic RAG notebook to a Chainlit application!

## 🚧 CHALLENGE MODE 🚧

For an extra challenge - modify the behaviour of your applciation by integrating changes you made to your Pythonic RAG notebook (using new retrieval methods, etc.)

If you're still looking for a challenge, or didn't make any modifications to your Pythonic RAG notebook:

1) Allow users to upload PDFs (this will require you to build a PDF parser as well)
2) Modify the VectorStore to leverage [Qdrant](https://python-client.qdrant.tech/)

> NOTE: The motivation for these challenges is simple - the beginning of the course is extremely information dense, and people come from all kinds of different technical backgrounds. In order to ensure that all learners are able to engage with the content confidently and comfortably, we want to focus on the basic units of technical competency required. This leads to a situation where some learners, who came in with more robust technical skills, find the introductory material to be too simple - and these open-ended challenges help us do this! 





