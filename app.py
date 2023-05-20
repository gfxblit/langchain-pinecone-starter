# %%
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

import pinecone 
import os
# %%
loader = TextLoader('resources/state_of_the_union.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(
    model='text-embedding-ada-002' # this is the default
)
# %%
# initialize pinecone
pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],
    environment=os.environ['PINECONE_ENV']
)

# Make sure to generate an index beforehand
# 1536 dimensions
# Use cosine similarity, per:
# https://platform.openai.com/docs/guides/embeddings/which-distance-function-should-i-use

index_name = "starter"

# This upserts the split docs and associated embeddings to Pinecone
docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)

# If you already have an index, you can load it like this
# docsearch = Pinecone.from_existing_index(index_name, embeddings)
# %%
query = "What did the president say about Ketanji Brown Jackson"
docs = docsearch.similarity_search(query)
# %% Connect with to a chain
chain = RetrievalQAWithSourcesChain.from_chain_type(
    ChatOpenAI(temperature=0, verbose=True),
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    verbose=True
)

# %%

with get_openai_callback() as cb:
    response = chain({"question": "What did the president say about justice?"})
    print(response)
    print('[tot: %d, prmt: %d, cmpl: %d cost$: %f]' %
        (cb.total_tokens, cb.prompt_tokens, cb.completion_tokens, cb.total_cost))