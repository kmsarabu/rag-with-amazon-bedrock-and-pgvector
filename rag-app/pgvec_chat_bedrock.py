import logging
import os
import sys

import boto3
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.vectorstores.pgvector import PGVector
from langchain_community.chat_models import BedrockChat
from langchain.memory import ConversationSummaryBufferMemory

import helper_functions as hfn


class MissingEnvironmentVariable(Exception):
    """Raised if a required environment variable is missing"""


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


MAX_HISTORY_LENGTH = 5

COLLECTION_ENV_VAR = "COLLECTION_NAME"
DB_SECRET_ENV_VAR = "DB_CREDS"
API_KEY_SECRET_ENV_VAR = "API_KEY_SECRET_NAME"

DEFAULT_LOG_LEVEL = logging.INFO
LOGGER = logging.getLogger(__name__)
LOGGING_FORMAT = "%(asctime)s %(levelname)-5.5s " \
                 "[%(name)s]:[%(threadName)s] " \
                 "%(message)s"


def build_chain(db_creds, collection):
  """Build conversational retrieval chain

  :param db_creds: Dictionary
  :param collection: String

  :rtype: ConversationalRetrievalChain
  """
  region = os.environ["AWS_REGION"]
  BEDROCK_CLIENT = boto3.client("bedrock-runtime", region)

  llm = BedrockChat(model_id="anthropic.claude-3-haiku-20240307-v1:0", client=BEDROCK_CLIENT)
  llm.model_kwargs = {"temperature": 0.5, "max_tokens": 8191, "top_k": 500, "top_p": 1,
                      "stop_sequences": ["\n\nHuman"], }

  # anthropic.claude-v2:1
  conn_str = PGVector.connection_string_from_db_params(
     driver=os.environ.get("PGVECTOR_DRIVER", "psycopg2"),
     host=db_creds["host"],
     port=db_creds["port"],
     database=os.environ.get("PGVECTOR_DATABASE", "postgres"),
     user=db_creds["username"],
     password=db_creds["password"],
  )

  embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=BEDROCK_CLIENT)

  store = PGVector(
    collection_name=collection,
    connection_string=conn_str,
    embedding_function=embeddings,
  )
  retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 3, "include_metadata": True})

  prompt_template = """Human: You are a helpful assistant that answers questions directly and only using the information provided in the context below. 
  Guidance for answers:
      - Always use English as the language in your responses.
      - In your answers, always use a professional tone.
      - Begin your answers with "Based on the context provided: "
      - Simply answer the question clearly and with lots of detail using only the relevant details from the information below. If the context does not contain the answer, say "Sorry, I didn't understand that. Could you rephrase your question?"
      - Use bullet-points and provide as much detail as possible in your answer. 
      - Always provide a summary at the end of your answer.

  Now read this context below and answer the question at the bottom.

  Context: {context}

  Question: {question}

  Assistant:"""

  PROMPT = PromptTemplate(
      template=prompt_template, input_variables=["context", "question"]
  )

  condense_qa_template = """{chat_history}
  Human:
  Given the previous conversation and a follow up question below, rephrase the follow up question
  to be a standalone question.

  Follow Up Question: {question}
  Standalone Question:

  Assistant:"""
  standalone_question_prompt = PromptTemplate.from_template(condense_qa_template)


  memory = ConversationSummaryBufferMemory(
        llm=llm,
        memory_key='chat_history',
        return_messages=True,
        ai_prefix="Assistant",
        output_key='answer')

  conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        return_source_documents=True,
        retriever=retriever,
        get_chat_history=lambda h: h,
        memory=memory,
        combine_docs_chain_kwargs={'prompt': PROMPT}
    )
  return conversation_chain.invoke

def run_chain(chain, prompt: str, history=[]):
  return chain({"question": prompt, "chat_history": history})


if __name__ == "__main__":

  # logging configuration
  log_level = DEFAULT_LOG_LEVEL
  if os.environ.get("VERBOSE", "").lower() == "true":
     log_level = logging.DEBUG
  logging.basicConfig(level=log_level, format=LOGGING_FORMAT)

  # vector db secret fetch
  secret_name = os.environ.get(DB_SECRET_ENV_VAR)
  if not secret_name:
     raise MissingEnvironmentVariable(f"{DB_SECRET_ENV_VAR} environment variable is required")

  # get collection name
  collection = os.environ.get(COLLECTION_ENV_VAR)
  if not collection:
     raise MissingEnvironmentVariable(f"{COLLECTION_ENV_VAR} environment variable is required")
  
  LOGGER.info("starting conversational retrieval chain now..")
  
  # langchain stuff
  chat_history = []
  qa = build_chain(
     hfn.get_secret_from_name(secret_name), 
     collection
  )
  
  print(bcolors.OKBLUE + "Hello! How can I help you?" + bcolors.ENDC)
  print(bcolors.OKCYAN + "Ask a question, start a New search: or CTRL-D to exit." + bcolors.ENDC)
  print(">", end=" ", flush=True)
  
  for query in sys.stdin:
    if (query.strip().lower().startswith("new search:")):
      query = query.strip().lower().replace("new search:","")
      chat_history = []
    elif (len(chat_history) == MAX_HISTORY_LENGTH):
      chat_history.pop(0)

    result = run_chain(qa, query, chat_history)

    chat_history.append((query, result["answer"]))

    print(bcolors.OKGREEN + result['answer'] + bcolors.ENDC)
    if 'source_documents' in result:
      print(bcolors.OKGREEN + 'Sources:')
      for d in result['source_documents']:
        print(d.metadata['source'])
    print(bcolors.ENDC)
    print(bcolors.OKCYAN + "Ask a question, start a New search: or CTRL-D to exit." + bcolors.ENDC)
    print(">", end=" ", flush=True)
  
  print(bcolors.OKBLUE + "Bye" + bcolors.ENDC)
