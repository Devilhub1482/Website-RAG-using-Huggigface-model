# Website Chunks Creation

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
loader=WebBaseLoader("https://en.wikipedia.org/wiki/Machine_learning")
doc=loader.load()
splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunked_docs=splitter.split_documents(doc)


#vectorestore creation

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
embeddings=HuggingFaceEmbeddings(model_name='avsolatorio/GIST-small-Embedding-v0')
db=FAISS.from_documents(chunked_docs, embeddings)

#Retriever

retriever=db.as_retriever(search_type='similarity',search_kwargs={"k":2})

#Generator
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.chains import RetrievalQA
import torch

model_name='microsoft/Phi-3-mini-128k-instruct'

model=AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16,device_map="auto",offload_folder="offload")
tokenizer=AutoTokenizer.from_pretrained(model_name)

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import BaseOutputParser
from langchain.llms import HuggingFacePipeline
from transformers import pipeline


prompt_template="""
<|system|>
You are a high-precision, accurate assistant. Your task is to extract and provide specific information from the given context only without any elaboration. Follow these guidelines strictly:

1. If the exact answer is not in the context, respond with 'It is outside the context of the document'.
2. Give only the exact answer requested - no additional explanation or information.
3. If the answer is a number, provide only the number without units or symbols, unless explicitly part of the data (e.g., percentages).
4. If the answer is text, provide only the relevant text answer without any additional words from context, template or query. Do not use your own knowledge to answer the question, strictly adhere to make sure that your answer is from within the document only.
5. Answer solely based on the information in the provided context only.
6. Strictly maintain the temperature as zero and don't hallucinate.
7. Prioritize accuracy in your responses.
8. Do not use introductory phrases or explanations or redundancy in your answer.
9. After receiving the output, go through the process from 1-9 again to re-check.
10. Once the re-check is done and good with results, deliver the answer.
Use the following context to help:

{context}

</s>
<|user|>
{question}
</s>
<|assistant|>
"""

model_pipeline=pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,
    task='text-generation',
    temperature=0,
    max_new_tokens=512,
    repetition_penalty=1.1
)

llm=HuggingFacePipeline(pipeline=model_pipeline)

prompt=PromptTemplate(template=prompt_template,input_variables=['context','question'])

class AssistantOnlyOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
      assistant_start = text.find("<|assistant|>") + len("<|assistant|>")
      if assistant_start == -1:
          return ""

      response = text[assistant_start:].strip()
      response = response.replace("<|user|>", "").replace("<|system|>", "").strip()
      return response

llm_chain=prompt | llm | AssistantOnlyOutputParser()



#Ragchain

from langchain_core.runnables import RunnableLambda, RunnablePassthrough
rag_chain = {
    "context": RunnableLambda(lambda x: x["question"]) | retriever,
    "question": RunnablePassthrough()
} | llm_chain


#Inference

rag_chain.invoke({"question": "What is ML?"})
