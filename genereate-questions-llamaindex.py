# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 15:10:17 2024

@author: ayhan
"""

import logging
import sys
import pandas as pd
from langchain_community.llms import LlamaCpp

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


from llama_index.core.evaluation import DatasetGenerator, RelevancyEvaluator
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Response
from llama_index.llms.openai import OpenAI

#gguf file path
gguf_model_path="./gemma-2b-it-q8_0.gguf"
#text file path
textfiles_path="./llmdocs"

llm = LlamaCpp(
        model_path=gguf_model_path,
        temperature=1,
        max_tokens=500,
        top_p=1,
        n_ctx=2048, 
        verbose=True,)

reader = SimpleDirectoryReader("./llmdocs")
documents = reader.load_data()

import nest_asyncio

nest_asyncio.apply()

data_generator = DatasetGenerator.from_documents(documents,llm=llm)
eval_questions = data_generator.generate_questions_from_nodes()

print(eval_questions)
eval_questions=eval_questions[1:]


#save questiosn to a file
with open("llamaindex-doc-questions.txt", "w") as file:
    # Iterate through the list and write each element to the file
    for item in eval_questions:
        file.write(item + "\n")

print("The list has been written to output.txt.")