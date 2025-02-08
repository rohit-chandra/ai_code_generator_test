from llama_index.llms.ollama import Ollama
# load semi structured data like PDFs
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from pydantic import BaseModel
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_pipeline import QueryPipeline # combine mulitple steps

# import our prompt template
from prompts import context, code_parser_template

# import code_reader tool 
from code_reader import code_reader

# load python code
import ast

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


llm = Ollama(model="llama3.2", request_timeout=500.0)

# request = llm.complete("hello world")
# print(request)

# it'll automatically fetch the API key from the environment variable
parser = LlamaParse(result_type="markdown")

# when pdf, use parser
file_extractor = {".pdf": parser}

current_path = Path.cwd()
data_path = current_path / "data"
print(data_path)
documents = SimpleDirectoryReader(data_path, file_extractor=file_extractor).load_data()

# pass the docs to vector store to create embeddings
embed_model = resolve_embed_model("local:BAAI/bge-m3") # local embedding model

# create the index
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# wrap the entire thing in query engine
query_engine = vector_index.as_query_engine(llm=llm)

# test the input prompt
# result = query_engine.query("What are some of the routes in the api?")
# print(result)

tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="api_documentation",
            description="this gives documentation about the code for an API. Use this for reading docs for the API",
        ),
    ),
    code_reader,
]

# another LLM to generate code
code_llm = Ollama(model = "codellama", request_timeout=500.0) # does code generation
agent = ReActAgent.from_tools(tools=tools, llm=code_llm, verbose=True, context=context)

class CodeOutput(BaseModel):
    code: str
    description: str
    filename: str

output_parser = PydanticOutputParser(CodeOutput)
json_prompt_str = output_parser.format(code_parser_template)
json_prompt_template = PromptTemplate(json_prompt_str)
output_pipeline = QueryPipeline(chain=[json_prompt_template, llm])

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    retries = 0
    
    while retries < 3:
        try:
            result = agent.query(prompt)
            # print(result)
            next_result = output_pipeline.run(response=result)
            # print(next_result)
            # load python code as dict
            cleaned_json = ast.literal_eval(str(next_result).replace("assistant:", ""))
            break
        
        except Exception as e:
            print(f"Error occured, retry-->{retries}: {e}")
            retries += 1
        
    if retries >= 3:
        print("Failed after 3 attempts. Please try again.")
        continue # ask new prompt from while loop
    
    print("Code generated")
    print(cleaned_json["code"])
    print("\n\nDescription:", cleaned_json["description"])
    
    filename = cleaned_json["filename"]
    
    # save it to file
    try:
        output_path = Path.cwd() / "output" / filename
        with open(output_path, "w") as f:
            f.write(cleaned_json["code"])
        print(f"Code saved to {filename}")
    except:
        print(f"Error saving file")