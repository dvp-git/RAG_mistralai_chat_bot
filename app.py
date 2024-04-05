# Install all the necessary libraries
import fitz
import os
import time
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import random
import re
import faiss
import torch
from time import perf_counter as timer
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig


embed_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
embed_model.to('cuda')
device = "cuda" # the device to load the model onto

# https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/utils/quantization_config.py#L182
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")

# Model instantiated with quantization config
llm_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", \
                                                   quantization_config=quant_config,
                                                 low_cpu_mem_usage=True)

# Model tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", load_in_4bit=True, )




def pdf_file(path):
    pdf_files = [f for f in os.listdir(path) if f.endswith(".pdf")]
    print("SUCCESS pdf")
    return pdf_files

# Get Text content of the pdf
# For each pdf , get the pdf file contents 
# Store meta-data information about each pdf.

def format_text(text):
    text = text.strip("\n\n")
    text=text.replace("\n",'')
    
    return text

# # https://community.adobe.com/t5/acrobat-discussions/page-number-in-print-does-not-display-in-adobe-s-page-number-box/td-p/13781534
# Doesn't use Logical Page numbers. Use the normal Page numbers
# Logical causes issues during the rendering as you can't generalize to multiple pdfs

def get_text_content(pdf_text_, pdf_):
    doc = fitz.open(pdf_)
    print(len(doc))
    for page_num, page in enumerate(doc):
        # Extract the text content of the page
        text = page.get_text()
        text = format_text(text)
        name = pdf_.strip('.pdf')
        pdf_text_.append((text, page_num+1, name))
    print("SUCCESS text conversion")
    return pdf_text_


# Create a dictionary to store each pages documents into sentences

# Estimating that each token is 4 characters
# Page Info
def page_formatter(page):
    page_ = {}
    for pg in page:
        page_['doc_name'] = page[2]
        page_['text'] = page[0]
        page_['pg_num'] = page[1]
        page_['pg_num_chars'] = len(page[0])
        page_['pg_num_words'] = len(page[0].split(' '))
        page_['pg_num_sentences'] = len(page[0].split('. ')) # Since sentences usually begin with '. '
        page_['pg_num_tokens'] = page_['pg_num_chars'] / 4
    return page_
                         

# Calculate the mean number of sentences in each document.Then use sentence_splitter 
# to split the long sentences
def calc_mean(doc):
    doc_df = pd.DataFrame(doc)
    doc_mean = doc_df['pg_num_tokens'].mean()
    doc_sentences_mean = doc_df['pg_num_sentences'].mean()
    return doc_mean, doc_sentences_mean


# Each sentence is long , the mean token size are above and won't fit the embedding model for tokenization
# Split sentences into smaller chunks
# Method to put all sentences in an array called sentences for making the splitting easier
def sentence_formatter_per_page(doc):
    for page in doc:
        page['sentences'] = [sentence for sentence in page['text'].split('. ')]
    
              
def sentence_formatter_all_books(books):
    for item in books:
        sentence_formatter_per_page(item)
    print("SUCCESS sentence_formatter_all_books")

# Keep overlap of 1 sentence default for every 6 sentences
def slice_sentences(page_sentences, slice_size=5,overlap_size=1) :
    return [page_sentences[i:i + slice_size] for i in range(0, len(page_sentences) - slice_size + 1, slice_size - overlap_size)]


def all_content_sentence_splitter(all_content):
    for book in all_content:
        for page in tqdm(book):
            page["sentence_chunks"] = slice_sentences(page["sentences"],5,1)

            page["num_chunks"] = len(page["sentence_chunks"])


def final_chunk_dict(all_content, all_pdf_formatted):
    for doc in tqdm(all_content):
        for page in doc:
            for sentence_chunk in page["sentence_chunks"]:
                chunk_ = {}
                chunk_["pg_num"] = page["pg_num"]
                # Join sentences like a paragraph structure
                joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
                joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk) # spacing issue after join.
                chunk_["sentence_chunk"] = joined_sentence_chunk
                #write metadata
                chunk_["chunk_num_chars"] = len(joined_sentence_chunk)
                chunk_["chunk_num_tokens"] = len(joined_sentence_chunk) / 4
                chunk_["chunk_num_words"] = len([word for word in joined_sentence_chunk.split(" ")])
                chunk_["doc_name"] = page["doc_name"]
                all_pdf_formatted.append(chunk_)

# Rectifying the max_length tokens
def check_and_rectify_token_length(docs_df):
    # 384 is the max_token_length for some smaller models
    df_exceeding_length = docs_df[docs_df['chunk_num_tokens'] > 384]
    # Split further
    # Remove the df
    docs_df = docs_df[docs_df['chunk_num_tokens'] <= 384]
    # Convert them to dict
    doc_token_length_exceeded = df_exceeding_length.to_dict("records")
    # Run the splitter further. Split the chunks into two
    for doc in doc_token_length_exceeded:
        # Get sentencfe length
        doc_new_1 = {}
        doc_new_2 = {}
        length_chunk = len(doc['sentence_chunk'])
        # Split it into two
        # Assume `text` is your long text
        # Find the midpoint
        midpoint = length_chunk // 2

        # Find the last period before the midpoint
        split_point = doc['sentence_chunk'].rfind('. ', 0, midpoint)

        # Split the text into two parts
        first_half = doc['sentence_chunk'][:split_point+1]  # +1 to include the period
        second_half = doc['sentence_chunk'][split_point:]  # +2 to skip the period and space

        # Reclaculate all metadata except pg_num and doc_name            
        doc_new_1['pg_num'] = doc['pg_num']
        doc_new_2['pg_num'] = doc['pg_num']

        doc_new_1['doc_name'] = doc['doc_name']
        doc_new_2['doc_name'] = doc['doc_name']


        #write metadata
        doc_new_1['sentence_chunk'] = first_half
        doc_new_1["chunk_num_chars"] = len(first_half)
        doc_new_1["chunk_num_tokens"] = len(first_half) / 4
        doc_new_1["chunk_num_words"] = len([word for word in first_half.split(" ")])

        #write metadata
        doc_new_2['sentence_chunk'] = second_half
        doc_new_2["chunk_num_chars"] = len(first_half)
        doc_new_2["chunk_num_tokens"] = len(first_half) / 4
        doc_new_2["chunk_num_words"] = len([word for word in first_half.split(" ")])
        

        # Add it to the original dataframe

        docs_df = pd.concat([docs_df, pd.DataFrame([doc_new_1]), pd.DataFrame([doc_new_2])], ignore_index=True) 
    return docs_df



# def encode_sentences(documents):
#     for doc in tqdm(documents):
#         doc["embeddings"] = embed_model.encode(doc["sentence_chunk"], convert_to_tensor=True)
def convert_to_np(embeddings):
    batched_embeddings_np = embeddings.detach().cpu().numpy()
    return batched_embeddings_np

def save_embedding_metadata_mapping(preprocessed_data_with_embedding):
    data_csv = pd.DataFrame(preprocessed_data_with_embedding).to_csv("data_.csv", index=False, escapechar='/')
    return 

def encode_sentences_batched(documents):
    sentence_chunks = [doc['sentence_chunk'] for doc in documents]
    #print(len(sentence_chunks))
    #print(len(all_pdf_formatted_dict))
    batched_sentence_embeddings = embed_model.encode(sentence_chunks, batch_size=15, convert_to_tensor=True)
    documents = pd.DataFrame(documents)
    embeddings_np = convert_to_np(batched_sentence_embeddings)
    documents['embeddings'] = [embedding for embedding in embeddings_np]
    documents = documents.to_dict('records')
    return documents,batched_sentence_embeddings, embeddings_np


def create_and_save_faiss_index(dimension, embeddings_np):
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    faiss.write_index(index, "index.bin")
    return index


def retrieve_docs(query, doc_embeddings, all_docs_array, model, faiss_index,n_return_docs=5, print_time=True, print_docs=True):
    """ Embed the query , and return the top docs distance, indices"""
    model = model.to('cuda')
    # Query embedding
    query_embedding = model.encode(query)
    
    # Reshaping since faiss expects nxd ndarry format
    query_embedding = np.expand_dims(query_embedding, axis=0)
    
    # start time
    time_strt = timer()
    distances, retrieved_ids = faiss_index.search(query_embedding, n_return_docs)
    time_end = timer()
    
    if print_time:
        print(f"[INFO] Time taken to get search on {len(doc_embeddings)} embeddings: {time_end-time_strt:.5f} seconds")
    
    if print_docs:
        print(f"QUERY: {query} \n")
        print("Retrieved Docs: \n")
        for doc_num,doc_id in enumerate(retrieved_ids[0]):
            print(f"DOC {doc_num}")
            print(f"\n {all_docs_array[doc_id]['sentence_chunk']}")
            print("\n")
    return retrieved_ids[0]


def get_current_gpu_memory():
    gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
    gpu_memory_gb = round(gpu_memory_bytes / (2**30))
    print(f"Available GPU memory: {gpu_memory_gb} GB")


def get_model_params(model):
    return sum([torch.numel(param) for param in model.parameters()])


# Define model memory size on our machine
def get_model_mem_size(model: torch.nn.Module):
    """
    Get how much memory a PyTorch model takes up.

    See: https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822
    """
    # Get model parameters and buffer sizes
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])

    # Calculate various model sizes
    model_mem_bytes = mem_params + mem_buffers # in bytes
    model_mem_mb = model_mem_bytes / (1024**2) # in megabytes
    model_mem_gb = model_mem_bytes / (1024**3) # in gigabytes

    return {"model_mem_bytes": model_mem_bytes,
            "model_mem_mb": round(model_mem_mb, 2),
            "model_mem_gb": round(model_mem_gb, 2)}





# Using Zero shot learning
def data_augmenter(query, documents):
    context = "\n - " + "\n - ".join([doc["sentence_chunk"] for doc in documents])
    base_prompt = f"""Based on the following context documents, please answer the query.
    Think about the question and use only the context to answer the query.
    Do not return the thinking, only return the answer.
    Make sure to provide detailed explainations.If you think there is not enough information in the context documents, say that you do not have enough information.
    Here are a few examples for your referance. Use it as a reference for the answer.
    \nExample 1:
    Query: What is the best resource to learn generative AI ?
    Answer: Generative AI is a new and emerging field. The best way to learn is by practically implementing projects using HuggingFace transformers, LangChain and other available tools.
    \nExample 2:
    Query: What is asdsdfvxcvsd ?
    Answer: I am sorry but I do not understand the question. Kindly re-correct the question or rephrase it to a meaningful ask.
    \nNow use the following context below to answer the user query: 
    {context}
    \nRelevant passages: <extract relevant passages from the context here>
    User Query: {query}
    Answer:"""
    
    # Chat Template
    chat_template = [
                    {"role": "user", 
                     "content": base_prompt
                    }
                  ]
    
    prompt = tokenizer.apply_chat_template(conversation=chat_template, tokenize=False, add_generation_prompt=True)
    return prompt



def generate_answer(query, doc_embeddings, all_pdf_formatted_dict, index , tokenizer,embed_model, llm_model):
    doc_ids =  retrieve_docs(query, doc_embeddings, all_pdf_formatted_dict, embed_model, index, n_return_docs=5, print_time=False, print_docs=False)
    context_docs = [all_pdf_formatted_dict[i] for i in doc_ids]
        
    # Format the prompt with context items
    prompt = data_augmenter(query=query,documents=context_docs)
    
    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generate an output of tokens
    outputs = llm_model.generate(**input_ids,
                                 temperature=0.3,
                                 do_sample=True,
                                 max_new_tokens=512)
    
    # Turn the output tokens into text
    generated_output = tokenizer.decode(outputs[0])        
    print(f"Query: {query}")
    print(f"Answer:{generated_output.replace(prompt[3:], '')}")

####################################################################################
# Data Pre-processing
def preprocessing_data(pdf_path):
    pdf_files = pdf_file(pdf_path)
    books_ = [get_text_content([], pdf_) for pdf_ in pdf_files]
    all_content_ = [[page_formatter(page) for page in books] for books in books_]
    sentence_formatter_all_books(all_content_)
    all_content_sentence_splitter(all_content_)
    all_pdf_formatted = []
    final_chunk_dict(all_content_, all_pdf_formatted )
    all_pdf_formatted_df = pd.DataFrame(all_pdf_formatted)
    all_pdf_formatted_df = all_pdf_formatted_df[all_pdf_formatted_df['chunk_num_tokens'] >= 10]
    all_pdf_formatted = check_and_rectify_token_length(all_pdf_formatted_df).to_dict('records')
    return all_pdf_formatted


###########################################################################



# Process the documents to list[dicts]
documents_processed = preprocessing_data("./")

# Get the embeddings of the documents map it to documents
documents_processed, batched_sentence_embeddings, embeddings_np = encode_sentences_batched(documents_processed)

# Map it to a csv for storage
save_embedding_metadata_mapping(documents_processed)

# Get the mode dimension
dimension = embed_model[1].word_embedding_dimension

faiss_index = create_and_save_faiss_index(dimension, embeddings_np)


# retrieve_docs("Explain recurrent nueral networks", embeddings_np,documents_processed, embed_model, \
#               faiss_index, n_return_docs=6, print_time=True, print_docs=True)
question=''
while(question!='exit' or question == 'quit'):
    question = input("\nKindly ask your question\n")
    if (question == 'quit') or (question == 'exit'):
        break
    elif question == '':
        continue
    else:
        generate_answer(question, batched_sentence_embeddings,documents_processed, faiss_index, tokenizer, embed_model, llm_model )