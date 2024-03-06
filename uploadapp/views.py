from django.shortcuts import render
from .forms import UploadFileForm
from django.core.files.storage import FileSystemStorage
#import openai
from openai import OpenAI
import openai
import tiktoken



# Initialize your OpenAI API key here
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="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",
)
#openai.api_key = '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'
def upload_syllabus(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            fs = FileSystemStorage()
            filename = fs.save(file.name, file)
            uploaded_file_url = filename
            pargraphs = extract_paragraphs_from_pdf(uploaded_file_url)
            paragraph_embeddings = create_embeddings(pargraphs)
            prompt_embedding = create_embeddings(["Generate a study plan based on this syllabus"])
            sorted_indices = find_most_relevant_paragraphs(prompt_embedding, paragraph_embeddings, pargraphs)
            final_prompt = construct_chat_prompt(pargraphs, sorted_indices, max_tokens=1000)
            study_plan = generate_study_plan_with_gpt4("Generate a detailed study plan with exact topics and dates based on this syllabus" + final_prompt)
            return render(request, 'uploadapp/study_plan.html', {
                'study_plan': study_plan
            })
    else:
        form = UploadFileForm()
    return render(request, 'uploadapp/upload_form.html', {
        'form': form
    })


import fitz  # PyMuPDF

def extract_paragraphs_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    paragraphs = []
    for page in document:
        text = page.get_text("text")
        # Split text into paragraphs. This might need customization based on the PDF structure
        paragraphs.extend(text.split('\n\n'))
    document.close()
    return paragraphs


def create_embeddings(text_list):
    # Assuming 'text_list' is a list of strings (paragraphs or prompts)
    embeddings = client.embeddings.create(
        model="text-embedding-ada-002",  # Choose an appropriate embedding model
        input=text_list
    )
    return embeddings.data[0].embedding

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def find_most_relevant_paragraphs(prompt_embedding, paragraph_embeddings, paragraphs):
    prompt_embedding_array = np.array([[embedding] for embedding in prompt_embedding])
    paragraph_embedding_arrays = np.array([[embedding] for embedding in paragraph_embeddings])
    prompt_embedding_array.reshape(-1, 1)
    paragraph_embedding_arrays.reshape(-1, 1)
    
    similarities = cosine_similarity(prompt_embedding_array, paragraph_embedding_arrays)
    sorted_indices = np.argsort(similarities[0])[::-1]  # Sort indices by descending similarity
    
    valid_indices = [index for index in sorted_indices if index < len(paragraphs)]

    return valid_indices  # Returns indices of paragraphs in descending order of relevance

def construct_chat_prompt(paragraphs, sorted_indices, max_tokens=10000):
    selected_text = ""
    current_tokens = 0

    print(len(paragraphs))
    encoding = tiktoken.get_encoding("cl100k_base")
    
    for index in sorted_indices:
        paragraph = paragraphs[index]
        paragraph_tokens = len(encoding.encode(paragraph))
        
        if current_tokens + paragraph_tokens > max_tokens:
            break  # Stop adding paragraphs if max_tokens limit is reached
        
        selected_text += paragraph + "\n\n"
        current_tokens += paragraph_tokens
    
    return selected_text


def generate_study_plan_with_gpt4(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content



def generate_study_plan(text):
    prompt=f"Generate a study plan based on this syllabus: {text}"
    response = client.chat.completions.create(
      model="gpt-4",
      messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].text
