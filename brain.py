# Load API Key with dotenv
from dotenv import load_dotenv
import os
# use for encoding image to base64
import base64
# use for making requests to the Groq API
from groq import Groq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Verify the API key exists
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Image to required format
#image_path = r'C:\Users\25ikb\OneDrive\Desktop\Medical_Voicer_chatbot\acne.jpg'
    # use this base64.64encode convert bainary to readable stirng format

def encode_image(image_path):   
    image_file=open(image_path, "rb")
    return base64.b64encode(image_file.read()).decode('utf-8')


# Initialize Groq client with API key
query = "What is this skin condition?"
model = "llama-3.2-90b-vision-preview"
def analyze_image_and_query(query,model,encoded_image): 
    client = Groq(api_key=GROQ_API_KEY)

    
   
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
            ]
        }
    ]

    # Creating the completion
    completion = client.chat.completions.create(
        model=model,
        messages=messages
    )

    # Print the result
    return(completion.choices[0].message.content)
        

