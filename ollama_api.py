import requests
url="http://localhost:11434/api/generate"
common_model="llama3.1:latest"
num_ctx=2000

def send(chat):
      # Prompt for summarization
    prompt =chat# Combine the prompt and the text
    # Parameters to pass to Ollama for generating a summary
    payload = {
        "model": common_model,  
        "prompt": prompt,
        "stream": False,
        "options":{
            "num_ctx":num_ctx,
           
        }
    }
    response = requests.post(url, json=payload)
    ret=response.json()["response"]
    return ret

