## Development and Deployment of a 'Chat with LLM' Application Using the Gradio Blocks Framework

### AIM:
To design and deploy a "Chat with LLM" application by leveraging the Gradio Blocks UI framework to create an interactive interface for seamless user interaction with a large language model.

### PROBLEM STATEMENT:
Develop a conversational application that connects a user interface built with Gradio Blocks to a backend powered by a Hugging Face (HF) Inference Endpoint (e.g., Falcon-40B-Instruct).
The application must:

Accept user prompts and display the LLM’s responses dynamically.

Maintain conversational history.

Include advanced features such as system instructions, temperature control, and real-time streaming of generated text.
### DESIGN STEPS:

STEP 1:
Import required Python libraries and load environment variables (HF API key, base endpoint).

STEP 2:
Initialize the LLM client using the text_generation.Client from the Hugging Face Inference API.

STEP 3:
Create the format_chat_prompt() function to structure the dialogue context from chat history.

STEP 4:
Define the respond() function to process user input, generate responses via the LLM, and update chat history dynamically.

STEP 5:
Use Gradio Blocks to design the application interface — including a Chatbot, Prompt Textbox, Submit Button, Clear Button, and Accordion for advanced options.

STEP 6:
Enable real-time streaming of generated tokens for smoother user experience.

STEP 7:
Deploy the Gradio app 

### PROGRAM:
```
import os
from dotenv import load_dotenv, find_dotenv
from text_generation import Client
import gradio as gr

_ = load_dotenv(find_dotenv())
hf_api_key = os.environ['HF_API_KEY']
base_url = os.environ['HF_API_FALCOM_BASE']

client = Client(base_url, headers={"Authorization": f"Basic {hf_api_key}"}, timeout=120)

def format_chat_prompt(message, chat_history, instruction):
    prompt = f"System: {instruction}"
    for user_msg, bot_msg in chat_history:
        prompt += f"\nUser: {user_msg}\nAssistant: {bot_msg}"
    prompt += f"\nUser: {message}\nAssistant:"
    return prompt

def respond(message, chat_history, instruction, temperature=0.7):
    prompt = format_chat_prompt(message, chat_history, instruction)
    chat_history = chat_history + [[message, ""]]
    stream = client.generate_stream(prompt, max_new_tokens=512, temperature=temperature, stop_sequences=["\nUser:", "<|endoftext|>"])
    acc_text = ""
    for idx, response in enumerate(stream):
        token = response.token.text
        if idx == 0 and token.startswith(" "):
            token = token[1:]
        acc_text += token
        last_turn = list(chat_history.pop(-1))
        last_turn[-1] += acc_text
        chat_history = chat_history + [last_turn]
        yield "", chat_history
        acc_text = ""

with gr.Blocks() as demo:
    gr.Markdown("### 💬 Chat with Falcon LLM — Powered by Hugging Face API")
    chatbot = gr.Chatbot(height=280)
    msg = gr.Textbox(label="Type your message...")
    with gr.Accordion("⚙️ Advanced Options", open=False):
        system = gr.Textbox(label="System Instruction", lines=2, value="A helpful, honest AI assistant.")
        temperature = gr.Slider(label="Creativity (Temperature)", minimum=0.1, maximum=1, value=0.7, step=0.1)
    btn = gr.Button("Send")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear Chat")
    btn.click(respond, inputs=[msg, chatbot, system, temperature], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot, system, temperature], outputs=[msg, chatbot])

gr.close_all()
demo.queue().launch(share=True, server_port=int(os.environ['PORT4']))
```
### OUTPUT:
![Screenshot 2025-11-14 114400](https://github.com/user-attachments/assets/60a8dbe6-ed63-40f2-a274-29f6366d27d1)


<img width="1019" height="378" alt="Screenshot 2025-11-14 114419" src="https://github.com/user-attachments/assets/91c731cb-b40c-46f1-b935-ffbed48888bf" />

### RESULT:
The “Chat with LLM” application was successfully developed and deployed using Gradio Blocks and Hugging Face Inference API, enabling real-time, multi-turn conversation with an LLM and advanced control over response parameters — demonstrating seamless integration of UI and AI backend.
