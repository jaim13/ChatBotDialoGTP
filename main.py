from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# modelo DialoGPT-medium de Microsoft
model_name = "microsoft/DialoGPT-medium"

# Carga del tokenizer y modelo preentrenado
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Variable para guardar el historial del chat (las conversaciones previas)
chat_history_ids = None

print("Hello, I am your assistant!")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Thanks for everything!")
        break

    # Codifica la entrada del usuario y añade el token especial de fin de secuencia (eos)
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

    # Máscara de atención para ver que debe ser atendido y que no
    attention_mask = torch.ones_like(new_input_ids)

    if chat_history_ids is not None:
        # concadenma con historial
        bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
        # Actualiza la máscara
        attention_mask = torch.ones_like(bot_input_ids)
    else:
        #Nueva entrada si no hay historial 
        bot_input_ids = new_input_ids

    # Genera la respuesta del modelo
    outputs = model.generate(
        bot_input_ids,
        attention_mask=attention_mask, 
        max_length=1000,                  
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,                 
        top_k=50,                        
        top_p=0.95                        
    )

    # Decodifica la respuesta generada excluyendo la parte de la entrada
    response = tokenizer.decode(outputs[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print("Assistant:", response)
