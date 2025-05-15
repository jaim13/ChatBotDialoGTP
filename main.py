from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

chat_history_ids = None

print("Hello, I am your assistant!")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Thanks for everything!")
        break

    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    attention_mask = torch.ones_like(new_input_ids)  # create attention mask of 1s for input tokens

    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
        attention_mask = torch.ones_like(bot_input_ids)  # attention mask covering entire input
    else:
        bot_input_ids = new_input_ids

    outputs = model.generate(
        bot_input_ids,
        attention_mask=attention_mask,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

    response = tokenizer.decode(outputs[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print("Assistant:", response)
