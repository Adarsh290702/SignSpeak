from transformers import AutoModelForCausalLM, AutoTokenizer

# Define the path to your model and tokenizer
model_path = "E:/SignSpeak/sign-language-detector-python-master/consolidated.07.pt"
tokenizer_path = "E:/SignSpeak/sign-language-detector-python-master/tokenizer_model"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Define your input text
text = "Hello my name is"

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt")

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_path)

# Generate text using the model
outputs = model.generate(**inputs, max_length=20, num_return_sequences=1)

# Decode and print the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)