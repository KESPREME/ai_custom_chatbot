import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# To Load the pre-trained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# To Set the device to CPU
device = 'cpu'
model.to(device)

# To Define a function to generate text
def generate_text(prompt, max_length=50, num_return_sequences=1):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=num_return_sequences, do_sample=True, top_k=50, top_p=0.95, num_beams=1)

    generated_text = [tokenizer.decode(output[i], skip_special_tokens=True) for i in range(num_return_sequences)]
    return generated_text

# Example usage
prompt = input("Enter a prompt: ")
generated_text = generate_text(prompt, max_length=100, num_return_sequences=3)

for text in generated_text:
    print(text)
    print()

    import numpy as np
    from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

# To Load the pre-trained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# To Load your dataset

dataset = load_dataset('text', data_files='dialogs.txt') #Example for loading a dataset 

# To Preprocess the dataset
def preprocess_function(examples):
    inputs = [text for text in examples["text"]]
    targets = [text for text in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
    labels = model_inputs.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return model_inputs, labels

dataset = dataset.map(preprocess_function, batched=True)

# To Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# To Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
)

# To Fine-tune the model
trainer.train()

# To Save the finetuned model
trainer.save_model('Nipuns_finetuned_chatbot_model')