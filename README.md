# NSFW-finetuned-on-llm using google colab
Here we have taken falcon 7B as the LLM and finetuned NSFW dataset with it# NSFW Classification using Falcon-7b

This repository contains code for training a neural network model to classify NSFW (Not Safe for Work) content using the Falcon-7b model. The model is fine-tuned on the NSFW dataset and utilizes the Peft library for efficient training.

## Usage

To use the NSFW classification code:

1. Install the required dependencies:

   - trl
   - transformers
   - accelerate
   - peft
   - datasets
   - bitsandbytes
   - einops
   - wandb

2. Load the NSFW dataset.
3. Load the Falcon-7b model and tokenizer.
4. Configure the Peft library for the LoRA algorithm.
5. Set the training arguments.
6. Train the model using the SFTTrainer.

Please refer to the code in the `flacon7b.ipynb` notebook for the detailed implementation.

Note: Running the training code may take time and require sufficient computational resources(but can be run on a free google colab version).
## Run the model
```
text = "your input here"

inputs = tokenizer(text, return_tensors="pt")

inputs.pop("token_type_ids", None)

outputs = model.generate(**inputs, max_length=100)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

```
