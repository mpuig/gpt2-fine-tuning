# Job Experiences

Welcome to my Github repository where I explored the possibilities of fine-tuning the language model GPT-2 with real job
experiences. In this experiment, I aimed to train the model to generate new job experiences from a given role, using a
dataset of real job experiences.

While this may not have practical applications in the real world, it served as a valuable learning experience for
understanding the process of fine-tuning a language learning model. Through this repository, I hope to share my insights
and findings on the capabilities and limitations of GPT-2 in generating job experiences.

The goal was to obtain a model where, starting with a sentence like "As a Software Engineer, I", the model generates a
complete new sentence related to the job title ("Software Engineer").

## Process

### Step 1 - Getting the training dataset

I downloaded multi-labeled dataset of resumes labeled with occupations from the following repository, with around 30k
resumes:

https://github.com/florex/resume_corpus

### Step 2 - Data wrangling

To train a LLM data needs to be in a concrete format, so the first step is to build a unique csv from all the resumes:

```
python step1.py
```

And then, using a jupyter notebook to do some data wrangling and build the final csv datasets:

```
jupyter notebook
```

Open http://localhost:8888/notebooks/step2.ipynb

### Step 3 - Training

Finally, upload the step3.ipynb to Google Colab, altogether with the training data, and train the model.

At the end of the training, you'll have a model that can be used like this:

```python
model.eval()

prompt = "As a sowtware architect, I"

generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
generated = generated.to(device)

print(generated)

sample_outputs = model.generate(
    generated,
    do_sample=True,
    top_k=50,
    max_length=300,
    top_p=0.95,
    num_return_sequences=5
)

for i, sample_output in enumerate(sample_outputs):
    print("{}: {}\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
```

Obtaining an output like this:

```
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.

tensor([[ 1722,   257, 45125,    83,  1574,  7068,    11,   314]],
       device='cuda:0')
0: As a sowtware architect, I identified, researched, designed and built the architecture of the SCCM platform.

1: As a sowtware architect, I worked closely with the network engineers to integrate the business logic to the application.

2: As a sowtware architect, I performed manual installation of the RTA server using SSIS.

3: As a sowtware architect, I coordinated with the Marketing department to identify problems encountered and provide solutions to resolve them.

4: As a sowtware architect, I used various types of data base and Hadoop/Hibernate relational databases, including MySQL, MongoDB, Cassandra to extract information from the database and to insert/delete data from the database.

```

## To reproduce the experiment

First, you need to install the following packages:

```
pip install jupyter torch torchvision torchaudio datasets ftfy flashtext pandas numpy scikit-learn accelerate
```

Run Jupyter notebooks:

```
jupyter notebook
```

## Alternatives to train a LLM with your own dataset

### Using HuggingFace examples:

Get the transformers code from Huggingface:

```
git clone https://github.com/huggingface/transformers.git
```

Go to the examples directory:

```
cd transformers/main/examples/pytorch/language-modeling/run_clm_no_trainer.py
```

Run the following python command:

```
python run_clm_no_trainer.py \
    --model_name_or_path gpt2 \
    --train_file dataset_train.csv \
    --validation_file dataset_test.csv \
    --output_dir test-clm
```

### Using Andrej Karpathy's nanoGPT

https://github.com/karpathy/nanoGPT

_The simplest, fastest repository for training/finetuning medium-sized GPTs. It is a rewrite of minGPT that prioritizes
teeth over education. Still under active development, but currently the file train.py reproduces GPT-2 (124M) on
OpenWebText, running on a single 8XA100 40GB node in 38 hours of training. The code itself is plain and readable:
train.py is a ~300-line boilerplate training loop and model.py a ~300-line GPT model definition, which can optionally
load the GPT-2 weights from OpenAI. That's it._
