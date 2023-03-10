# Fine-tune GPT-2 model

Welcome to my GitHub repository where I explore the potential of fine-tuning the GPT-2 language model with specific type
of sentences like job experience descriptions written in sites like LinkedIn.

## Goal: Create fake job experience descriptions

The aim of this experiment is to train the model to generate new job experiences for a given role, using a dataset of
real job experiences.

Although the practical applications of this may be limited, it served as a valuable learning opportunity for
understanding the fine-tuning process of a language model. Through this repository, I hope to share my insights and
findings on GPT-2's capabilities and limitations in generating job experiences.

The ultimate goal was to create a model that, given a sentence such as "As a Software Engineer, I ", generates a
complete, relevant sentence related to the job title ("Software Engineer"). Using the default GPT-2 model to complete
the sentence "As a Software Engineer, I " may result in something like:

```text
As a Software Engineer, I have to confess I was not thrilled with the use of Linux in my own process. It's not a
free software choice; there's a huge amount of work involved, and in the end of my days of code.
```

or

```text
As a Software Engineer, I can attest to that. He is a long time user of my Java projects. I found out about
the development process for my own open source platform of C++, and decided to make it my personal project.
```

...and we can see that these are not the kind of sentences to add to a job experience list as a Software Engineer :)

Test it here: https://huggingface.co/mpuig/job-experience

## Process

### Step 1 - Getting the dataset

The first step to retrain a language model is to obtain a good dataset. With this goal in mind, I did some research and
I found [this repository](https://github.com/florex/resume_corpus) with a multi-labeled dataset of resumes labeled with
occupations from the following repository, with around 30k resumes.

### Step 2 - Data wrangling

However, it is important to note that the quality and relevance of the dataset will have a significant impact on the
performance of the retrained model. For this reason I spent some time removing duplicate data, correcting errors, and
removing irrelevant or confidential information.

To train a LLM data needs to be in a concrete format, so the first step is to build a unique csv from all the resumes:

```bash
python step1.py
```

The output of the first step is the `job_experiences.csv` file with two columns: `title` and `description`. eg:

| title                                         | description                                                                                                          |
|-----------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| Autodesk Lead Database Administrator Autodesk | Created a custom monitoring solution for MariaDB MaxScale using the MariaDB Connect Engine.                          |
| Java Senior Software Engineer Java Senior     | Expertise in software development methodologies like Waterfall, Agile development and Rapid Application Development. |
| Project Manager                               | Praised for efficiency and reducing labor costs.                                                                     |

The next step, using a jupyter notebook, is to do some data wrangling and build the final csv datasets:

```bash
jupyter notebook
```

The notebook uses [flashtext](https://github.com/vi3k6i5/flashtext) to search for different uses of the same title and
group them under the same name. E.g.

```python
    ...
"web developer": ["web developer", "web programmer"],
"RoR developer": ["RoR developer", "ruby on rails", "RoR", "Ruby"],
...
```

With the correct titles, we run a typical clean up to the descriptions (remove bullet points, html tags, quotes,
duplicates, URLs, very short descriptions,...)

With the previous aggregation, we obtain the following results:

```python
df2.groupby('keywords').count()

RoR developer,1381
analyst,46086
backend developer,21268
consultant,14797
customer service,6648
data engineer,2953
database administrator,32956
frontend developer,22669
java developer,82282
javascript developer,14007
network engineer,9214
```

Finally we use [spacy](https://spacy.io/) to filter only the rows with a description using a verb, non-3rd person
singular present (tag "VBP").

And we can prepare the sentences to write to the dataset with the correct format as a csv file:

```python
f"as a {title}, {description}<|endoftext|>"
```

The final csv has 395.401 rows.

Open the [jupyter notebook](step2.ipynb) to see the code

### Step 3 - Training

Once the data is cleaned, it can be further preprocessed, such as tokenizing and vectorizing the text, in order to
prepare it for training. This process is done opening the step3.ipynb in Google Colab, altogether with the training
data, and train the model.
After ~45 min of training, we obtain a new fine tuned model that can be used like this:

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

```bash
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

Clearly a better result than the initial ones.

## To reproduce the experiment

First, you need to install the following packages:

```bash
pip install jupyter torch torchvision torchaudio datasets ftfy flashtext pandas numpy scikit-learn accelerate
```

Run Jupyter notebooks:

```bash
jupyter notebook
```

## Alternatives to train a LLM with your own dataset

### Using HuggingFace examples:

Get the transformers code from Huggingface:

```bash
git clone https://github.com/huggingface/transformers.git
```

Go to the examples directory:

```bash
cd transformers/main/examples/pytorch/language-modeling/run_clm_no_trainer.py
```

Run the following python command:

```bash
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
