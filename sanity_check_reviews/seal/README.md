

## Setup

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Activate the virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Login to hugging face to download the models:
   ```bash
   huggingface-cli login
   ```


## Run the script

To run the first experiment regarding knowledge incorporation, use the following command:

```bash
python knowledge_incorporation.py
```

To run the second experiment about few shot learning with unsupervised fine tuning, use the following command:

```bash
python few_shot_learning.py
```