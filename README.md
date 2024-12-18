# Service Desk Ticket Classification

This is an expansion of a DataCamp project ["Service Desk Ticket Classification with Deep Learning"](https://app.datacamp.com/learn/projects/2148) and involves predicting the category of a service desk ticket based on the text of the message using a neural net.

## Data
The vocabulary (of 10146 words), messages and labels (5000 observations) are available in the [Data](Data) folder. The messages are anonymised, uncapitalised and split into individual words and punctuation in order. The labels are encoded numerically, but by looking at samples of the data, the categories can be deduced as:
- `0` : Banking Disputes
- `1` : Credit/Debit Card Disputes
- `2` : Debt Collection
- `3` : Online Payment/Money Transfers
- `4` : Mortgage/Loan Servicing

## The Task

The original task of the project is to split the dataset, create a classifier by training a CNN with a 1D Convolutional layer for 3 epochs on the training set and then to evaluate the predictions on the test set.

Instead, in the [notebook](notebook.ipynb) I've split the dataset into training, validation and test chunks, using the first two for model selection - to train the CNN as long as possible, while preventing overfitting. The classification report is available inside the notebook.

## What's next

There is a couple of directions I intend to expand this into: 
 1. Create a tool so that the model can be called (and a prediction of the category of a new ticket can be displayed) without having to interract with a jupyter notebook. 
 2. Improve preprocessing
    - tokenizer: currently tokenizing happens by splitting the text by the whitespaces (already done for us) and doesn't relate words such as "use" and "used". Perhaps SpaCy, Moses or a BertTokenizer can help increase performance
    - embedding: I've used torch.nn.Embedding for that job. Fine-tuning a pre-trained embedder might result in better accuracy.
 3. Play around with the architecture and see if it can produce a better classifier.
 4. Compare results (trade-off between accuracy and cost/resource consumption) with LLMs such as GPT-4o mini or Llama-3.2-1B.
