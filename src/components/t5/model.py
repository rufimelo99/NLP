from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
tqdm.pandas()

class t5:
    def __init__(self, name) -> None:
        self.tokenizer = T5Tokenizer.from_pretrained(name)
        self.model = T5ForConditionalGeneration.from_pretrained(name)

    def load_pre_trained_model(self, model_name) -> None:
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def encode_data_for_training_task(self, input_sequences, output_sequences, task_prefix = "Summarize: ",max_source_length = 512*10, max_target_length = 1024):
        encoding = self.tokenizer(
            [task_prefix + sequence for sequence in input_sequences],
            padding="longest",
            max_length=max_source_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

        # encode the targets
        target_encoding = self.tokenizer(
            output_sequences,
            padding="longest",
            max_length=max_target_length,
            truncation=True,
            return_tensors="pt",
        )
        labels = target_encoding.input_ids

        # replace padding token id's of the labels by -100 so it's ignored by the loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return input_ids, attention_mask, labels

    def train_model(self, model, input_ids, attention_mask, labels, optimizer, epochs = 1):
        for epoch in tqdm(range(epochs)):
            # forward pass
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            loss.item()

            loss.backward()
            optimizer.step()

    def save_model(self, model_name):
        self.model.save_pretrained(model_name)

    def generate_summaries(self, text_list, max_length = 1024):
        input_sequences = text_list
        encoding = self.tokenizer(
            input_sequences,
            padding="longest",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
        output = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)
