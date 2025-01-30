import streamlit as st
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split

# Function to preprocess the dataset for HTML correction
def preprocess_dataset(df):
    # Corrects various HTML issues based on flag columns
    def correct_html(row):
        html = row["html"]
        
        # Add missing alt text for images
        if row["missing_alt"] == 1 and "img" in html:
            html = html.replace("<img", "<img alt='Image description'")

        # Ensure unclosed <p> tags are properly closed
        if row["unclosed_tag"] == 1:
            if "<p>" in html and "</p>" not in html:
                html += "</p>"

        # Replace inline style with class attribute
        if row["inline_style"] == 1:
            if "style=" in html:
                html = html.replace("style=", "class=")

        # Update deprecated <center> tag to modern <div> with inline styles
        if row["deprecated_tag"] == 1:
            if "<center>" in html:
                html = html.replace("<center>", "<div style='text-align: center;'>").replace("</center>", "</div>")

        # Convert uppercase tags to lowercase
        if row["uppercase_tag"] == 1:
            html = html.lower()

        # Ensure unquoted attributes are properly quoted
        if row["unquoted_attr"] == 1:
            if "href=example.com" in html:
                html = html.replace("href=example.com", "href='example.com'")

        # Replace <b> and <i> tags with semantically correct tags
        if row["b_or_i_tag"] == 1:
            html = html.replace("<b>", "<strong>").replace("</b>", "</strong>")
            html = html.replace("<i>", "<em>").replace("</i>", "</em>")

        # Fix incorrect tag nesting
        if row["incorrect_nesting"] == 1:
            if "<p><div>" in html:
                html = html.replace("<p><div>", "<div><p>").replace("</p></div>", "</div></p>")

        # Add missing DOCTYPE declaration if necessary
        if row["missing_doctype"] == 1:
            if not html.strip().lower().startswith("<!doctype html>"):
                html = f"<!DOCTYPE html>\n{html}"

        # Replace non-semantic div with semantic section
        if row["missing_semantic"] == 1:
            if "<div>Non-semantic container</div>" in html:
                html = html.replace("<div>Non-semantic container</div>", "<section>Non-semantic container</section>")

        return html
    
    # Apply corrections to the entire dataset
    df["corrected_html"] = df.apply(correct_html, axis=1)
    return df

# Function to validate and correct HTML using BeautifulSoup
def validate_and_correct_html(html):
    try:
        # Parse HTML using BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        
        # Define tag replacements
        replacements = {
            "center": ("div", {"class": "center", "style": "text-align: center;"}),
            "b": ("strong", {}),
            "i": ("em", {}),
            "font": ("span", {})
        }
        
        # Replace tags based on predefined mappings
        for tag_name, (new_tag, attrs) in replacements.items():
            for tag in soup.find_all(tag_name):
                new_tag_obj = soup.new_tag(new_tag, **attrs)
                new_tag_obj.string = tag.string if tag.string else ""
                tag.replace_with(new_tag_obj)
        
        # Add alt text to images if missing
        for tag in soup.find_all("img"):
            if not tag.get("alt"):
                tag["alt"] = "Image description"
        
        # Prettify and add DOCTYPE if missing
        corrected_html = f"<!DOCTYPE html>\n{soup.prettify()}" if not html.strip().lower().startswith("<!doctype html>") else soup.prettify()
        
        return corrected_html
    except Exception as e:
        st.error(f"Error processing HTML: {e}")
        return html

# Custom Dataset class for HTML correction model
class HTMLCorrectionDataset(Dataset):
    def __init__(self, tokenizer, data, max_length=128):
        """
        Args:
            tokenizer (T5Tokenizer): The tokenizer for the model.
            data (DataFrame): The dataset containing HTML and corrected HTML.
            max_length (int): The maximum length of input tokens.
        """
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get input and target HTML for training
        input_text = self.data.iloc[idx]["html"]
        target_text = self.data.iloc[idx]["corrected_html"]
        
        # Tokenize input and target HTML
        inputs = self.tokenizer(input_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        targets = self.tokenizer(target_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        
        # Return the tokenized data
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": targets["input_ids"].squeeze()
        }

# Function to train the model for HTML correction
def train_model(df, epochs=3, batch_size=8, learning_rate=5e-5):
    try:
        # Load pre-trained tokenizer and model
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
        
        # Split the data into training and evaluation datasets
        train_df, eval_df = train_test_split(df, test_size=0.2)
        train_dataset = HTMLCorrectionDataset(tokenizer, train_df)
        eval_dataset = HTMLCorrectionDataset(tokenizer, eval_df)

        # Set up training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
        )
        
        # Initialize the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset  
        )

        # Train the model
        trainer.train()

        # Save the trained model and tokenizer
        model.save_pretrained("./html_correction_model")
        tokenizer.save_pretrained("./html_correction_model")

        return model, tokenizer
    except Exception as e:
        st.error(f"Error during training: {e}")
        return None, None

# Function to correct HTML using the trained model
def correct_html_with_model(input_html, model, tokenizer):
    try:
        # Tokenize the input HTML
        inputs = tokenizer(input_html, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
        
        # Generate corrected HTML using the model
        outputs = model.generate(**inputs, max_length=128)
        
        # Decode the output into human-readable text
        corrected_html = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return corrected_html
    except Exception as e:
        st.error(f"Error correcting HTML: {e}")
        return input_html

# Streamlit app interface
st.title("HTML Autocorrect & AI Text Correction Tool")
st.write("Upload your dataset to fine-tune the AI for text correction or paste HTML for automatic fixes.")

# Upload training data (CSV)
train_file = st.file_uploader("Upload training data (CSV)", type=["csv"])

if train_file is not None:
    df = pd.read_csv(train_file)
    df = preprocess_dataset(df)

    # Model training parameters
    epochs = st.number_input("Epochs", min_value=1, max_value=10, value=3)
    batch_size = st.number_input("Batch Size", min_value=1, max_value=32, value=8)
    learning_rate = st.number_input("Learning Rate", min_value=1e-5, max_value=1e-3, value=5e-5)

    if st.button("Train Model"):
        with st.spinner("Training model..."):
            model, tokenizer = train_model(df, epochs, batch_size, learning_rate)
            if model and tokenizer:
                st.success("Model trained successfully!")

# Input HTML to be corrected
html_input = st.text_area("Paste your HTML code here:", height=300)

if st.button("Correct HTML"):
    if html_input.strip():
        # Use the trained model for correction if available
        if "model" in locals() and "tokenizer" in locals():
            corrected_html = correct_html_with_model(html_input, model, tokenizer)
        else:
            corrected_html = validate_and_correct_html(html_input)
        st.write("### Corrected HTML:")
        st.code(corrected_html, language="html")

# Upload an HTML file
uploaded_file = st.file_uploader("Or upload an HTML file:", type=["html"])
if uploaded_file is not None:
    html_input = uploaded_file.read().decode("utf-8")
    st.write("### Uploaded HTML:")
    st.code(html_input, language="html")
    
    # Correct the uploaded HTML
    if st.button("Correct Uploaded HTML"):
        if "model" in locals() and "tokenizer" in locals():
            corrected_html = correct_html_with_model(html_input, model, tokenizer)
        else:
            corrected_html = validate_and_correct_html(html_input)
        st.write("### Corrected HTML:")
        st.code(corrected_html, language="html")
