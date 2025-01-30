### HTML Autocorrect & AI Text Correction Tool

## Overview
This project is a **Streamlit-based web application** that allows users to correct HTML errors either manually or by using a **fine-tuned T5 model**. Users can upload a dataset of HTML errors and train an AI model for automatic correction. Alternatively, they can paste or upload HTML code for validation and automatic fixes.

## Features
- **Manual HTML Validation & Correction** using BeautifulSoup
- **AI-Based HTML Autocorrection** using a fine-tuned T5 model
- **Custom Model Training** with user-uploaded datasets
- **File Upload Support** for datasets and HTML files
- **Interactive UI** built with Streamlit

## Dependencies
Ensure you have the following libraries installed before running the application:

```bash
pip install streamlit pandas torch transformers scikit-learn beautifulsoup4
```

## How It Works
### 1. **Preprocessing HTML Errors**
- The function `preprocess_dataset(df)` takes a DataFrame containing HTML with common errors and corrects them.
- Fixes include missing alt attributes, unclosed tags, inline styles, deprecated tags, uppercase tags, incorrect nesting, and more.

### 2. **Manual HTML Validation**
- `validate_and_correct_html(html)` uses BeautifulSoup to correct common HTML errors without AI assistance.

### 3. **Custom Dataset & AI Model Training**
- Users can upload a CSV dataset containing `html` and `corrected_html` columns.
- The AI model (T5-small) is fine-tuned using this dataset via `train_model()`.

### 4. **AI-Powered HTML Correction**
- After training, the model can predict corrections for new HTML input using `correct_html_with_model()`.

## How to Run the App
1. Clone this repository:
```bash
git clone <repo-link>
cd <repo-directory>
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage
### **Training the Model**
1. Upload a CSV file with `html` and `corrected_html` columns.
2. Set training parameters (epochs, batch size, learning rate).
3. Click **Train Model** to fine-tune the AI.

### **Correcting HTML Code**
1. Paste HTML into the text area and click **Correct HTML**.
2. Or, upload an HTML file and click **Correct Uploaded HTML**.
3. If a trained model exists, AI will generate corrections. Otherwise, BeautifulSoup will validate and fix common issues.

## File Formats
### CSV Dataset Example:
| html | corrected_html |
|------|--------------|
| `<img src='image.jpg'>` | `<img src='image.jpg' alt='Image description'>` |
| `<center>Text</center>` | `<div style='text-align: center;'>Text</div>` |

### HTML Upload:
- Users can upload `.html` files for correction.

## Model Storage
- After training, the model is saved in `./html_correction_model/`.
- The trained tokenizer is also saved for later use.

## Future Enhancements
- Support for **larger models** like T5-base.
- Additional HTML error correction rules.
- Integration with **real-time HTML validators**.
- Support for **GPT-based correction models**.
- Add a **REST API** for external integration.


