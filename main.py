import streamlit as st
import os
import logging
import pathlib
import textwrap
import fitz
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Get the API key from the environment
api_key = os.getenv("GOOGLE_API_KEY")
if api_key is None:
    logging.error("GOOGLE_API_KEY is not set in the environment variables.")
    st.error(
        "API key is missing. Please set the API key in the environment variables.")
else:
    genai.configure(api_key=api_key)

# Function to load OpenAI model and get respones


def get_gemini_response(prompt, text=None, image=None):
    logging.info("Starting LLM API call")
    model = genai.GenerativeModel('gemini-1.5-flash')
    inputs = [prompt]
    if text:
        logging.info(f"Text length: {len(text)} characters")
        text = text[:5000]  # Limiting text length
        inputs.append(text)
    if image:
        inputs.append(image)
    try:
        response = model.generate_content(inputs)
        logging.info("LLM API call successful")
        return response.text
    except Exception as e:
        logging.error(f"Failed to generate response: {e}")
        st.error(f"Failed to generate response: {e}")
        return None


def extract_text_from_pdf(uploaded_file):
    logging.info("Starting PDF text extraction")
    pdf_text = ""
    try:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as pdf_document:
            for page_num in range(len(pdf_document)):
                logging.info(f"Processing page {page_num}")
                page = pdf_document.load_page(page_num)
                pdf_text += page.get_text()
        logging.info("PDF text extraction completed")
    except Exception as e:
        logging.error(f"Failed to process PDF: {e}")
        st.error(f"Failed to process PDF: {e}")
        return None
    return pdf_text


# Function to process uploaded images
def input_image_setup(uploaded_file):
    logging.info("Setting up image input")
    if uploaded_file is not None:
        try:
            bytes_data = uploaded_file.getvalue()
            image_parts = [
                {
                    "mime_type": uploaded_file.type,
                    "data": bytes_data
                }
            ]
            logging.info("Image input setup successful")
            return image_parts
        except Exception as e:
            logging.error(f"Failed to process image: {e}")
            st.error(f"Failed to process image: {e}")
            return None
    else:
        logging.warning("No image file uploaded")
        raise FileNotFoundError("No file uploaded")


# initialize our streamlit app
st.set_page_config(page_title="Invoice Information Extracter")
st.header("Invoice Details Application")

# input = st.text_input("Input Prompt: ", key="input")
uploaded_file = st.file_uploader(
    "Choose a file", type=["jpg", "jpeg", "png", "pdf"])

image = None
pdf_text = None

if uploaded_file is not None:
    file_type = uploaded_file.type.split('/')[1]
    logging.info(f"Uploaded file type: {file_type}")

    if file_type in ["jpg", "jpeg", "png"]:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        image = input_image_setup(uploaded_file)
    elif file_type == "pdf":
        pdf_text = extract_text_from_pdf(uploaded_file)

submit = st.button("Analyze Invoice")
input_prompt = """
               You are an expert in understanding invoices.
               You will receive input images and pdfs as invoices &
               you will have to only extract customer, product details and total amount
               based on the input. Please represent the above mentioned data in proper readable 
               format. Also dont provide other details
               """

# If submit button is clicked
if submit:
    if image or pdf_text:
        response = get_gemini_response(
            input_prompt, text=pdf_text, image=image)
        if response:
            st.subheader("The Response is")
            st.write(response)
    else:
        logging.warning("No file content to process")
        st.error("No file content to process. Please upload a PDF or image.")
