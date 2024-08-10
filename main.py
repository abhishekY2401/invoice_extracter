import streamlit as st
import os
import pathlib
import textwrap
import fitz
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

print()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to load OpenAI model and get respones


def get_gemini_response(prompt, text=None, image=None):
    model = genai.GenerativeModel('gemini-1.5-flash')
    # Prepare input data for the Gemini model
    inputs = [prompt]
    if text:
        inputs.append(text)
    if image:
        inputs.append(image)
    response = model.generate_content(inputs)
    return response.text


def extract_text_from_pdf(uploaded_file):
    # Extract text from the PDF file
    pdf_text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as pdf_document:
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pdf_text += page.get_text()

    return pdf_text


# Function to process image file
def process_image_file(uploaded_file):
    image = Image.open(uploaded_file)
    return image


# initialize our streamlit app
st.set_page_config(page_title="Invoice Information Extracter")
st.header("Invoice Details Application")

# input = st.text_input("Input Prompt: ", key="input")
uploaded_file = st.file_uploader(
    "Choose a file", type=["jpg", "jpeg", "png", "pdf"])

submit = st.button("Analyze Invoice")
input_prompt = """
               You are an expert in understanding invoices.
               You will receive input images and pdfs as invoices &
               you will have to only extract customer, product details and total amount
               based on the input image. Please represent the above mentioned data in proper readable 
               format.
               """
# If submit button is clicked
if submit and uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        st.write("PDF file uploaded.")
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(uploaded_file)
        response = get_gemini_response(input_prompt, text=pdf_text)
        st.subheader("The Response is")
        st.write(response)
    else:
        st.write("Image file uploaded.")
        image = process_image_file(uploaded_file)
        response = get_gemini_response(input_prompt, image=image)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        st.subheader("The Response is")
        st.write(response)
