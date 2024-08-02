# Main app code
"""
As a highly skilled medical practitioner specializing in image analysis, you are tasked with
        examining medical images of a patient for a hospital. Your expertise is crucial in identifying
        anomalies, diseases, or health issues present in the image.
"""
# importing libraries
import streamlit as st
from pathlib import Path
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

#import key
from api_key import api_key


class Diagnosis:
    def __init__(self):
        self.uploaded_file = None
        self.submit_button = None
        self.gen_config = {}
        self.safety_settings = {}
        self.model = None
        self.imagedata = None
        self.image = []
        self.prompt = []
        self.response = None
        self.system_prompt = """ 
        Look at the image and give the details in a structured approach outlined below with the 2 headings:
        1. Detailed Analysis: Thorough analysis of image, focusing on identifying anomalies.
        2. Findings Report: Document all observed signs of anomalies or disease. Clearly articulate
        these findings.
        
        Important:
        1. Scope of response: Only if image pertains to human issues.
        2. Clarity of image: If image is unclear, you can mention "Unable to determine as image is unclear"
        """

    def model_setup(self):
        # Configure genai with api key
        genai.configure(api_key = api_key)

        # Create the model
        self.gen_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        # Apply safety settings
        self.safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH
            }


        # Configure model
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash",
                                           generation_config=self.gen_config,
                                           safety_settings=self.safety_settings)
    def create_page(self):
        st.set_page_config(page_title="Medical Image Diagnosis", page_icon=":computer:")
        st.title("Medical Image Diagnosis")
        st.subheader("This Application gives a Diagnosis based on the image provided by the Patient.")
        self.uploaded_file = st.file_uploader("Upload the image", type=["png","jpg","jpeg"])

        if self.uploaded_file:
            st.image(self.uploaded_file, width=300, caption="Uploaded image")

        self.submit_button = st.button("Generate Diagnosis")
        if self.submit_button:
            # Processing image
            self.imagedata = self.uploaded_file.getvalue()
            self.image = [
                {"mime_type": "image/png",
                 "data": self.imagedata
                 },
            ]
            self.prompt = [
                "\n",
                self.image[0],
                self.system_prompt
            ]

            self.response = self.model.generate_content(self.prompt)
            st.write(self.response.text)

    def __call__(self):
        self.model_setup()
        self.create_page()



