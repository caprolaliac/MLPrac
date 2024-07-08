import os
import streamlit as st
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama

ollama_model = Ollama(model="phi3")
template = """
              Recognize all aspect terms with their corresponding sentiment polarity in the given review delimited by triple quotes. The aspect terms are nouns or phrases appearing in the review that indicate specific aspects or features of the product/service. Determine the sentiment polarity from the options ["positive", "negative", "neutral"]. Answer in the format ["aspect", "sentiment"] without any explanations. If no aspect term exists, then only answer "[]".
               ```{text}```
           """
prompt = PromptTemplate(template=template, input_variables=["text"])
llm_chain = LLMChain(prompt=prompt, llm=ollama_model)

@st.cache_data
def get_sentiment(user_input):
    return llm_chain.run(user_input)

def main():
    st.title("Sentiment Analysis")
    query = st.text_input("Enter your query:")
    if st.button("Submit"):
        result = get_sentiment(query)
        st.write(result)
        
if __name__ == "__main__":
    main()