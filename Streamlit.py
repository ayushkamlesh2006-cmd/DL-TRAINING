import streamlit as st 
import pandas as pd
import numpy as np


st.title("Hello, Streamlit !")
st.write(":streamlit: This is yours first streamlit app :streamlit:")
st.text("Lets get started ")


#Conditional logic with weights

name= st.text_input("Enter your name : ")
if st.button("Greet"):
    st.success(f"Hello, {name} !")

#Display Data and Charts 
df = pd.DataFrame(np.random.randn(20,4), columns =['A' , 'B' , 'C' , 'D'])
st.line_chart(df)
st.bar_chart(df)



#File Uploading  and caching topics
upload_file = st.file_uploader("Upload a CSV :", type='csv')
if upload_file:
    df = pd.read_csv(upload_file)
    st.dataframe(df)


st.title("Text and Markdown Demo")
st.header("This is a header")
st.subheader("This is a sunheader")
st.markdown("**Bold**, *Italic*, 'Code', [Link](https://www.youtube.com/shorts/Ay8lynMZ4mE)")
st.code("for i in range(5): print(i)",language="python")


st.text_input("What's your name ?")
st.text_area("Write something...")
st.number_input("Pick a number",min_value=0,max_value=100)
st.slider("Choose a range", 0 , 100)
st.selectbox("Select a fruit",['Apple','Banana','Mango'])
st.multiselect("choose toppings", ['Cheese' , 'Tomato' , 'Olives' , "Chocolate"])
st.radio("Pick one ", ["Option A", "Option B"])
st.checkbox("I agree to the terms")

if st.checkbox("Show Details"):
    st.info("Here are more details.......")


option = st.selectbox("Choose a View", ["Show Chart", "Show Table"])
if option == "Show Chart":
    st.write("Here is a chart")
else:
    st.write("Here is a table") 

with st.form("Login Form :"):
    username = st.text_input("Username :")
    password = st.text_input("Password :", type="password")
    submit_button = st.form_submit_button("Login")

    if submit_button:
        st.success(f"Welcome, {username} !")
    else:
        st.warning("Please enter your credentials.")


st.image(
    "https://images.unsplash.com/photo-1518770660439-4636190af475",
    caption="Technology and Electronics",
    width="stretch"
)


st.video("https://www.youtube.com/watch?v=Ay8lynMZ4mE")