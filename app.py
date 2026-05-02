import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression,LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import time

#load css
with open ("style.css")as f:
        css = f.read()
        st.markdown(f"<style>{css}</style>",unsafe_allow_html= True)

#----Splash Screen -----
placeholder = st.empty()
with placeholder.container():
    st.markdown("""<div style='text-align:center;margin-top:150px;'>
                <h1 style='color:#2196F3;font-size:45px;'> Smart Expense Tracker</h1> 
                <p style='font-size:20px;'>Initialising your data.....</p>
                <p style='font-size:14px;color:gray;'>Please wait</p>
                </div>""",unsafe_allow_html=True)

    time.sleep(2)
    #remove splash
placeholder.empty()
#Spinner
with st.spinner("Loading your dashboard....."):
    time.sleep(2)


 #Main app       
st.set_page_config(page_title="Smart Expense Tracker",layout="wide")

st.info("Track, analyse and predict your expenses smartly!")

# Load data
try:
    data = pd.read_csv("expenses.csv")
except:
    data = pd.DataFrame(columns=["Date","Description","Amount","Category"])

 #train ML model
model = None
if len(data)>5:
    X = data["Description"]
    y = data["Category"]

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_vec,y) 

#UI

st.markdown("<h1>Smart Expense Tracker</h1>",unsafe_allow_html=True)
st.markdown("<h2>Add Expense</h2>",unsafe_allow_html=True)
col1,col2 = st.columns(2)
with col1:
    desc = st.text_input("Description")
with col2:
    amount = st.number_input("Enter amount", min_value=0.0)


# Button
if st.button("Add Expense"):
    if model:
        pred = model.predict(vectorizer.transform([desc]))[0]
    else:
        pred = "Other"
    new_data = pd.DataFrame([[datetime.now().strftime("%Y-%m-%d"), desc, amount,pred]],
                            columns=["Date","Description","Amount","Category"])

    data = pd.concat([data, new_data], ignore_index=True)
    data.to_csv("expenses.csv", index=False)

    st.success(f" Expense Added! Category : {pred}")
    #alert
    avg = data["Amount"].mean()
    st.write("Average:", avg)

# Overspending Alert
    if amount > 1.5*avg and amount > 0:
        st.warning("you are spending more than usual!")

#Summary Dashboard
if not data.empty :
    total = data["Amount"].sum()
    avg = data["Amount"].mean()
    c1,c2  = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="card">
            <h3>Total Spending</h3>
            <h2 style='color:#E74C3C;'>₹{round(total,2)}</h2>
        </div>
        """, unsafe_allow_html=True)

    # 🟢 SECOND CARD
    with c2:
        st.markdown(f"""
        <div class="card">
            <h3>Average Spending</h3>
            <h2 style='color:#27AE60;'>₹{round(avg,2)}</h2>
        </div>
        """, unsafe_allow_html=True)

# Show table
st.markdown("<h2>Expense History</h2>",unsafe_allow_html=True)
st.dataframe(data,use_container_width=True,height = 300)

# Graph
st.markdown("<h2>Monthly Spending</h2>",unsafe_allow_html=True)

if not data.empty:
    # Convert to datetime safely
     data["Date"] = pd.to_datetime(data["Date"], errors='coerce')

# Remove invalid rows
     data = data.dropna(subset=["Date"])

# Now use dt safely
     monthly = data.groupby(data["Date"].dt.month)["Amount"].sum()
     st.line_chart(monthly)  

    #category Distribution
if not data.empty:
    st.bar_chart(data["Category"].value_counts())

    #add pie chart
st.markdown("<h2>Category Share</h2>",unsafe_allow_html=True)
if not data.empty:
    category_data  = data["Category"].value_counts()
    fig, ax = plt.subplots(figsize=(3,3))
    ax.pie(category_data,labels=category_data.index,autopct='%1.1f%%',startangle=90,textprops={'fontsize':10})
    ax.set_title("Spending by Category",fontsize=13,pad =12)
    ax.axis('equal')
    #center chart
    col1,col2,col3 = st.columns([1,2,1])
    with col2:
      st.markdown('<div class="card">',unsafe_allow_html=True)
      st.pyplot(fig)
      st.markdown('</div>',unsafe_allow_html=True)


#Filter option
st.subheader("Filter by Category")
if not data.empty:
    selected = st.selectbox("Choose Category",["All"]+list(data["Category"].unique()))
    if selected != "All":
        filtered_data = data[data["Category"] == selected]
    else:
        filtered_data = data

    st.dataframe(filtered_data,use_container_width=True,height =300)



#Future Prediction
st.markdown("<h2> Future Prediction</h2>",unsafe_allow_html=True)
if not data.empty:
    data["Date"] = pd.to_datetime(data["Date"],errors='coerce')
    data= data.dropna(subset=["Date"])

    monthly = data.groupby(data["Date"].dt.to_period("M"))["Amount"].sum().reset_index()
    monthly["Date"]=monthly["Date"].dt.to_timestamp()
    months_available = len(monthly)

    #case1: not enough data
    if months_available < 3:
        remaining = 3 - months_available
        st.warning(f"you have data for {months_available}month(s).")
        st.info(f"Add {remaining}more month(s) of expenses to enable prediction.")
        #show current monthly data
        st.write("Your current monthly spending:")
        st.dataframe(monthly) 
    
    #Case 2: Enough data
    else:
        X= np.arange(len(monthly)).reshape(-1,1)
        y= monthly["Amount"].values
        #train ml model
        model_lr = LinearRegression()
        model_lr.fit(X,y)
        next_month = np.array([[len(monthly)]])
        pred = model_lr.predict(next_month)[0]
        st.success("Prediction Ready!")
        st.markdown(f"""<div class="card"><h3>Predicted next month spending:</h3><h2> **{round(pred,2)}** </h2></div>""",unsafe_allow_html=True)

        #show trend graph including prediction
        fig, ax = plt.subplots()
        ax.plot(X,y,marker='o',label="Past Spending")
        ax.plot(months_available,pred,'ro',label="Prediction")
        ax.set_title("Spending Trend + Prediction")
        ax.set_xlabel("Months")
        ax.set_ylabel("Amount")
        ax.legend()
        st.pyplot(fig)




