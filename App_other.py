import streamlit as st
import pandas as pd
import plotly.express as px
import nltk
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from urllib.request import urlopen
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Download necessary NLTK data
nltk.download('punkt_tab')

st.title("CSV & NLP Processing App with ML Classification")

# Upload CSV file or Read from URL
option = st.radio("Choose Data Input Method:", ("Upload CSV", "Read from URL"))

df = None
if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
elif option == "Read from URL":
    url = st.text_input("Enter CSV URL:")
    if url:
        df = pd.read_csv(urlopen(url))

if df is not None:
    st.write("### Raw Data")
    st.write(df.head())
    
    # Display Column Types
    st.write("### Column Data Types")
    st.write(df.dtypes)
    
    # Data Cleaning
    st.write("### Data Cleaning")
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    st.write("Data after cleaning:")
    st.write(df.head())
    
    # Sidebar for Chart Selection
    st.sidebar.header("Choose a Chart Type")
    chart_type = st.sidebar.selectbox("Select Chart", ["Histogram", "Correlation Heatmap", "Box Plot", "Pair Plot", "Line Plot", "Scatter Plot", "Map Plot", "Word Cloud", "Sentiment Analysis", "Text Classification"])
    
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    if chart_type == "Word Cloud" and categorical_columns:
        text_column = st.sidebar.selectbox("Select Text Column", categorical_columns)
        text_data = " ".join(df[text_column].dropna().astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)
    
    elif chart_type == "Sentiment Analysis" and categorical_columns:
        text_column = st.sidebar.selectbox("Select Text Column", categorical_columns)
        df['Sentiment'] = df[text_column].dropna().astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
        st.write(df[[text_column, 'Sentiment']].head())
        fig = px.histogram(df, x='Sentiment', nbins=20, title="Sentiment Distribution")
        st.plotly_chart(fig)
    
    elif chart_type == "Text Classification" and categorical_columns:
        text_column = st.sidebar.selectbox("Select Text Column", categorical_columns)
        label_column = st.sidebar.selectbox("Select Label Column", df.columns)
        
        if text_column and label_column:
            X = df[text_column].astype(str)
            y = df[label_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = make_pipeline(TfidfVectorizer(), MultinomialNB())
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            
            st.write(f"### Model Accuracy: {acc:.2f}")
            st.text("Classification Report:")
            st.text(report)
    
    elif chart_type and numeric_columns:
        if chart_type == "Histogram":
            selected_column = st.sidebar.selectbox("Select column for histogram", numeric_columns)
            fig = px.histogram(df, x=selected_column, marginal="box", nbins=30)
            st.plotly_chart(fig)
        
        elif chart_type == "Correlation Heatmap":
            fig = px.imshow(df[numeric_columns].corr(), text_auto=True, color_continuous_scale='RdBu_r')
            st.plotly_chart(fig)
        
        elif chart_type == "Box Plot":
            selected_y = st.sidebar.selectbox("Select Y-axis for box plot", numeric_columns)
            selected_hue = st.sidebar.selectbox("Select category (optional)", categorical_columns, index=0) if categorical_columns else None
            fig = px.box(df, y=selected_y, color=selected_hue)
            st.plotly_chart(fig)
        
        elif chart_type == "Pair Plot":
            fig = px.scatter_matrix(df, dimensions=numeric_columns)
            st.plotly_chart(fig)
        
        elif chart_type == "Line Plot":
            selected_x = st.sidebar.selectbox("Select X-axis", numeric_columns)
            selected_y = st.sidebar.selectbox("Select Y-axis", numeric_columns)
            fig = px.line(df, x=selected_x, y=selected_y)
            st.plotly_chart(fig)
        
        elif chart_type == "Scatter Plot":
            selected_x = st.sidebar.selectbox("Select X-axis", numeric_columns)
            selected_y = st.sidebar.selectbox("Select Y-axis", numeric_columns)
            selected_hue = st.sidebar.selectbox("Select category (optional)", categorical_columns, index=0) if categorical_columns else None
            fig = px.scatter(df, x=selected_x, y=selected_y, color=selected_hue)
            st.plotly_chart(fig)
        
        elif chart_type == "Map Plot" and 'latitude' in df.columns and 'longitude' in df.columns:
            fig = px.scatter_mapbox(df, lat='latitude', lon='longitude', hover_data=df.columns,
                                    mapbox_style="open-street-map", zoom=3)
            st.plotly_chart(fig)
    else:
        st.write("No numeric columns found for visualization or invalid selection.")
