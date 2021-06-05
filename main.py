### ****************************************** BOOKS - GOODREADS.CSV **********************************************
### **PRE-REQUSITES**
import time
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import surprise
import plotly_express as px
import warnings
import base64
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------------------------------------------------
# BACKEND CODE - READING CSV FILES
# Importing all data
books = pd.read_csv('DATA\\books.csv', encoding="ISO-8859-1")
ratings = pd.read_csv('DATA\\ratings.csv', encoding="ISO-8859-1")
book_tags = pd.read_csv('DATA\\book_tags.csv', encoding="ISO-8859-1")
tags = pd.read_csv('DATA\\tags.csv')
to_read = pd.read_csv('DATA\\to_read.csv')

# some other variables
tags_join_DF = pd.merge(book_tags, tags, left_on='tag_id', right_on='tag_id', how='inner')
books_merge_ratings = pd.merge(books, ratings)
books_df = pd.DataFrame(books)
userRatings = books_merge_ratings.pivot_table(index=['user_id'], columns=['title'], values='rating')
userRatings = userRatings.dropna(thresh=10, axis=1).fillna(0)
all_book_name = userRatings.columns
books_with_tags = pd.merge(books, tags_join_DF, left_on='book_id', right_on='goodreads_book_id', how='inner')
#---------------------------------------------------------------------------------------------------------------------
                                        # Self defined methods - EDA SECTION

def cloud_author(original_title_string, max_word, max_font, random):
    stop_words = set(STOPWORDS)
    original_title_string = " ".join(books['authors'])
    wc = WordCloud(background_color="white", colormap="hot", max_words=max_word,
                   stopwords=stop_words, max_font_size=max_font, random_state=random).generate(original_title_string)
    plt.figure(figsize=(100, 400))
    fig, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [8, 1]})
    axes[0].imshow(wc, interpolation="bilinear")
    for ax in axes:
        ax.set_axis_off()
    st.pyplot(fig)

def cloud_title(authors_string, max_word, max_font, random):
    stop_words = set(STOPWORDS)
    authors_string = " ".join(books['title'])
    wc = WordCloud(background_color="white", colormap="hot", max_words=max_word,
                   stopwords=stop_words, max_font_size=max_font, random_state=random).generate(authors_string)
    plt.figure(figsize=(100, 400))
    fig, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [8, 1]})
    axes[0].imshow(wc, interpolation="bilinear")
    for ax in axes:
        ax.set_axis_off()
    st.pyplot(fig)

#***********************************
i = 100
rows_count = []
while (i >= 1):
    rows_count.append(i)
    i = i - 1
#***********************************

# ---------------------------------------------------------------------------------------------------------------------
                                                  #UI Codes - SIDEBAR
pages = ["Get Recommendation", "Exploratory Data Analysis", "The Data", "Flipkart Dataset"]
section1 = st.sidebar.selectbox('', pages)

st.sidebar.header("RECOMMENDATION SYSTEM")
st.sidebar.write("Based on Book Author")
st.sidebar.write("Based on Book Tags")
st.sidebar.write("User's Preference")
st.sidebar.write("Item's Preference")

st.sidebar.header("GET INSIGHTS FROM DATA")
st.sidebar.write("check for null values")
st.sidebar.write("Top Authors")
st.sidebar.write("Languages")
st.sidebar.write("WordCloud")
st.sidebar.write("Correlation")

st.sidebar.header("ABOUT THE DATA")
st.sidebar.write("books.csv")
st.sidebar.write("ratings.csv")
st.sidebar.write("book_tags.csv")
st.sidebar.write("tags.csv")
st.sidebar.write("to_read.csv")

st.sidebar.header("FLIPKART DATASET")
st.sidebar.write("Recommendations")
#---------------------------------------------------------------------------------------------------------------------
                                         # UI Codes - RECOMMENDATION SECTION
#common section above
if section1 == "Get Recommendation":
    st.markdown("# RECOMMENDATION SYSTEM")

    #for gif image
#    file_ = open("confused_lines.gif", "rb")
#    contents = file_.read()
#    data_url = base64.b64encode(contents).decode("utf-8")
#    file_.close()
#    st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="gif" width = "750">',unsafe_allow_html=True)#
    st.image("books-shelf.jpg")

#                                        ---- This is for Book input section ----
    st.subheader("Provide us Book details:")
    book_name_selectbox = books_df['title'].tolist()
    option = st.selectbox(label='Select one:', options=book_name_selectbox)

    i=20
    list_of_recom = []
    while (i >= 1):
        list_of_recom.append(i)
        i = i-1
    innercol1, innercol2, innercol3 = st.beta_columns(3)
    innercol1.write("You've selected:")
    innercol2.info(option)
    no_of_recom = innercol3.selectbox('No. of Recommendations:', list_of_recom)

#                                            ---- AUTHOR/TAGS/BOTH/ITEM UI ----

    st.subheader("Get Recommendation based on Book attributes:")
    st.write("By which way would you should be recommended:")
    col1, col2, col3, col4 = st.beta_columns(4)
    checkbox1 = col1.checkbox("BOOK AUTHOR")
    checkbox2 = col2.checkbox("BOOK TAGS")
    checkbox3 = col3.checkbox("BOTH")
    checkbox4 = col4.checkbox("ITEM PREF")
#                                                    --- USER ---
    checkbox5 = st.checkbox("USER'S PREFERENCE")
    if checkbox5 == True:
        my_expander = st.beta_expander("Select User:")
        all_user_list = ratings['user_id'].head(100).tolist()
        selected_user_id = my_expander.selectbox("Select a user: ", options=all_user_list)
        rating_count = ratings[ratings['user_id'] == selected_user_id]
        my_expander.write(rating_count)
    submit_btn2 = st.button("SUBMIT", key=1)
#---------------------------------------------------------------------------------------------------------------------
                                         # Training Model - USER PREFERENCE
#                                                -- PEARSON SIMILARITY --
    item_similarity_pearson = userRatings.corr(method='pearson')
    #50 seconds to run this training set

    ratings = ratings[['user_id','book_id','rating']]
    ratings = ratings.iloc[:20000,:]
    reader = surprise.Reader(rating_scale=(1,5))
    dataset1 = surprise.Dataset.load_from_df(ratings, reader)

    #                                       --------  KNN BASIC MODEL  ---------
    from surprise import KNNBasic, accuracy
    from surprise.model_selection import train_test_split
    train1,test1 = train_test_split(dataset1,test_size=0.2)
#-----------------------------------------------------------------------------------------------------------------------
#                            ******* Radio click recommendation code logic goes here *******
#                                           --- AUTHOR / BOOK TAG / BOTH ---

    #                                                -- Author Based --
    if submit_btn2 == True and checkbox1 == True:
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.1)
            progress_bar.progress(i + 1)
        st.info("These are the Recommendations based on Author's Preference")

        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
        tfidf_matrix = tf.fit_transform(books['authors'])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        titles = books['title']
        indices = pd.Series(books.index, index=books['title'])

        def authors_recommendation(title):
            idx = indices[title]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:21]
            book_indices = [i[0] for i in sim_scores]
            return titles.iloc[book_indices]

        result_df = authors_recommendation(option).head(no_of_recom)
        st.table(result_df)
        st.success('I think these are the right book(s) for you! :smile:')

#                                                   -- Books tags Based --
    if submit_btn2 == True and checkbox2 == True:
        progress_bar2 = st.progress(0)
        for i in range(100):
            time.sleep(0.1)
            progress_bar2.progress(i + 1)
        st.info("These are the Recommendations based on Book Tags's Preference")

        books_with_tags = pd.merge(books, tags_join_DF, left_on='book_id', right_on='goodreads_book_id', how='inner')
        tf1 = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
        tfidf_matrix1 = tf1.fit_transform(books_with_tags['tag_name'].head(10000))
        cosine_sim1 = linear_kernel(tfidf_matrix1, tfidf_matrix1)
        titles1 = books['title']
        indices1 = pd.Series(books.index, index=books['title'])

        def tags_recommendation(title):
            idx = indices1[title]
            sim_scores = list(enumerate(cosine_sim1[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:21]
            book_indices = [i[0] for i in sim_scores]
            return titles1.iloc[book_indices]

        result_df = tags_recommendation(option).head(no_of_recom)
        st.table(result_df)
        st.success('I think these are the right book(s) for you! :smile:')

#                                                       -- Both --
    if submit_btn2 == True and checkbox3 == True:
        progress_bar3 = st.progress(0)
        for i in range(100):
            time.sleep(0.1)
            progress_bar3.progress(i + 1)
        st.info("These are the Recommendations based on BOTH Author and Tags")
        temp_df = books_with_tags.groupby('book_id')['tag_name'].apply(' '.join).reset_index()
        books = pd.merge(books, temp_df, left_on='book_id', right_on='book_id', how='inner')
        books['corpus'] = (pd.Series(
            books[['authors', 'tag_name']]
                .fillna('')
                .values
                .tolist())
                           .str.join(' '))
        tf_corpus = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
        tfidf_matrix_corpus = tf_corpus.fit_transform(books['corpus'])
        cosine_sim_corpus = linear_kernel(tfidf_matrix_corpus, tfidf_matrix_corpus)
        titles = books['title']
        indices1 = pd.Series(books.index, index=books['title'])

        def corpus_recommendation(title):
            idx = indices1[title]
            sim_scores = list(enumerate(cosine_sim_corpus[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:21]
            book_indices = [i[0] for i in sim_scores]
            return titles.iloc[book_indices]

        result_df = corpus_recommendation(option).head(no_of_recom)
        st.table(result_df)
        st.success('I think these are the right book(s) for you! :smile:')

    #                                                       -- ITEM --
    if submit_btn2 == True and checkbox4 == True:
        progress_bar4 = st.progress(0)
        for i in range(100):
            time.sleep(0.1)
            progress_bar4.progress(i+1)

        st.info("These are the Recommendations based on your Ratings")
        item_similarity_pearson = userRatings.corr(method='pearson')
        def get_similar_book_pearson_itemtoitem(bookname):
            similar_score = item_similarity_pearson[bookname] * (2.5)
            similar_score = similar_score.sort_values(ascending=False)
            return similar_score

        item_model = surprise.KNNBasic(k=40, sim_options={'name': 'pearson', 'user_based': False})
        item_model.fit(train1)
        preds = item_model.test(test1)
        accuracy.rmse(preds, verbose=True)

        recommended_pearson_item = get_similar_book_pearson_itemtoitem(option).index
        result_df = pd.DataFrame(data=recommended_pearson_item)
        result_df.iloc[1:].head(no_of_recom)
        result_list = []
        for ind in recommended_pearson_item[1:no_of_recom]:
            result_list.append(ind)
        result_df = pd.DataFrame(data=result_list)
        st.table(result_df)

        st.success('I think these are the right book(s) for you! :smile:')

#                                                      -- USER --

    if submit_btn2 == True and checkbox5 == True:
#        setting up data of specific user's ratings given to books
        progress_bar4 = st.progress(0)
        for i in range(100):
            time.sleep(0.1)
            progress_bar4.progress(i + 1)

        my_df = pd.merge(books, rating_count, left_index=True, right_index=True)
        my_df = my_df.loc[:, my_df.columns.intersection(['title','rating'])]
        user1_tuple = my_df.to_records(index=False)
        user1_list = list(user1_tuple)
        st.info("These are the Recommendations based on your Ratings")
        #st.write("User", selected_user_id, "has rated", user1_list)
        user1_list = zip(all_book_name, userRatings.iloc[0, :])

#USER BASED PEARSON SIMILARITY
        item_similarity_pearson = userRatings.corr(method='pearson')
# 50 seconds to run this training set

        ratings = ratings[['user_id', 'book_id', 'rating']]
        ratings = ratings.iloc[:20000, :]
        reader = surprise.Reader(rating_scale=(1, 5))
        dataset1 = surprise.Dataset.load_from_df(ratings, reader)

        def get_similar_book_pearson_user(book_name, user_rating):
            similar_score = item_similarity_pearson[book_name] * (user_rating-2.5)
            similar_score = similar_score.sort_values(ascending=False)
            return similar_score

        user_model = surprise.KNNBasic(k=40,sim_options={'name': 'pearson','user_based': True})
        user_model.fit(train1)
        preds = user_model.test(test1)
        accuracy.rmse(preds,verbose=True)

        similar_book = pd.DataFrame()
        for bk_name, rating in user1_list:
            similar_book = similar_book.append(get_similar_book_pearson_user(bk_name, rating), ignore_index=True)
        recommended_pearson = similar_book.sum().sort_values(ascending=False).index
        result_list = []
        for ind in recommended_pearson[:10]:
            result_list.append(ind)
        result_df = pd.DataFrame(data=result_list)
        st.table(result_df)
        st.success('I think these are the right book(s) for you! :smile:')
        #st.balloons()

#---------------------------------------------------------------------------------------------------------------------
                                                  # UI Codes - EDA
# This is for EDA section
if section1 == "Exploratory Data Analysis":
    st.markdown("# GET INSIGHTS FROM DATA")

    # ** check for null values **
    st.subheader("check for null values")
    books_merge_ratings=pd.merge(books, ratings)
    fig, ax = plt.subplots()
    plt.figure(figsize=(12,8))
    sns.heatmap(books_merge_ratings.isnull(), ax=ax, cbar = True)
    st.write(fig)

    # ** which rating is highest from 1- 5 **
    # distribution of average ratings of all the 10000 books
    st.subheader("Which Ratings are more on books")
    plt.title("Distribution of Average Ratings")
    histogram = books["average_rating"]
    st.line_chart(data=histogram)

    # ** which author has more books **
    st.subheader("Which Author has more Books")
    top_author_counts = books['authors'].value_counts().reset_index()
    top_author_counts.columns = ['value', 'count']
    top_author_counts['value'] = top_author_counts['value']
    top_author_counts = top_author_counts.sort_values('count')
    fig = px.bar(top_author_counts.tail(10), x="count", y="value", orientation='h', color='value',width=800, height=600)
    st.write(fig)

    # ** Which year has maximum number of books published **
    st.subheader("Which year has maximum number of books published")
    years = books['original_publication_year'].value_counts().reset_index()
    years.columns = ['year', 'count']
    years['year'] = years['year']
    years = years.sort_values('count')
    fig = px.bar(years.tail(50), x="count", y="year", orientation='h', color='count',width=800, height=600)
    st.write(fig)

    # ** Count of Book's Langauge **
    st.subheader("Count of Book's Langauges")
    lang = books['language_code'].value_counts().reset_index()
    lang.columns = ['value', 'count']
    lang['value'] = lang['value']
    lang = lang.sort_values('count')
    fig = px.bar(lang.tail(10), x="count", y="value", orientation='h', color='count',width=800, height=600)
    st.write(fig)

    # ** which books has highest no of average rating **
    st.subheader("Which books has the highest no of Average Ratings")
    selected_rows = st.selectbox("Select no. of rows:", options=rows_count, key=7)
    books_filter = pd.DataFrame(books, columns=['book_id', 'authors', 'original_title', 'average_rating'])
    books_filter = books_filter.sort_values('average_rating', ascending=False)
    st.write(books_filter.head(20))
    #books_filter_chart = books_filter.drop(columns={'book_id','average_rating'})
    #st.bar_chart(books_filter_chart.head(20))

    # ** WordCloud for Book Title **
    st.markdown("## **Word Cloud - Book Author**")
    authors_string = " ".join(books['authors'])
    my_expander = st.beta_expander("Customize here:")
    max_word1 = my_expander.slider("Set words", 200, 1000, 500, key=1)
    max_font1 = my_expander.slider("Set Font Size", 50, 350, 60, key=2)
    random1 = my_expander.slider("Set Random State", 30, 100, 42, key=3)
    st.write(cloud_author(authors_string, max_word1, max_font1, random1))

    # ** WordCloud for Book Title**
    st.markdown("## **Word Cloud - Book Title**")
    my_expander1 = st.beta_expander("Customize here:")
    original_title_string = " ".join(books['title'])
    max_word2 = my_expander1.slider("Set words", 200, 1000, 500)
    max_font2 = my_expander1.slider("Set Max Font Size", 50, 350, 60)
    random2 = my_expander1.slider("Set Random State", 30, 100, 42)
    st.write(cloud_title(original_title_string,max_word2,max_font2,random2))

    # ** Total number of users **
    st.header("Total Number of Users")
    total_users = ratings['user_id'].unique()[-1]
    st.markdown(total_users)

    # ** A specific user has what rating to which book**"""
    # input user_id and get the count of books he rated
    st.markdown("## **Count of books a user rated**")
    all_user_list = ratings['user_id'].head(100).tolist()
    selected_user_id = st.selectbox("Select a user: ",options=all_user_list)
    my_expander2 = st.beta_expander("Select counts:")
    i=20
    user_counts_list = []
    while(i>=1):
        user_counts_list.append(i)
        i=i-1
    user_counts = my_expander2.selectbox("drop down:", options=user_counts_list)
    rating_count = ratings[ratings['user_id'] == selected_user_id].head(user_counts)
    st.write("User",selected_user_id ,"has rated", rating_count)

    # ** check CORRELATION **
    st.subheader("check for CORRELATION:")
    books_merge_ratings = pd.merge(books, ratings)
    fig, ax = plt.subplots()
    plt.figure(figsize=(12, 8))
    sns.heatmap(books_merge_ratings.corr(), ax=ax, cbar=True)
    st.write(fig)

    #----------------------------------------------------------------------------------------------------------------
                                                # UI Codes - BOOKS DATASET
# This is for books data section
if section1 == "The Data":
    st.markdown("# ABOUT THE DATA")
    st.write("This data is extracted from GoodReads")
    image3 = Image.open('banner_pic.jpg')
    st.image(image3)

    st.header("Get Recommendation based on Book attributes like:")
    st.write("The Data set includes:")

    #BOOK SECTION
    st.markdown("## **BOOKS.csv**")
    selected_rows = st.selectbox("Select no. of rows:", options=rows_count, key=1)
    st.dataframe(books.head(selected_rows))
    st.subheader("The Rows and Columns it contain:")
    st.code(books.shape)
    st.subheader("How many empty cells does it contain:")
    st.text(books.isnull().sum())

    # RATINGS SECTION
    st.markdown("## **RATINGS.csv**")
    selected_rows = st.selectbox("Select no. of rows:", options=rows_count, key=2)
    st.dataframe(ratings.head(selected_rows))
    st.subheader("The Rows and Columns it contain:")
    st.code(ratings.shape)
    st.subheader("How many empty cells does it contain:")
    st.text(ratings.isnull().sum())

    # BOOK_TAGS SECTION
    st.markdown("## **BOOK_TAGS.csv**")
    selected_rows = st.selectbox("Select no. of rows:", options=rows_count, key=3)
    st.dataframe(book_tags.head(selected_rows))
    st.subheader("The Rows and Columns it contain:")
    st.code(book_tags.shape)
    st.subheader("How many empty cells does it contain:")
    st.text(book_tags.isnull().sum())

    # TAGS SECTION
    st.markdown("## **TAGS.csv**")
    selected_rows = st.selectbox("Select no. of rows:", options=rows_count, key=4)
    st.dataframe(tags.head(selected_rows))
    st.subheader("The Rows and Columns it contain:")
    st.code(tags.shape)
    st.subheader("How many empty cells does it contain:")
    st.text(tags.isnull().sum())

    # TO_READ SECTION
    st.markdown("## **TO_READ.csv**")
    selected_rows = st.selectbox("Select no. of rows:", options=rows_count, key=5)
    st.dataframe(to_read.head(selected_rows))
    st.subheader("The Rows and Columns it contain:")
    st.code(to_read.shape)
    st.subheader("How many empty cells does it contain:")
    st.text(to_read.isnull().sum())

    # TAGS_JOIN_DF SECTION
    st.markdown("## **TAGS with BOOKS**")
    selected_rows = st.selectbox("Select no. of rows:", options=rows_count, key=6)
    st.dataframe(tags_join_DF.tail(selected_rows))
    st.subheader("The Rows and Columns it contain:")
    st.code(tags_join_DF.shape)
    st.subheader("How many empty cells does it contain:")
    st.text(tags_join_DF.isnull().sum())
    #---------------------------------------------------------------------------------------------------------------------
#************************************************   FLIPKART DATASET   ***************************************************
if section1 == "Flipkart Dataset":
    st.markdown("# RECOMMENDATION SYSTEM - FLIPKART")
    st.write("The dataset is extracted from Flipkart - Books section where you can get the data of books, "
             "locally sold in India. The data is extracted using web scrapping through BeautifulSoup.")
    st.image("flipkart-img.jpg")

    f_books= pd.read_csv("DATA\\Flipkart_books110.csv")
    st.subheader("Provide us Book Details")
    f_book_name_list = f_books['title'].tolist()
    f_book_selected = st.selectbox("Select a Book:",options=f_book_name_list)

    i = 1
    f_list_of_recom = []
    while (i <= 10):
        f_list_of_recom.append(i)
        i = i + 1
    innercol1, innercol2, innercol3 = st.beta_columns(3)
    innercol1.write("You've selected:")
    innercol2.info(f_book_selected)
    no_of_recom = innercol3.selectbox('No. of Recommendations:', f_list_of_recom)

    st.subheader("Get Recommendation based on Book attributes:")
    st.write("By which way would you should be recommended:")
    f_col1, f_col2, f_col3, f_col4 = st.beta_columns(4)
    f_checkbox1 = f_col1.checkbox("BOOK AUTHOR")
    f_checkbox2 = f_col2.checkbox("BOOK GENRE")
    f_checkbox3 = f_col3.checkbox("BOTH")
    f_checkbox4 = f_col4.checkbox("ITEM BASED")
    f_btn = st.button("SUBMIT", key=5)

    if f_checkbox1 == True and f_btn == True:
        progress_bar = st.progress(0)
        for i in range(10):
            #time.sleep(0.1)
            progress_bar.progress(i + 2)

        st.info("These are the Recommendations based on Author's Preference")

        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
        tfidf_matrix = tf.fit_transform(f_books['author'])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        titles = f_books['title']
        indices = pd.Series(f_books.index, index=f_books['title'])

        def authors_recommendation(title):
            idx = indices[title]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:21]
            book_indices = [i[0] for i in sim_scores]
            return titles.iloc[book_indices]

        f_result_df = authors_recommendation(f_book_selected).head(no_of_recom)
        st.table(f_result_df)

    if f_checkbox2 == True and f_btn == True:
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.1)
            progress_bar.progress(i+1)

        st.info("These are the Recommendations based on Book Tags's Preference")

        tf1 = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
        tfidf_matrix1 = tf1.fit_transform(f_books['tags'].head(110))
        cosine_sim1 = linear_kernel(tfidf_matrix1, tfidf_matrix1)
        titles1 = f_books['title']
        indices1 = pd.Series(f_books.index, index=f_books['title'])

        def genre_recommendation(title):
            idx = indices1[title]
            sim_scores = list(enumerate(cosine_sim1[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:21]
            book_indices = [i[0] for i in sim_scores]
            return titles1.iloc[book_indices]

        f_result_df = genre_recommendation(f_book_selected).head(no_of_recom)
        st.table(f_result_df)
        st.success('I think these are the right book(s) for you! :smile:')

    if f_checkbox3 == True and f_btn == True:
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.1)
            progress_bar.progress(i + 1)

        st.info("These are the Recommendations based on BOTH Author and Tags")
        temp_df = books_with_tags.groupby('book_id')['tag_name'].apply(' '.join).reset_index()
        f_books['corpus'] = (pd.Series(
            f_books[['author', 'tags']]
                .fillna('')
                .values
                .tolist())
                           .str.join(' '))
        tf_corpus = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
        tfidf_matrix_corpus = tf_corpus.fit_transform(f_books['corpus'])
        cosine_sim_corpus = linear_kernel(tfidf_matrix_corpus, tfidf_matrix_corpus)
        titles = f_books['title']
        indices1 = pd.Series(f_books.index, index=f_books['title'])

        def both_recommendation(title):
            idx = indices1[title]
            sim_scores = list(enumerate(cosine_sim_corpus[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:21]
            book_indices = [i[0] for i in sim_scores]
            return titles.iloc[book_indices]

        f_result_df = both_recommendation(f_book_selected).head(no_of_recom)
        st.table(f_result_df)
        st.success('I think these are the right book(s) for you! :smile:')

