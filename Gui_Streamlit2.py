#IMPORT LIBRARIES
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
import squarify
from io import BytesIO
from datetime import datetime
from underthesea import word_tokenize, pos_tag, sent_tokenize
import jieba
import re
import string
from wordcloud import WordCloud

STOP_WORD_FILE = 'stopwords-en.txt'
with open(STOP_WORD_FILE, 'r', encoding='utf-8') as file:
    stop_words = file.read()
stop_words = stop_words.split('\n')

# LOADING DATA
df_product = pd.read_csv('Products_with_Prices.csv')
df = pd.read_csv('Transactions.csv')
df = df.merge(df_product, how='left', on='productId')
df['Sales'] = df['items'] * df['price']
df['Transaction_id'] = df.index
string_to_date = lambda x : datetime.strptime(x, "%d-%m-%Y").date()

# Convert InvoiceDate from object to datetime format
df['Date'] = df['Date'].apply(string_to_date)
df['Date'] = df['Date'].astype('datetime64[ns]')

# Drop NA values
df = df.dropna()
# RFM
# Convert string to date, get max date of dataframe
max_date = df['Date'].max().date()

Recency = lambda x : (max_date - x.max().date()).days
Frequency  = lambda x: len(x.unique())
Monetary = lambda x : round(sum(x), 2)

df_RFM = df.groupby('Member_number').agg({'Date': Recency,
                                        'Transaction_id': Frequency,
                                        'Sales': Monetary })
# Rename the columns of DataFrame
df_RFM.columns = ['Recency', 'Frequency', 'Monetary']
# Descending Sorting
df_RFM = df_RFM.sort_values('Monetary', ascending=False)
# Create labels for Recency, Frequency, Monetary
r_labels = range(4, 0, -1) # số ngày tính từ lần cuối mua hàng lớn thì gán nhãn nhỏ, ngược lại thì nhãn lớn
f_labels = range(1, 5)
m_labels = range(1, 5)
# Assign these labels to 4 equal percentile groups
r_groups = pd.qcut(df_RFM['Recency'].rank(method='first'), q=4, labels=r_labels)
f_groups = pd.qcut(df_RFM['Frequency'].rank(method='first'), q=4, labels=f_labels)
m_groups = pd.qcut(df_RFM['Monetary'].rank(method='first'), q=4, labels=m_labels)
# Create new columns R, F, M
df_RFM = df_RFM.assign(R = r_groups.values, F = f_groups.values,  M = m_groups.values)
def join_rfm(x): return str(int(x['R'])) + str(int(x['F'])) + str(int(x['M']))
df_RFM['RFM_Segment'] = df_RFM.apply(join_rfm, axis=1)
def rfm_level(df):
    if df['RFM_Segment'] == '444':
        return 'VIP'
    elif df['R'] == 4 and df['F'] == 1 and df['M'] == 1:
        return 'NEW'
    elif df['F'] == 4 and df['R'] >= 3:
        return 'LOYAL'
    elif df['F'] == 1:
        return 'TRAVELLER'
    elif df['R'] == 1:
        return 'LOST'
    else:
        return 'REGULARS'
# Create a new column RFM_Level
df_RFM['RFM_Level'] = df_RFM.apply(rfm_level, axis=1)
# Calculate average values for each RFM_Level, and return a size of each segment
rfm_agg = df_RFM.groupby('RFM_Level').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count']}).round(0)

rfm_agg.columns = rfm_agg.columns.droplevel()
rfm_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
rfm_agg['Percent'] = round((rfm_agg['Count']/rfm_agg.Count.sum())*100, 2)

# Reset the index
rfm_agg = rfm_agg.reset_index()
# Reset the index
df_RFM = df_RFM.reset_index()
df_RFM.head()
df_cust = df.merge(df_RFM[['Member_number','RFM_Level']], on='Member_number', how='left')
df_cust.rename(columns={'RFM_Level':'CustGroup'}, inplace = True)

rfm = df_RFM[['Member_number','Recency','Frequency','Monetary']]

df['Year'] = df['Date'].dt.year
df_plot = df.groupby('Year').agg({'Member_number': lambda x: len(x.unique()),
                                             'Transaction_id': lambda x: len(x.unique())
                                              }).reset_index()
df_plot.columns = ['Năm', 'Số lượng KH', 'Số lượng đơn hàng']

df_plot.sort_values(by = 'Số lượng KH', ascending = False, inplace = True)
df_plot.reset_index(drop=True, inplace=True)

# USING MENU
st.title("Customer Segmentation")
menu = ["Trang chủ", "Tổng quan", "Cơ chế phân nhóm", "Phân tích khách hàng"]
choice = st.sidebar.selectbox('Trang chủ', menu)
if choice == "Trang chủ":
    st.image('UserSegmentation.png')
    st.write('#### Author 1: Triệu Thị Kim Trang')
    st.write('#### Author 2: Phan Thị Tuyết')
elif choice =='Tổng quan':
    st.image('Business Overview.jpg')
    st.write("""Cửa hàng X chủ yếu bán các sản phẩm thiết yếu cho khách hàng như rau, củ, quả, thịt, cá, trứng, sữa, nước giải khát... Khách hàng của cửa hàng là khách hàng mua lẻ.
                Ứng dụng xây dụng giúp cửa hàng X có thể bán được nhiều hàng hóa hơn cũng như giới thiệu sản phẩm đến đúng đối tượng khách hàng, chăm sóc và làm hài lòng khách hàng.""")
    st.subheader("TỔNG QUAN TÌNH HÌNH KINH DOANH")

    st.write("##### 1. Tổng quan đơn hàng:")
    
    # Đếm số lượng đơn hàng
    a = len(df['Transaction_id'].unique())
    b = len(df.loc[df['items'] >= 3]['Transaction_id'].unique())
    c = len(df.loc[df['Sales'] >= 50]['Transaction_id'].unique())
    st.dataframe(pd.DataFrame({'Đơn hàng': ['Tổng', 'Số lượng mua trên 3 cái/lần','Giá trị trên 50 USD/ lần'], 'Số lượng': [a,b,c]}))
    
    # Vẽ biểu đồ Số lượng đơn hàng theo năm
    plt.figure(figsize=(8, 6))
    bars2 = plt.bar(df_plot['Năm'], df_plot['Số lượng đơn hàng'], color='orange')
    plt.xlabel('Năm')
    plt.ylabel('Số lượng đơn hàng')
    plt.title('Thống kê số lượng đơn hàng theo năm')
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height,
                f'{height}', ha='center', va='bottom')
    st.pyplot(plt)
    
    st.write("##### 2. Tổng quan sản phẩm:")
    # Thống kê số lượng KH và đơn hàng theo sản phẩm
    df_g_product = df.groupby('productName').agg({'Member_number': lambda x: len(x.unique()), 'Transaction_id': lambda x: len(x.unique())}).reset_index()
    df_g_product.columns = ['Sản phẩm', 'Số lượng KH', 'Số lượng đơn hàng']
    df_g_product.sort_values(by = 'Số lượng KH', ascending = False, inplace = True)
    df_g_product.reset_index(drop=True, inplace=True)
    df_bar = df_g_product.head()
    ## Top 5 SP theo số lượng khách hàng
    colors = ['salmon', 'limegreen','gold', 'pink','skyblue']
    sorted_indices = sorted(range(len(df_bar['Số lượng KH'])), key=lambda i: df_bar['Số lượng KH'][i], reverse=False)
    sorted_countries = [df_bar['Sản phẩm'][i] for i in sorted_indices]
    sorted_num_countries = [df_bar['Số lượng KH'][i] for i in sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_countries, sorted_num_countries, color=colors)
    plt.xlabel('Số lượng KH')
    plt.ylabel('Sản phẩm')
    plt.title('Top 5 sản phẩm có nhiều khách hàng nhất')
    plt.tight_layout()
    st.pyplot(plt)
    ## Top 5 SP theo số lượng đơn hàng
    df_g_product1 = df.groupby('productName').agg({'Member_number': lambda x: len(x.unique()), 'Transaction_id': lambda x: len(x.unique())}).reset_index()
    df_g_product1.columns = ['Sản phẩm', 'Số lượng KH', 'Số lượng đơn hàng']
    df_g_product1.sort_values(by = 'Số lượng đơn hàng', ascending = False, inplace = True)
    df_g_product1.reset_index(drop=True, inplace=True)
    df_bar1 = df_g_product1.head()
    sorted_indices1 = sorted(range(len(df_bar1['Số lượng đơn hàng'])), key=lambda i: df_bar1['Số lượng đơn hàng'][i], reverse=False)
    sorted_countries1 = [df_bar1['Sản phẩm'][i] for i in sorted_indices1]
    sorted_num_countries1 = [df_bar1['Số lượng đơn hàng'][i] for i in sorted_indices1]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_countries1, sorted_num_countries1, color=colors)
    plt.xlabel('Số lượng đơn hàng')
    plt.ylabel('Sản phẩm')
    plt.title('Top 5 sản phẩm có nhiều đơn hàng nhất')
    plt.tight_layout()
    st.pyplot(plt)
      
    st.write("##### 3. Tổng quan lượng khách hàng:")
    # Vẽ biểu đồ số lượng KH theo năm
    plt.figure(figsize=(8, 6))
    bars = plt.bar(df_plot['Năm'], df_plot['Số lượng KH'], color='skyblue')
    plt.xlabel('Năm')
    plt.ylabel('Số lượng KH')
    plt.title('Thống kê số lượng khách hàng theo năm')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height,
                f'{height}', ha='center', va='bottom')
    st.pyplot(plt)
    
    st.write("##### 4. Tổng quan một số cụm khách hàng:")
    
    st.write('#### WordCloud top sản phẩm của PK KH VIP:')
    st.image('WC_TopProduct_VIP.png')
    st.write('#### WordCloud top sản phẩm của PK KH Trung thành:')
    st.image('WC_TopProduct_LOYAL.png')
    st.write('#### WordCloud top sản phẩm của PK KH Mới:')
    st.image('WC_TopProduct_NEW.png')
    st.write('#### WordCloud top sản phẩm của PK KH Vãng lai:')
    st.image('WC_TopProduct_TRAVELLER.png')
    st.write('#### WordCloud top sản phẩm của PK KH Rời bỏ:')
    st.image('WC_TopProduct_LOST.png')
    st.write('#### WordCloud top sản phẩm của PK KH Thông thường:')
    st.image('WC_TopProduct_REGULARS.png')


elif choice == "Cơ chế phân nhóm":
    st.image('customer-segmentation-social.png')
    st.write('### Manual Segmentation')
    st.write("""Customer Segmentation là một công cụ mạnh mẽ giúp doanh nghiệp hiểu sâu hơn về khách hàng của họ và cách tùy chỉnh chiến lược tiếp thị.
                            Đây là một bước không thể thiếu để đảm bảo rằng bạn đang tiếp cận và phục vụ mọi nhóm khách hàng một cách hiệu quả""")
    st.write("""**Tiêu chí phân loại Khách hàng:**""")
    st.write("""
            + VIP: Khách hàng có lượng chi tiêu lớn, tần suất tiêu thụ thường xuyên, và vừa mua hàng gần đây
            + NEW: Khách hàng mới đến gần đây, chưa quan tâm đến mức độ chi tiêu
            + LOYAL: Khách hàng thường đến và vẫn còn đến gần đây
            + TRAVELLER: Khách hàng có tuần suất đến thấp nhất
            + LOST: Khách hàng quá lâu chưa đến
            + REGULARS: Nhóm còn lại, thường ở mức trung bình ở 3 khía cạnh M, F, R
            """)
    st.write('### Giá trị trung bình Recency-Frequency-Monetary theo các phân cụm')
    st.dataframe(rfm_agg)
    # Show biểu đồ phân cụm
    st.write('### TreeMap')
    st.image('RFM_Plot.png')
    st.write('### Scatter Plot (RFM)')
    st.image('Scatter Plot.png')

elif choice=='Phân tích khách hàng':
    st.image('segmentation.webp')
    st.subheader("PHÂN TÍCH KHÁCH HÀNG")
    type = st.radio("### Nhập thông tin khách hàng", options=["Xem KH hiện hữu", "Hành vi mua hàng"])
    if type == "Xem KH hiện hữu":
        st.subheader("Mã khách hàng")
        # Tạo điều khiển để người dùng nhập và chọn nhiều mã khách hàng từ danh sách gợi ý
        st.markdown("**Có thể nhập và chọn nhiều mã khách hàng từ danh sách gợi ý**")

        all_ids = df['Member_number'].unique()
        # Chọn nhiều ID từ danh sách
        selected_ids = st.multiselect("Chọn Member_number:", all_ids)
        # In ra danh sách ID đã chọn
        st.write("#### Bạn đã chọn các KH sau:")
        st.write(selected_ids)

        if any(id in df['Member_number'].values for id in selected_ids):

            # Đề xuất khách hàng thuộc cụm nào
            df_cust_rfm = df_RFM[df_RFM['Member_number'].isin(selected_ids)].sort_values(['Member_number'], ascending= False, ignore_index= True)
            st.write(f"#### Khách hàng đã chọn thuộc nhóm")
            st.dataframe(df_cust_rfm[['Member_number', 'RFM_Level', 'RFM_Segment', 'Recency', 'Frequency', 'Monetary']])
            filtered_df_new = df[df['Member_number'].isin(selected_ids)].sort_values(['Member_number', 'Date'], ascending= False, ignore_index= True)
            
            st.write("#### Khoảng chi tiêu ($):")
            grosssale = filtered_df_new.groupby('Member_number').agg({'Sales': ['min', 'max', 'sum']}).reset_index()
            grosssale.columns = ['Member_number', 'Min', 'Max', 'Total']
            st.dataframe(grosssale)
            
            st.write("#### Thông tin mua hàng sắp xếp theo lần gần nhất:")
            st.dataframe(filtered_df_new)

            def cust_top_product(selected_ids):
                df_choose = df[df['Member_number'].isin(selected_ids)]
                df_top = df_choose.groupby(['productName','price']).value_counts().sort_values(ascending=False).head(20).reset_index()
                return df_top
            def text_underthesea(text):
                products_wt = text.str.lower().apply(lambda x: word_tokenize(x, format="text"))
                products_name_pre = [[text for text in set(x.split())] for x in products_wt]
                products_name_pre = [[re.sub('[0-9]+','', e) for e in text] for text in products_name_pre]
                products_name_pre = [[t.lower() for t in text if not t in ['', ' ', ',', '.', '...', '-',':', ';', '?', '%', '_%' , '(', ')', '+', '/', 'g', 'ml']]
                                    for text in products_name_pre] # ký tự đặc biệt
                products_name_pre = [[t for t in text if not t in stop_words] for text in products_name_pre] # stopword
                return products_name_pre
            def wcloud_visualize(input_text):
                flat_text = [word for sublist in input_text for word in sublist]
                text = ' '.join(flat_text)
                wc = WordCloud(
                                background_color='white',
                                colormap="ocean_r",
                                max_words=50,
                                width=1600,
                                height=900,
                                max_font_size=400)
                wc.generate(text)
                plt.figure(figsize=(8,12))
                plt.imshow(wc, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)

            top_products = cust_top_product(selected_ids)
            st.write("#### Top 5 đơn hàng của được mua nhiều nhất của KH được chọn:")
            st.dataframe(top_products.head())
            st.write("#### Word Cloud Visualization:")
            sample_text = top_products['productName']
            processed_text = text_underthesea(sample_text)
            wcloud_visualize(processed_text)

        else:
            # Không có khách hàng
            st.write("Vui lòng chọn ID ở khung trên :rocket:")


    elif type == "Hành vi mua hàng":
        # Nếu người dùng chọn nhập thông tin khách hàng vào dataframe có 3 cột là Recency, Frequency, Monetary
        st.write("##### 2. Thông tin khách hàng")
        # Tạo điều khiển table để người dùng nhập thông tin khách hàng trực tiếp trên table
        st.write("Nhập thông tin khách hàng (Tối đa 3 KH)")
        # Loop to get input from the user for each customer
            # Get input using sliders
        # Tạo DataFrame rỗng
        df_customer = pd.DataFrame(columns=["Recency", "Frequency", "Monetary"])

        # Lặp qua 3 khách hàng
        for i in range(3):
            st.write(f"##### Khách hàng {i+1}")
            
            # Sử dụng sliders để nhập giá trị cho Recency, Frequency và Monetary
            recency = st.slider("Recency", 1, 365, 100, key=f"recency_{i}")
            frequency = st.slider("Frequency", 1, 50, 5, key=f"frequency_{i}")
            monetary = st.slider("Monetary", 1, 1000, 100, key=f"monetary_{i}")
            
            # Thêm dữ liệu nhập vào DataFrame
            df_customer = df_customer.append({"Recency": recency, "Frequency": frequency, "Monetary": monetary}, ignore_index=True)
            
        # Hiển thị DataFrame
        st.dataframe(df_customer)

        # frames = [rfm,df_customer]
        # df_customer_full = pd.concat(frames)

        # Create labels for Recency, Frequency, Monetary
        r_labels = range(4, 0, -1) #số ngày tính từ lần cuối mua hàng lớn thì gán nhãn nhỏ, ngược lại thì nhãn lớn
        f_labels = range(1, 5)
        m_labels = range(1, 5)

        # Assign these labels to 4 equal percentile groups
        r_groups = pd.qcut(df_customer['Recency'].rank(method = 'first'), q = 4, labels = r_labels)
        f_groups = pd.qcut(df_customer['Frequency'].rank(method = 'first'), q = 4, labels = f_labels)
        m_groups = pd.qcut(df_customer['Monetary'].rank(method = 'first'), q = 4, labels = m_labels)

        # Create new columns R, F, M
        df_customer = df_customer[['Recency','Frequency','Monetary']].assign(R = r_groups.values, F = f_groups.values, M = m_groups.values)
        def join_rfm(x): return str(int(x['R'])) + str(int(x['F'])) + str(int(x['M']))
        df_customer['RFM_Segment'] = df_customer.apply(join_rfm, axis=1)
        # Calculate RFM_Score
        df_customer['RFM_Score'] = df_customer[['R', 'F', 'M']].sum(axis=1)

        def rfm_level(df):
            if df['RFM_Score'] == 12:
                return "VIP"
            elif df['R'] == 4 and df['F'] == 1 and df['M'] == 1:
                return "NEW"
            elif df['F'] == 4 and df['R'] >= 3:
                return "LOYAL"
            elif df['F'] == 1: 
                return "TRAVELLER"
            elif df['R'] == 1:
                return "LOST"
            else:
                return "REGULARS"

        # Create a new column RFM_Level
        df_customer['RFM_Level'] = df_customer.apply(rfm_level, axis=1)
        st.dataframe(df_customer)


