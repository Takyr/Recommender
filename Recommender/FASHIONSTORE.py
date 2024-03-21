from pathlib import Path
import streamlit as st  
from streamlit_option_menu import option_menu
from PIL import Image  
from collections import defaultdict
import random
import numpy as np
import pandas as pd
import pickle
import sqlite3                              
from passlib.hash import pbkdf2_sha256      

# --- PATH SETTINGS ---
THIS_DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()
ASSETS_DIR = THIS_DIR / "assets"
STYLES_DIR = THIS_DIR / "styles"
CSS_FILE = STYLES_DIR / "main.css"


# --- GENERAL SETTINGS ---
STRIPE_CHECKOUT = "https://buy.stripe.com/test_fZe2963rr1u22J2001"
CONTACT_EMAIL = "tungnguyen.260302@gmail.com"
DEMO_VIDEO = "https://www.youtube.com/watch?v=RDeIM_yOzZU"
PRODUCT_NAME = "FASHION STORE"
formatted_product_name = f'<span style="font-size: 55px; font-weight: bold; color: #FF5733;">{PRODUCT_NAME}</span>'
st.markdown(formatted_product_name, unsafe_allow_html=True)
st.sidebar.title("FASHION STORE")
PRODUCT_DESCRIPTION = """
Fashion Store - Thương hiệu thời trang thiết kế dành cho phái đẹp. Cửa hàng mang đến tinh thần thiết kế:

Phóng khoáng, tự do, lãng mạng và sang trọng.

Khát khao mang đến vẻ đẹp cá tính nhưng vẫn đủ dịu dàng, những thiết kế của shop tập trung hướng đến nhóm khách hàng từ 25 tuổi - 34 tuổi, có cá tính riêng và khát khao sự nổi bật. Chọn cho mình Slogan: Breath Of Beauty” Với mong muốn sẽ là nguồn cảm hứng về cái đẹp bất tận cho những cô gái khi đến với cửa hàng.

Thank you & Love You!

"""

# Create cart variable to store shopping cart
cart = {}

test_data = pickle.load(open('test_data.pkl', 'rb'))
test_data_ = pd.DataFrame(test_data)

# Load train data for seeing recommendations
train_data = pickle.load(open('img_data.pkl', 'rb'))
train_data_ = pd.DataFrame(train_data)

# Load model
knn = pickle.load(open('model_recommend.pkl', 'rb'))

# TF-IDF for text part
X_test = pickle.load(open('test_array.pkl', 'rb'))


# --- MAIN SECTION ---
selected = option_menu(None, ["Home", "Sắp Ra Mắt", "Giỏ Hàng", "Thanh Toán", "Gợi Ý Sản Phẩm", "Đánh Giá"], 
    menu_icon="cast", default_index=0, orientation="horizontal")


def load_css_file(css_file_path):
    with open(css_file_path) as f:
        return st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css_file(CSS_FILE)


if selected == "Home":
    left_col, right_col = st.columns((2, 1))
    with left_col:
        st.text("")
        st.write(PRODUCT_DESCRIPTION)
        st.markdown(
            f'<a href={STRIPE_CHECKOUT} class="button">👉 Mua Ngay</a>',
            unsafe_allow_html=True,
        )
    with right_col:
        product_image = Image.open(ASSETS_DIR / "crop.jpg")
        st.image(product_image, width=500)

elif selected == "Sắp Ra Mắt":
    # --- COLLECTION ---
    st.sidebar.markdown(
        f'<a href="#sap-ra-mat" style="text-decoration: none; color: inherit;"><h4>💡 Sắp Ra Mắt</h4></a>', 
        unsafe_allow_html=True
    )    
    st.subheader("💡 Sắp Ra Mắt")
    collection = {
        "glasses.jpg": [
            "Noir Glasses",
            50.0,  
            "Noir được yêu thích bởi giá thành hợp lý, nhất là các bạn sinh viên. Có thể thay tròng kính cận, khớp nối ốc chắc chắn, đệm mút êm ái ngay sống mũi, thiết kế tối giản phù hợp với đa số dáng mặt. Sản phẩm đi kèm với hộp kính và khăn lau riêng.",
        ],
        "downtown_small.jpg": [
            "Downtown Shopper - Small",
            75.0,  
            "Túi trong làm bằng chất liệu giả da, style minimal với zip khoá kéo. Túi ngoài có hoạ tiết lưới và quai cầm. Quai ngọc trai và dây xích trang trí có thể tách rời, tạo nên nhiều sự lựa chon để đi với nhiều outfit khác nhau.",
        ],
        "daphne_shoes.jpg": [
            "Daphne Heels",
            120.0,  
            "Daphne Heels được thiết kế trên form sandal cao gót mũi tròn, 8,5cm. Quai mũi trước thanh mảnh với chi tiết dây đá lấp lánh, chất liệu Satin đen sang trọng.",
        ],
    }

    # Hiển thị sản phẩm trong bộ sưu tập

    for image, description in collection.items():
        product_name, price, product_description = description
        image = Image.open(ASSETS_DIR / image)

        st.write("")
        left_col, right_col = st.columns(2)
        left_col.image(image, use_column_width=True)
        right_col.write(f"**{product_name}**")
        right_col.write(f"Price: ${price}")
        right_col.write(product_description)
    # --- DEMO ---
    st.sidebar.markdown(
        f'<a href="#ve-chung-toi" style="text-decoration: none; color: inherit;"><h4>🎥 Về Chúng Tôi</h4></a>', 
        unsafe_allow_html=True
    )
    st.write("<a name='ve-chung-toi'></a>", unsafe_allow_html=True)
    st.write("---")
    st.subheader("🎥 Về Chúng Tôi")
    st.video(DEMO_VIDEO, format="video/mp4", start_time=0)

# # --- ADD TO CART ---
# if selected == "Thêm Vào Giỏ Hàng":

#     st.sidebar.markdown(
#         f'<a href="#them-vao-gio-hang" style="text-decoration: none; color: inherit;"><h4>🛒 Thêm Vào Giỏ Hàng</h4></a>', 
#         unsafe_allow_html=True
#     )
#     st.write("<a name='them-vao-gio-hang'></a>", unsafe_allow_html=True)
#     st.write("---")
#     st.subheader("🛒 Thêm Vào Giỏ Hàng")
#     selected_products = st.multiselect("Chọn sản phẩm để thêm vào giỏ hàng:", list(test_data_['title']))

#     if selected_products:
#         for selected_product in selected_products:
#             quantity = st.number_input(f"Nhập số lượng {selected_product}:", min_value=1, value=1)

#             if st.button(f"Thêm {quantity} {selected_product} vào giỏ hàng"):
#                 if selected_product in test_data_['title'].values:
#                     cart[selected_product] = cart.get(selected_product, 0) + quantity
#                     st.success(f"Đã thêm {quantity} {selected_product} vào giỏ hàng.")
#                 else:
#                     st.warning(f"{selected_product} sản phẩm không tồn tại.")


# --- VIEW CART ---
if selected == "Giỏ Hàng":
    st.write("<a name='gio-hang'></a>", unsafe_allow_html=True)
    st.subheader("🛒 Giỏ Hàng")
    st.sidebar.markdown(
        f'<a href="#gio-hang" style="text-decoration: none; color: inherit;"><h4>🛒 Giỏ Hàng</h4></a>', 
        unsafe_allow_html=True
    )
    if not cart:
            st.write("Giỏ hàng trống.")
    else:
            total_price = 0
            for product, quantity in cart.items():
                product_info = test_data_[test_data_['title'] == product].iloc[0]
                price = product_info['formatted_price']
                total_price += price * quantity
                st.write(f"{product} x {quantity} - ${price * quantity}")

            st.write(f"**Tổng giá:** ${total_price}")

#--- RECOMMENDATION ---
if selected == "Gợi Ý Sản Phẩm":
    st.sidebar.markdown(
        f'<a href="#goi-y-san-pham" style="text-decoration: none; color: inherit;"><h4>🔎 Gợi Ý Sản Phẩm</h4></a>', 
        unsafe_allow_html=True
    )
    st.write("<a name='goi-y-san-pham'></a>", unsafe_allow_html=True)

    # load tập dữ liệu test để tìm hình ảnh hiện tại cho sản phẩm
    test_data = pickle.load(open('test_data.pkl','rb'))

    # load tập dữ liệu train để tìm các sản phẩm được recommend
    train_data = pickle.load(open('img_data.pkl','rb'))
    train_data_ = pd.DataFrame(train_data)

    # load model
    knn = pickle.load(open('model_recommend.pkl','rb'))

    # tfidf cho phần text
    X_test = pickle.load(open('test_array.pkl','rb'))

    st.title("DANH SÁCH SẢN PHẨM")
    # Connect to the Google Sheet
    df = pd.DataFrame(test_data)

    import streamlit.components.v1 as components
    from pandas.api.types import (
        is_categorical_dtype,
        is_datetime64_any_dtype,
        is_numeric_dtype,
        is_object_dtype,
    )

    def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:

        st.data_editor(
        df,
        column_order=('titleref', 'product_type_name','medium_image_url','coloref','brandref','formatted_price'),
        column_config={
            
        
            "medium_image_url": st.column_config.ImageColumn(
                "Preview Image", help="Streamlit app preview screenshots"
            )
        },
        hide_index=True,
    )

        modify = st.checkbox("Bộ lọc")
        if not modify:
            return df[['titleref', 'product_type_name','coloref','brandref','formatted_price']]
    
        df = df.copy()
        
        # Try to convert datetimes into a standard format (datetime, no timezone)
        for col in df.columns:
            if is_object_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass
            if is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.tz_localize(None)

        modification_container = st.container()

        with modification_container:
            to_filter_columns = st.multiselect("Nhập tên thương hiệu, màu sắc, giá cả,... cần tìm", df.columns)
            for column in to_filter_columns:
                left, right = st.columns((1, 20))
                # Treat columns with < 10 unique values as categorical
                if is_categorical_dtype(df[column]) or df[column].nunique() < 50:
                    user_cat_input = right.multiselect(
                        f"Values for {column}",
                        df[column].unique(),
                        default=list(df[column].unique()),
                    )
                    df = df[df[column].isin(user_cat_input)]
                elif is_numeric_dtype(df[column]):
                    _min = float(df[column].min())
                    _max = float(df[column].max())
                    step = (_max - _min) / 100
                    user_num_input = right.slider(
                        f"Values for {column}",
                        min_value=_min,
                        max_value=_max,
                        value=(_min, _max),
                        step=step,
                    )
                    df = df[df[column].between(*user_num_input)]
                elif is_datetime64_any_dtype(df[column]):
                    user_date_input = right.date_input(
                        f"Values for {column}",
                        value=(
                            df[column].min(),
                            df[column].max(),
                        ),
                    )
                    if len(user_date_input) == 2:
                        user_date_input = tuple(map(pd.to_datetime, user_date_input))
                        start_date, end_date = user_date_input
                        df = df.loc[df[column].between(start_date, end_date)]
                else:
                    user_text_input = right.text_input(
                        f"{column}",
                    )
                    if user_text_input:
                        df = df[df[column].astype(str).str.contains(user_text_input)]
        return df[['titleref', 'product_type_name','coloref','brandref','formatted_price']]
        

    test_data_ = pd.DataFrame(test_data)
    test_data_2 = pd.DataFrame(test_data)
    filterdf = pd.DataFrame(filter_dataframe(df))

    # Use a text_input to get the keywords to filter the dataframe
    text_search = st.text_input("Bạn tìm gì...", value="")

    # Filter the dataframe using masks
    m1 = df["product_type_name"].str.contains(text_search)
    m2 = df["coloref"].str.contains(text_search)
    m3 = df["brand"].str.contains(text_search)
    m4 = df["titleref"].str.contains(text_search)
    df_search = df[m1 | m2 | m3 | m4]

    # Another way to show the filtered results
    # Show the cards
    N_cards_per_row = 5
    if text_search:
        for n_row, row in df_search.reset_index().iterrows():
            i = n_row%N_cards_per_row
            if i==0:
                st.write("---")
                cols = st.columns(N_cards_per_row, gap="large")
            # draw the card
            with cols[n_row%N_cards_per_row]:
                st.caption(f"{row['brandref'].strip()} - {row['coloref'].strip()} ")
                st.markdown(f"**{row['titleref'].strip()}**")
                st.text(row['formatted_price'])
                st.image(f"{row['medium_image_url']}")
            

    title_current = st.selectbox('Tìm sản phẩm:',
                        list(filterdf['titleref']))

    product = test_data_[(test_data_['titleref'] == title_current)]

    s1 = product.index[0]
    captions = [test_data_['brand'].values[s1],test_data_['formatted_price'].values[s1]]
    c1,c2,c3 = st.columns(3)
    with c1:
        st.image(test_data_['medium_image_url'].values[s1])
    with c2:
        st.text('Nhãn hiệu: ')
        st.text('Màu sắc: ')
        st.text('Giá tiền: ')
    with c3:
        st.text(test_data_['brandref'].values[s1])
        st.text(test_data_['coloref'].values[s1])
        st.text(test_data_['formatted_price'].values[s1])

    distances, indices = knn.kneighbors([X_test.toarray()[s1]])
    result1 = list(indices.flatten())[:5]
    result2 = list(indices.flatten())[5:]

    if st.button('Gợi ý sản phẩm cho bạn'):
        st.success('Danh sách sản phẩm:')
        col1,col2,col3,col4,col5 = st.columns(5)
        lst1 = [col1,col2,col3,col4,col5]
        for i,j in zip(lst1,result1):
            with i:
                st.text(train_data_['titleref'].values[j])
                st.text(train_data_['brandref'].values[j])
                st.text(train_data_['coloref'].values[j])
                st.text(train_data_['formatted_price'].values[s1])
                st.image(train_data_['medium_image_url'].values[j])

        col6, col7, col8, col9, col10 = st.columns(5)
        lst2 = [col6, col7, col8, col9, col10]
        for k,l in zip(lst2,result2):
            with k:
                st.text(train_data_['titleref'].values[j])
                st.text(train_data_['brandref'].values[l])
                st.text(train_data_['coloref'].values[l])
                st.text(train_data_['formatted_price'].values[s1])
                st.image(train_data_['medium_image_url'].values[l])

        st.success('Cảm ơn đã mua hàng')

# #--- THANH TOÁN ---
# if selected == "Thanh Toán":
#     st.sidebar.markdown(
#     f'<a href="#thanh-toan" style="text-decoration: none; color: inherit;"><h4>💳 Thanh Toán</h4></a>', 
#     unsafe_allow_html=True
# )
# st.write("<a name='thanh-toan'></a>", unsafe_allow_html=True)
 
# # st.subheader("💳 Thanh Toán")

# if not cart:
#         st.write("Giỏ hàng trống.")
# else:
#         total_price = 0
#         for product, info_list in cart.items():
#             for info in info_list:
#                 st.write(f"{product} x {info['quantity']} - ${info['price'] * info['quantity']}")
#                 total_price += info['price'] * info['quantity']

#         st.write(f"**Tổng giá:** ${total_price}")

#         # Tạo liên kết đến trang thanh toán hoặc sử dụng cổng thanh toán bên ngoài 
#         checkout_url = "https://buy.stripe.com/test_fZe2963rr1u22J2001"  
#         st.markdown(
#             f'<a href={STRIPE_CHECKOUT} class="button">👉 Thanh Toán Ngay</a>',
#             unsafe_allow_html=True,
#         )

# #--- FAQ ---
# if selected == "FAQ":
#     st.sidebar.markdown(
#         f'<a href="#faq" style="text-decoration: none; color: inherit;"><h4>:raising_hand: FAQ</h4></a>', 
#         unsafe_allow_html=True
#     )
#     st.write("<a name='faq'></a>", unsafe_allow_html=True)
#     st.write("---")
#     st.subheader(":raising_hand: FAQ")
#     faq = {
#         "Question 1": "Some text goes here to answer question 1",
#         "Question 2": "Some text goes here to answer question 2",
#         "Question 3": "Some text goes here to answer question 3",
#         "Question 4": "Some text goes here to answer question 4",
#         "Question 5": "Some text goes here to answer question 5",
#     }
#     for question, answer in faq.items():
#         with st.expander(question):
#             st.write(answer)


    # --- CONTACT FORM ---
if selected == "Đánh Giá":
    # --- FAQ ---
    st.sidebar.markdown(
        f'<a href="#faq" style="text-decoration: none; color: inherit;"><h4>:raising_hand: FAQ</h4></a>', 
        unsafe_allow_html=True
    )
    st.write("<a name='faq'></a>", unsafe_allow_html=True)
    st.subheader(":raising_hand: FAQ")
    faq = {
        "Question 1": "Some text goes here to answer question 1",
        "Question 2": "Some text goes here to answer question 2",
        "Question 3": "Some text goes here to answer question 3",
        "Question 4": "Some text goes here to answer question 4",
        "Question 5": "Some text goes here to answer question 5",
    }
    for question, answer in faq.items():
        with st.expander(question):
            st.write(answer)
    #--- Đánh Giá ---
    st.sidebar.markdown(
        f'<a href="#lien-he" style="text-decoration: none; color: inherit;"><h4>:mailbox: Đánh Giá</h4></a>', 
        unsafe_allow_html=True
    )
    st.write("<a name='lien-he'></a>", unsafe_allow_html=True)
    st.write("---")
    st.subheader(":mailbox: Hãy Cho Chúng Tôi Biết Ý Kiến Của Bạn!")
    contact_form = f"""
    <form action="https://formsubmit.co/{CONTACT_EMAIL}" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Họ Tên" required>
        <input type="email" name="email" placeholder="Email" required>
        <textarea name="message" placeholder="Hãy để lại lời nhắn cho chúng tôi"></textarea>
        <button type="submit" class="button">Gửi ✉</button>
    </form>
    """
    st.markdown(contact_form, unsafe_allow_html=True)





