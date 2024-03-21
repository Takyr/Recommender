from pathlib import Path
import streamlit as st
from PIL import Image
from collections import defaultdict
import random
import numpy as np
import pandas as pd
import pickle
import sqlite3
from passlib.hash import pbkdf2_sha256

# Kết nối CSDL SQLite
conn = sqlite3.connect('user_database.db')
cursor = conn.cursor()

# Tạo bảng user nếu chưa tồn tại
cursor.execute('''
    CREATE TABLE IF NOT EXISTS user (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        password TEXT NOT NULL
    )
''')
conn.commit()

def create_user(username, password):
    hashed_password = pbkdf2_sha256.hash(password)
    cursor.execute('INSERT INTO user (username, password) VALUES (?, ?)', (username, hashed_password))
    conn.commit()

def verify_user(username, password):
    cursor.execute('SELECT password FROM user WHERE username=?', (username,))
    result = cursor.fetchone()
    if result:
        hashed_password = result[0]
        return pbkdf2_sha256.verify(password, hashed_password)
    return False

# Chức năng đăng kí
def register():
    st.subheader("Đăng Kí")
    new_username = st.text_input("Tên người dùng:")
    new_password = st.text_input("Mật khẩu:", type="password")
    confirm_password = st.text_input("Xác nhận mật khẩu:", type="password")

    if st.button("Đăng Kí"):
        if new_password == confirm_password:
            create_user(new_username, new_password)
            st.success("Đăng kí thành công! Đăng nhập ngay bây giờ.")
        else:
            st.error("Mật khẩu và xác nhận mật khẩu không khớp.")

# Chức năng đăng nhập
def login():
    st.subheader("Đăng Nhập")
    username = st.text_input("Tên người dùng:")
    password = st.text_input("Mật khẩu:", type="password")

    if st.button("Đăng Nhập"):
        if verify_user(username, password):
            st.success(f"Đăng nhập thành công, chào mừng {username}!")
        else:
            st.error("Đăng nhập không thành công. Vui lòng kiểm tra lại tên người dùng và mật khẩu.")

# Giao diện chính của ứng dụng
def main():
    st.title("CHÀO MỪNG BẠN ĐẾN VỚI FASHION STORE!")

    menu = ["Đăng Nhập", "Đăng Kí"]
    choice = st.sidebar.selectbox("Chọn chức năng:", menu)

    if choice == "Đăng Nhập":
        login()
    elif choice == "Đăng Kí":
        register()

if __name__ == '__main__':
    main()
