import streamlit as st
import tensorflow as tf
from PIL import Image

# 导入模型，注意修改模型的路径
model = tf.keras.models.load_model('Desktop/oooo/0010.h5')

labels = ['no smoking', 'smoking']

# 定义预测函数
def predict(image):
    # 将图片调整为指定大小
    image = image.resize((224, 224))
    # 将图片转换为numpy数组
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    # 扩展维度以符合模型输入要求
    input_arr = tf.expand_dims(input_arr, 0)
    # 预测
    pred = model.predict(input_arr)[0]
    # 获取预测结果
    result = labels[pred.argmax()]
    # 获取置信度
    confidence = str(round(100 * pred.max(), 2)) + '%'
    # 返回结果和置信度
    return result, confidence

# 设置页面标题
st.title("吸烟检测")

# 添加文件上传组件
uploaded_file = st.file_uploader('请选择一张图片', type=['jpg', 'jpeg', 'png'])

# 如果用户上传了图片
if uploaded_file is not None:
    # 加载图片
    image = Image.open(uploaded_file)
    # 显示图片
    st.image(image, caption='上传的图片', use_column_width=True)
    # 预测并显示结果
    result, confidence = predict(image)
    st.write('预测结果：', result)
    st.write('准确率：', confidence)
    
    
