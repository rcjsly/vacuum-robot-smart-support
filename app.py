import uuid

import streamlit as st
from agent.react_agent import ReactAgent

# 标题
st.title("智扫通机器人智能客服")
st.divider()
st.caption("使用报告为2025年模拟数据；可指定2025年的月份体验，未接入真实用户账户。")

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

# ===== 记忆功能：为当前浏览器会话生成唯一会话ID =====
# st.session_state 在页面刷新后保留、浏览器关闭后重建，
# 因此"同一浏览器持续使用 = 同一会话共享记忆"
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
# ==================================================

if "message" not in st.session_state:
    st.session_state["message"] = []

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 用户输入提示词
prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    with st.spinner("智能客服思考中..."):
        # 记忆功能：传入当前浏览器会话的 session_id，Agent 按会话读写对话历史
        res_stream = st.session_state["agent"].execute_stream(prompt, st.session_state["session_id"])
        answer = st.chat_message("assistant").write_stream(res_stream)
        st.session_state["message"].append({"role": "assistant", "content": answer})
