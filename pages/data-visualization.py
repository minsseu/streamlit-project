import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(page_title='데이터시각화')
st.header('데이터 시각화', divider = 'blue')

# 데이터로드
df = pd.read_csv('Obesity Classification.csv')

# 탭생성
t1, t2, t3, t4 = st.tabs(['BMI 분포 히스토그램', '성별 레이블 BMI 박스플롯', '레이블 카운트 막대그래프', '키 vs 몸무게 산점도'])

# BMI 분포 히스토그램
with t1:
    st.markdown('**BMI 분포 히스토리**')
    fig1 = go.Figure()
    for label in df['Label'].unique():
        fig1.add_trace(go.Histogram(
            x=df.loc[df['Label']==label, 'BMI'],
            name=label,
            opacity=0.7
        ))
        fig1.update_layout(
            barmode='overlay',
            xaxis_title = 'BMI',
            yaxis_title='Count'
        )
        st.plotly_chart(fig1, use_container_width=True, config={'responsive':True})

# 성별 레이블 BMI 박스플롯
with t2:
    st.markdown('**성별 레이블별 BMI 박스플롯**')
    fig2 = go.Figure()
    for label in df['Label'].unique():
        fig2.add_trace(go.Box(
            x=df.loc[df['Label']==label, 'Gender'],
            y=df.loc[df['Label']==label, 'BMI'],
            name=label,
            boxmean='sd'
        ))
    fig2.update_layout(
        xaxis_title = 'Gender',
        yaxis_title='BMI'
    )
    st.plotly_chart(fig2, use_container_width=True, config={'responsive':True})

# 레이블 카운트 막대그래프
with t3: 
    st.markdown('**체중 범주별 샘플수**')
    label_counts = df['Label'].value_counts().reset_index()
    label_counts.columns = ['Label', 'Count']
    fig3 = go.Figure(go.Bar(
        x=label_counts['Label'],
        y=label_counts['Count']
    ))
    fig3.update_layout(
        xaxis_title = 'Label',
        yaxis_title='Count'
    )
    st.plotly_chart(fig3, use_container_width=True, config={'responsive':True})

    # 키 vs 몸무게 산점도
    with t4:
        st.markdown('**키 vs 몸무게 산점도(체중 범주별)**')
        fig4 = go.Figure()
        for label in df['Label'].unique():
            subset = df[df['Label']==label]
            fig4.add_trace(go.Scatter(
                x=subset['Height'],
                y=subset['Weight'],
                mode='markers',
                name=label,
            ))
        fig4.update_layout(
            xaxis_title = 'Height (cm)',
            yaxis_title = 'Weight (kg)'
        )
        st.plotly_chart(fig4, use_container_width=True, config={'responsive':True})