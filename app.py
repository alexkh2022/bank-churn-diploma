import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from sklearn.base import BaseEstimator, TransformerMixin
import openai

# --- 0. НАЛАШТУВАННЯ КЛАСУ (Обов'язково для joblib) ---


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['Is_Zero_Balance'] = (df['Balance'] == 0).astype(int)
        df['Is_Risk_Active_Senior'] = ((df['Age'] >= 45) & (
            df['Age'] <= 60) & (df['Balance'] > 0)).astype(int)

        def product_segment(n):
            if n == 1:
                return 1
            elif n == 2:
                return 0
            else:
                return 2
        df['Product_Segment'] = df['NumOfProducts'].apply(product_segment)
        df['Balance_Per_Product'] = df['Balance'] / df['NumOfProducts']
        df['Is_Perfect_Score'] = (df['CreditScore'] == 850).astype(int)
        cols_to_drop = ['RowNumber', 'CustomerId', 'Surname', 'Geography']
        df = df.drop(
            columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
        return df


# --- 1. КОНФІГУРАЦІЯ СТОРІНКИ ---
st.set_page_config(page_title="Bank Retention System", layout="wide")

# Стилізація під "Bootstrap" (чистий вигляд)
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    h1 {
        color: #0d6efd;
    }
    .stButton>button {
        background-color: #0d6efd;
        color: white;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Система Прогнозування Відтоку Клієнтів")
st.markdown("---")

# --- 2. ЗАВАНТАЖЕННЯ МОДЕЛІ ---


@st.cache_resource
def load_model():
    try:
        return joblib.load('churn_model_final.joblib')
    except FileNotFoundError:
        return None


bundle = load_model()

if not bundle:
    st.error("Помилка: Файл моделі 'churn_model_final.joblib' не знайдено.")
    st.stop()

engineer = bundle['engineer']
model = bundle['model']
optimal_threshold = bundle['threshold']

# --- 3. БІЧНА ПАНЕЛЬ (ВВІД) ---
st.sidebar.header("Параметри Клієнта")

# Секція налаштувань моделі
with st.sidebar.expander("Налаштування Моделі", expanded=False):
    current_threshold = st.slider(
        "Поріг чутливості (Threshold)",
        min_value=0.0, max_value=1.0,
        value=float(optimal_threshold),
        step=0.01,
        help="Зменшення порогу збільшує кількість виявлених ризиків, але може дати більше хибних спрацювань."
    )

# API Key
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# Ввід даних
credit_score = st.sidebar.slider("Кредитний Рейтинг", 350, 850, 650)
age = st.sidebar.number_input("Вік", 18, 92, 45)
tenure = st.sidebar.slider("Термін співпраці (років)", 0, 10, 5)
balance = st.sidebar.number_input("Баланс рахунку ($)", 0.0, 250000.0, 50000.0)
products = st.sidebar.selectbox("Кількість продуктів", [1, 2, 3, 4], index=0)
salary = st.sidebar.number_input(
    "Орієнтовна зарплата ($)", 0.0, 200000.0, 60000.0)
gender = st.sidebar.selectbox("Стать", ["Male", "Female"])
has_cr_card = st.sidebar.checkbox("Наявність кредитної картки", value=True)
is_active = st.sidebar.checkbox("Активний учасник", value=True)

# Формування DataFrame
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': ['France'],  # Заглушка
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [products],
    'HasCrCard': [int(has_cr_card)],
    'IsActiveMember': [int(is_active)],
    'EstimatedSalary': [salary],
    'RowNumber': [0], 'CustomerId': [0], 'Surname': ['X']
})

# --- 4. ПРОГНОЗУВАННЯ ---
processed_data = engineer.transform(input_data)
prob = model.predict_proba(processed_data)[0][1]

# Логіка статусів
if prob < 0.20:
    status = "Низький ризик"
    color = "#198754"  # Green
elif prob < current_threshold:
    status = "Моніторинг"
    color = "#ffc107"  # Yellow
elif prob < 0.85:
    status = "Високий ризик"
    color = "#fd7e14"  # Orange
else:
    status = "Критичний ризик"
    color = "#dc3545"  # Red

# --- 5. ГОЛОВНА ПАНЕЛЬ ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Результат Аналізу")

    # Метрики
    c1, c2 = st.columns(2)
    c1.metric("Ймовірність відтоку", f"{prob*100:.1f}%")
    c2.metric("Статус", status)

    # Спідометр (Gauge)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "Score"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 20], 'color': "#198754"},
                {'range': [20, current_threshold*100], 'color': "#ffc107"},
                {'range': [current_threshold*100, 85], 'color': "#fd7e14"},
                {'range': [85, 100], 'color': "#dc3545"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': prob * 100
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Фактори Впливу")

    reasons = []
    if products >= 3:
        st.error("Кількість продуктів: 3+")
        reasons.append("надмірна кількість продуктів")
    if age >= 45 and balance > 0:
        st.error("Вікова група ризику (45+)")
        reasons.append("вікова група з високим ризиком відтоку капіталу")
    if is_active == 0:
        st.warning("Низька активність")
        reasons.append("відсутність активності")
    if balance == 0:
        st.info("Нульовий баланс")

    if not reasons and prob < 0.2:
        st.success("Позитивна історія")

# --- 6. GENERATIVE AI ---
st.markdown("---")
st.subheader("Генерація Стратегії Утримання (OpenAI)")

if st.button("Згенерувати План Дій"):
    if not api_key:
        st.warning("Для генерації тексту необхідний API ключ.")
    else:
        client = openai.OpenAI(api_key=api_key)

        risk_context = ", ".join(
            reasons) if reasons else "Явних факторів не виявлено"

        prompt = f"""
        Виступай як професійний банківський аналітик.
        
        Профіль клієнта:
        - Вік: {age}
        - Баланс: ${balance:,.2f}
        - Продуктів: {products}
        - Ймовірність відтоку: {prob*100:.1f}%
        - Статус: {status}
        - Фактори ризику: {risk_context}
        
        Завдання:
        1. Сформулювати сухий, діловий план дій для менеджера (3 пункти).
        2. Написати текст електронного листа для клієнта. Тон: ввічливий, професійний, ненав'язливий. Запропонувати консультацію або перегляд умов.
        """

        with st.spinner("Генерація відповіді..."):
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                st.markdown(response.choices[0].message.content)
            except Exception as e:
                st.error(f"Помилка при зверненні до API: {e}")
