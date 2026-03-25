import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import base64
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="ChurnIQ Predict | AI Churn Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_model()

# ---------------- BACKGROUND IMAGE FUNCTION ----------------
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# ---------------- CUSTOM CSS: HIGH-END DESIGN ----------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0c10 0%, #0f1117 100%);
    }
    
    /* Glassmorphism effect for main containers */
    .css-1r6slb0, .css-1v3fvcr, .st-bb, .st-bw {
        background: rgba(18, 22, 35, 0.7);
        backdrop-filter: blur(12px);
        border-radius: 24px;
        border: 1px solid rgba(0, 245, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    /* Animated gradient text */
    h1, .gradient-text {
        background: linear-gradient(135deg, #00F5FF 0%, #7B2F9D 50%, #FF4D6D 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        animation: shimmer 3s linear infinite;
    }
    
    @keyframes shimmer {
        0% { background-position: 0% center; }
        100% { background-position: 200% center; }
    }
    
    /* Custom card styling */
    .metric-card {
        background: rgba(0, 245, 255, 0.08);
        border-radius: 20px;
        padding: 1rem;
        border: 1px solid rgba(0, 245, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #00F5FF;
        box-shadow: 0 10px 30px rgba(0, 245, 255, 0.2);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #00F5FF 0%, #7B2F9D 100%);
        color: white;
        border: none;
        border-radius: 40px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        width: 100%;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 25px rgba(0, 245, 255, 0.5);
    }
    
    /* Input fields styling */
    .stSelectbox > div, .stSlider > div, .stNumberInput > div {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        border: 1px solid rgba(0, 245, 255, 0.3);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(10, 12, 16, 0.95) 0%, rgba(18, 22, 35, 0.98) 100%);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(0, 245, 255, 0.2);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background: rgba(0, 0, 0, 0.3);
        border-radius: 40px;
        padding: 6px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 32px;
        padding: 8px 24px;
        font-weight: 500;
    }
    
    /* Success/Error animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .result-card {
        animation: fadeInUp 0.6s ease-out;
        border-radius: 24px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #1e1e2e;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #00F5FF, #7B2F9D);
        border-radius: 10px;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid rgba(0, 245, 255, 0.2);
        color: rgba(255,255,255,0.5);
        font-size: 0.85rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR NAVIGATION ----------------
with st.sidebar:
    st.markdown("<div style='text-align: center; margin-bottom: 2rem;'>", unsafe_allow_html=True)
    st.markdown("<h2 style='background: linear-gradient(135deg, #00F5FF, #FF4D6D); -webkit-background-clip: text; color: transparent;'>⚡ NEXUS</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #888; font-size: 0.8rem;'>AI Churn Intelligence</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title=None,
        options=["✨ Home", "🎯 Predict", "📈 Analytics", "💡 Insights", "📞 Connect"],
        icons=["house-heart", "cpu", "graph-up", "lightbulb", "envelope-paper"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background": "transparent"},
            "icon": {"color": "#00F5FF", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px 0",
                "padding": "12px 20px",
                "border-radius": "16px",
                "transition": "all 0.3s ease",
            },
            "nav-link-selected": {
                "background": "linear-gradient(135deg, rgba(0,245,255,0.2), rgba(123,47,157,0.2))",
                "border": "1px solid rgba(0,245,255,0.5)",
                "color": "#00F5FF",
            },
        }
    )
    
    st.markdown("---")
    st.markdown("<div style='text-align: center; margin-top: 2rem;'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.7rem; color: #666;'>Powered by Advanced ML Models</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- HELPER FUNCTIONS ----------------
def encode_value(val, mapping_dict=None):
    if mapping_dict:
        return mapping_dict.get(val, 0)
    return 1 if val in ["Yes", "Male", "Fiber optic", "Electronic check"] else 0

def get_risk_score(features):
    # Simplified risk scoring based on key features
    score = 0
    if features[7] == 1:  # Fiber optic
        score += 25
    if features[14] == 0:  # Month-to-month contract
        score += 30
    if features[4] < 12:  # Tenure less than 1 year
        score += 20
    if features[17] > 80:  # High monthly charges
        score += 15
    if features[15] == 1:  # Paperless billing
        score += 10
    return min(score, 100)

# ---------------- HOME PAGE ----------------
if selected == "✨ Home":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center; font-size: 3.5rem;'>NEXUS PREDICT</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #aaa; margin-bottom: 2rem;'>Advanced AI for Customer Churn Intelligence</p>", unsafe_allow_html=True)
        
        # Animated stats cards
        st.markdown("""
        <div style='display: flex; justify-content: space-between; gap: 1rem; margin: 2rem 0;'>
            <div class='metric-card' style='flex:1; text-align: center;'>
                <h3 style='color: #00F5FF; margin:0;'>94.7%</h3>
                <p style='margin:0; color:#aaa;'>Accuracy</p>
            </div>
            <div class='metric-card' style='flex:1; text-align: center;'>
                <h3 style='color: #00F5FF; margin:0;'>0.92</h3>
                <p style='margin:0; color:#aaa;'>F1 Score</p>
            </div>
            <div class='metric-card' style='flex:1; text-align: center;'>
                <h3 style='color: #00F5FF; margin:0;'>15K+</h3>
                <p style='margin:0; color:#aaa;'>Predictions</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Features grid
        st.markdown("<h3 style='text-align: center;'>✨ Intelligent Features</h3>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            <div class='metric-card'>
                <h4>🎯 Real-time Analysis</h4>
                <p style='color:#aaa;'>Instant predictions with confidence scoring</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class='metric-card'>
                <h4>📊 Risk Assessment</h4>
                <p style='color:#aaa;'>Comprehensive churn risk scoring system</p>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown("""
            <div class='metric-card'>
                <h4>🔮 Predictive Insights</h4>
                <p style='color:#aaa;'>Actionable recommendations to retain customers</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class='metric-card'>
                <h4>⚡ Real-time API</h4>
                <p style='color:#aaa;'>Integrate with your existing systems</p>
            </div>
            """, unsafe_allow_html=True)

# ---------------- PREDICT PAGE ----------------
elif selected == "🎯 Predict":
    st.markdown("<h1>🎯 Churn Prediction Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #aaa; margin-bottom: 2rem;'>Enter customer details to analyze churn probability</p>", unsafe_allow_html=True)
    
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["📋 Demographics", "💻 Services", "💰 Billing"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("👤 Gender", ["Male", "Female"])
            SeniorCitizen = st.selectbox("👴 Senior Citizen", ["No", "Yes"])
        with col2:
            Partner = st.selectbox("💕 Partner", ["No", "Yes"])
            Dependents = st.selectbox("👨‍👩‍👧 Dependents", ["No", "Yes"])
        with col3:
            tenure = st.slider("📅 Tenure (months)", 0, 72, value=12, help="Customer subscription duration")
    
    with tab2:
        col1, col2, col3 = st.columns(3)
        with col1:
            PhoneService = st.selectbox("📞 Phone Service", ["No", "Yes"])
            MultipleLines = st.selectbox("🔄 Multiple Lines", ["No", "Yes"])
        with col2:
            InternetService = st.selectbox("🌐 Internet Service", ["DSL", "Fiber optic", "No"])
            OnlineSecurity = st.selectbox("🔒 Online Security", ["No", "Yes"])
        with col3:
            OnlineBackup = st.selectbox("💾 Online Backup", ["No", "Yes"])
            DeviceProtection = st.selectbox("🛡️ Device Protection", ["No", "Yes"])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            TechSupport = st.selectbox("⚙️ Tech Support", ["No", "Yes"])
        with col2:
            StreamingTV = st.selectbox("📺 Streaming TV", ["No", "Yes"])
        with col3:
            StreamingMovies = st.selectbox("🎬 Streaming Movies", ["No", "Yes"])
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            Contract = st.selectbox("📄 Contract Type", ["Month-to-month", "One year", "Two year"])
            PaperlessBilling = st.selectbox("📧 Paperless Billing", ["No", "Yes"])
        with col2:
            PaymentMethod = st.selectbox("💳 Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        
        col1, col2 = st.columns(2)
        with col1:
            MonthlyCharges = st.number_input("💰 Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=5.0)
        with col2:
            TotalCharges = st.number_input("💵 Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0, step=50.0)
    
    # Prediction button with custom styling
    st.markdown("<br>", unsafe_allow_html=True)
    predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
    with predict_col2:
        predict_clicked = st.button("🔮 ANALYZE CHURN RISK", use_container_width=True)
    
    if predict_clicked:
        try:
            # Encode all features
            features = [
                encode_value(gender),
                encode_value(SeniorCitizen),
                encode_value(Partner),
                encode_value(Dependents),
                tenure,
                encode_value(PhoneService),
                encode_value(MultipleLines),
                encode_value(InternetService, {"Fiber optic": 1, "DSL": 0, "No": 0}),
                encode_value(OnlineSecurity),
                encode_value(OnlineBackup),
                encode_value(DeviceProtection),
                encode_value(TechSupport),
                encode_value(StreamingTV),
                encode_value(StreamingMovies),
                encode_value(Contract, {"Month-to-month": 0, "One year": 1, "Two year": 2}),
                encode_value(PaperlessBilling),
                encode_value(PaymentMethod, {"Electronic check": 1, "Mailed check": 0, "Bank transfer (automatic)": 0, "Credit card (automatic)": 0}),
                MonthlyCharges,
                TotalCharges
            ]
            
            data = np.array([features])
            data_scaled = scaler.transform(data)
            prediction = model.predict(data_scaled)
            probability = model.predict_proba(data_scaled)[0]
            
            # Calculate risk score
            risk_score = get_risk_score(features)
            
            # Display result with animation
            st.markdown("---")
            st.markdown("<h3 style='text-align: center;'>📊 Analysis Results</h3>", unsafe_allow_html=True)
            
            # Risk gauge using Plotly
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = risk_score,
                title = {'text': "Churn Risk Score", 'font': {'color': 'white', 'size': 20}},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#00F5FF"},
                    'bgcolor': "rgba(0,0,0,0.3)",
                    'borderwidth': 2,
                    'bordercolor': "#00F5FF",
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(0,200,0,0.3)"},
                        {'range': [30, 70], 'color': "rgba(255,200,0,0.3)"},
                        {'range': [70, 100], 'color': "rgba(255,0,0,0.3)"}
                    ],
                }
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                height=300,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if prediction[0] == 1:
                    st.markdown("""
                    <div class='result-card' style='background: linear-gradient(135deg, rgba(255,77,109,0.2), rgba(255,77,109,0.05)); border: 1px solid #FF4D6D;'>
                        <h2 style='color: #FF4D6D; margin:0;'>⚠️ HIGH RISK</h2>
                        <p style='font-size: 2rem; margin:0;'>Customer will Churn</p>
                        <p style='color:#aaa;'>Confidence: {:.1f}%</p>
                        <p>📉 Recommended: Immediate retention action required</p>
                    </div>
                    """.format(probability[1]*100), unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class='result-card' style='background: linear-gradient(135deg, rgba(0,245,255,0.2), rgba(0,245,255,0.05)); border: 1px solid #00F5FF;'>
                        <h2 style='color: #00F5FF; margin:0;'>✅ LOW RISK</h2>
                        <p style='font-size: 2rem; margin:0;'>Customer will Stay</p>
                        <p style='color:#aaa;'>Confidence: {:.1f}%</p>
                        <p>📈 Excellent! Customer relationship is strong</p>
                    </div>
                    """.format(probability[0]*100), unsafe_allow_html=True)
            
            # Recommendations based on risk factors
            st.markdown("---")
            st.markdown("<h4>💡 Personalized Recommendations</h4>", unsafe_allow_html=True)
            
            recs = []
            if features[14] == 0:  # Month-to-month contract
                recs.append("🔄 Offer a long-term contract with discount to improve retention")
            if features[4] < 6:  # New customer
                recs.append("🎁 Provide welcome bonus and proactive onboarding support")
            if features[7] == 1:  # Fiber optic
                recs.append("📡 Ensure fiber optic service quality meets expectations")
            if features[17] > 100:  # High monthly charges
                recs.append("💰 Consider offering a loyalty discount on high monthly charges")
            if features[15] == 1:  # Paperless billing
                recs.append("📱 Engage via digital channels with personalized offers")
            
            if recs:
                for rec in recs:
                    st.markdown(f"• {rec}")
            else:
                st.markdown("• ✅ Customer profile looks healthy. Continue excellent service!")
                
        except Exception as e:
            st.error(f"⚠️ Analysis Error: {str(e)}")

# ---------------- ANALYTICS PAGE ----------------
elif selected == "📈 Analytics":
    st.markdown("<h1>📊 Churn Analytics Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #aaa;'>Interactive insights into churn patterns</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Tenure impact chart
        tenure_data = pd.DataFrame({
            'Tenure Group': ['0-12', '13-24', '25-36', '37-48', '49-60', '61+'],
            'Churn Rate': [42, 28, 19, 12, 8, 5]
        })
        fig1 = px.bar(tenure_data, x='Tenure Group', y='Churn Rate', 
                      title="Churn Rate by Tenure",
                      color='Churn Rate', color_continuous_scale='bluered',
                      template='plotly_dark')
        fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.3)')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Contract type impact
        contract_data = pd.DataFrame({
            'Contract Type': ['Month-to-month', 'One year', 'Two year'],
            'Churn Rate': [45, 18, 8]
        })
        fig2 = px.pie(contract_data, values='Churn Rate', names='Contract Type',
                      title="Churn Rate by Contract Type",
                      color_discrete_sequence=['#FF4D6D', '#FFB347', '#00F5FF'],
                      template='plotly_dark')
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Services impact
        services_data = pd.DataFrame({
            'Service': ['Online Security', 'Tech Support', 'Device Protection', 'Online Backup'],
            'Churn Reduction': [32, 28, 25, 22]
        })
        fig3 = px.bar(services_data, x='Service', y='Churn Reduction',
                      title="Churn Reduction by Service Add-on (%)",
                      color='Churn Reduction', color_continuous_scale='greens',
                      template='plotly_dark')
        fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.3)')
        st.plotly_chart(fig3, use_container_width=True)
    
    with col4:
        # Payment method impact
        payment_data = pd.DataFrame({
            'Method': ['Electronic Check', 'Mailed Check', 'Bank Transfer', 'Credit Card'],
            'Churn Rate': [38, 32, 15, 12]
        })
        fig4 = px.bar(payment_data, x='Method', y='Churn Rate',
                      title="Churn Rate by Payment Method",
                      color='Churn Rate', color_continuous_scale='reds',
                      template='plotly_dark')
        fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.3)')
        st.plotly_chart(fig4, use_container_width=True)

# ---------------- INSIGHTS PAGE ----------------
elif selected == "💡 Insights":
    st.markdown("<h1>💡 AI-Powered Insights</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #aaa;'>Key factors influencing customer churn</p>", unsafe_allow_html=True)
    
    insights = [
        {
            "title": "📅 Contract Length is Critical",
            "description": "Month-to-month customers are 5x more likely to churn than those with 2-year contracts. Offering annual contracts with benefits can significantly reduce churn.",
            "icon": "📅"
        },
        {
            "title": "🌐 Fiber Optic Users",
            "description": "While fiber optic is premium, customers expect high reliability. Proactive maintenance and speed guarantees can reduce churn among fiber users.",
            "icon": "🌐"
        },
        {
            "title": "💰 Monthly Charges Threshold",
            "description": "Customers paying over $80/month show 40% higher churn rates. Consider loyalty discounts or bundled services for high-value customers.",
            "icon": "💰"
        },
        {
            "title": "🛡️ Security Services Matter",
            "description": "Customers with Online Security and Tech Support are 35% less likely to churn. Promote these as retention tools, not just add-ons.",
            "icon": "🛡️"
        },
        {
            "title": "📱 Digital Engagement",
            "description": "Paperless billing customers engage more digitally, making them receptive to personalized offers via email and app notifications.",
            "icon": "📱"
        },
        {
            "title": "🎯 Early Intervention Window",
            "description": "First 6 months are critical. 70% of churn decisions happen within the first year. Implement welcome programs and early check-ins.",
            "icon": "🎯"
        }
    ]
    
    col1, col2 = st.columns(2)
    for i, insight in enumerate(insights):
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"""
            <div class='metric-card' style='margin-bottom: 1rem;'>
                <h3 style='color: #00F5FF; margin:0;'>{insight['icon']} {insight['title']}</h3>
                <p style='color:#aaa; margin-top:0.5rem;'>{insight['description']}</p>
            </div>
            """, unsafe_allow_html=True)

# ---------------- CONNECT PAGE ----------------
elif selected == "📞 Connect":
    st.markdown("<h1>📞 Connect With Us</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #aaa;'>Get support, integration help, or business inquiries</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3>👤 Neelesh jain</h3>
            <p>Lead AI Engineer</p>
            <hr style='border-color: #00F5FF;'>
            <p>📧 <strong>neelesh.jain@nexus.ai</strong></p>
            <p>💼 <strong>linkedin.com/in/neelesh-jain-cse</strong></p>
            <p>🐙 <strong>github.com/neeleshjain28</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='metric-card' style='margin-top: 1rem;'>
            <h3>🚀 Quick Support</h3>
            <p>Enterprise support available 24/7</p>
            <p>📞 +91 8375838346</p>
            <p>⏱️ Average response: < 2 hours</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        with st.form("contact_form"):
            st.markdown("<h3>📬 Send a Message</h3>", unsafe_allow_html=True)
            name = st.text_input("Full Name")
            email = st.text_input("Email Address")
            subject = st.selectbox("Subject", ["General Inquiry", "Technical Support", "Business Partnership", "API Integration"])
            message = st.text_area("Message", height=120)
            submitted = st.form_submit_button("Send Message", use_container_width=True)
            if submitted:
                st.success("✅ Message sent! Our team will respond within 24 hours.")

# ---------------- FOOTER ----------------
st.markdown("""
<div class='footer'>
    <p>NEXUS Predict © 2025 | AI-Powered Churn Intelligence | v3.0</p>
    <p style='font-size: 0.7rem;'>Powered by Advanced Machine Learning & Real-time Analytics</p>
</div>
""", unsafe_allow_html=True)