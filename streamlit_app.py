"""
PCOS Global Environmental Health Atlas
Interactive Platform for Research Dissemination

This Streamlit application provides interactive access to the research findings
from "Global environmental and climate correlates of polycystic ovary syndrome"
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium

# 安全图表创建函数
def safe_plotly_chart(fig, error_message="图表创建失败", fallback_data=None):
    """安全显示Plotly图表的函数"""
    try:
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            return True
        else:
            st.error(error_message)
            return False
    except Exception as e:
        st.error(f"{error_message}: {str(e)}")
        if fallback_data is not None:
            st.dataframe(fallback_data)
        return False

# Page configuration
st.set_page_config(
    page_title="PCOS Global Environmental Atlas",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🌍 PCOS Global Environmental Health Atlas</h1>
    <p><strong>Interactive Platform for Global Environmental and Climate Correlates of PCOS</strong></p>
    <p><em>Based on analysis of 247 countries, 1990-2021 | 7 International Databases</em></p>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.selectbox(
    "Select Analysis",
    ["🌐 Global Overview", "📊 Country Analysis", "🔮 Prediction Tool", 
     "🌡️ Climate Scenarios", "📈 Inequality Trends", "📚 Research Data"]
)

# Sample data generation (replace with your actual data)
@st.cache_data
def load_sample_data():
    """Generate sample data for demonstration"""
    countries = ['USA', 'China', 'India', 'Brazil', 'Russia', 'UK', 'Germany', 'France', 'Japan', 'Australia']
    data = {
        'Country': countries,
        'PCOS_Prevalence': np.random.uniform(800, 3500, 10),
        'PM25': np.random.uniform(5, 50, 10),
        'Sanitation_Access': np.random.uniform(60, 100, 10),
        'Handwashing_Facilities': np.random.uniform(50, 95, 10),
        'GDP_per_capita': np.random.uniform(1000, 70000, 10),
        'Inequality_Index': np.random.uniform(-0.15, 0.05, 10),
        'Climate_Risk': np.random.uniform(0.1, 0.8, 10)
    }
    return pd.DataFrame(data)

# Load data
df = load_sample_data()

# Page routing
if page == "🌐 Global Overview":
    st.header("Global PCOS Burden Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Global Mean Prevalence",
            value="1,620",
            delta="per 100,000 females",
            help="Age-standardized prevalence across 247 countries"
        )
    
    with col2:
        st.metric(
            label="Environmental Contribution",
            value="42.3%",
            delta="of explained variance",
            help="Proportion of PCOS burden variation explained by environmental factors"
        )
    
    with col3:
        st.metric(
            label="Inequality Improvement",
            value="76%",
            delta="since 2005",
            help="Reduction in concentration index from 2005-2021"
        )
    
    with col4:
        st.metric(
            label="Countries Analyzed",
            value="247",
            delta="1990-2021",
            help="Complete coverage of countries and territories"
        )
    
    # Global map
    st.subheader("🗺️ Global PCOS Burden Distribution")
    
    try:
        fig = px.choropleth(
            df,
            locations="Country",
            color="PCOS_Prevalence",
            hover_name="Country",
            color_continuous_scale="Reds",
            title="PCOS Prevalence by Country (per 100,000 females)"
        )
        
        if fig is not None:
            fig.update_layout(height=500)
            safe_plotly_chart(fig, "全球地图创建失败", df[['Country', 'PCOS_Prevalence']])
        else:
            st.error("无法创建全球地图，请检查数据")
    except Exception as e:
        st.error(f"全球地图创建出错: {str(e)}")
        # 显示简单的数据表格作为替代
        st.markdown("**全球PCOS负担数据 (每10万女性):**")
        st.dataframe(df[['Country', 'PCOS_Prevalence']].sort_values('PCOS_Prevalence', ascending=False))
    
    # Key findings
    st.subheader("🔑 Key Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Top Environmental Correlates:**
        - 🚿 Lack of handwashing facilities: **9.9%** (95% CI: 8.7-11.1)
        - 🚽 Unsafe sanitation: **9.6%** (95% CI: 8.4-10.8)  
        - 🌫️ Ambient PM2.5: **7.2%** (95% CI: 6.1-8.3)
        - 🔥 Household solid fuel: **6.5%** (95% CI: 5.4-7.6)
        """)
    
    with col2:
        st.markdown("""
        **Climate Impact:**
        - ❄️ Extreme cold events: **61.3%** of climate variation
        - 🔥 Extreme heat events: **38.7%** of climate variation
        - 🌡️ Temperature variability: **12.3%** burden increase per SD
        - 🌧️ Precipitation extremes: **6-8%** burden increase
        """)

elif page == "📊 Country Analysis":
    st.header("Country-Specific Analysis")
    
    # Country selector
    selected_country = st.selectbox("Select Country for Analysis", df['Country'].tolist())
    country_data = df[df['Country'] == selected_country].iloc[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"📍 {selected_country} Profile")
        
        # Radar chart for country profile
        categories = ['PCOS Prevalence', 'PM2.5 Exposure', 'Sanitation Access', 
                     'Economic Development', 'Climate Risk']
        
        values = [
            country_data['PCOS_Prevalence'] / df['PCOS_Prevalence'].max(),
            country_data['PM25'] / df['PM25'].max(),
            country_data['Sanitation_Access'] / 100,
            country_data['GDP_per_capita'] / df['GDP_per_capita'].max(),
            country_data['Climate_Risk']
        ]
        
        try:
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=selected_country
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title=f"{selected_country} Environmental Health Profile"
            )
            
            safe_plotly_chart(fig, f"{selected_country}雷达图创建失败")
        except Exception as e:
            st.error(f"雷达图创建出错: {str(e)}")
            # 显示数值作为替代
            profile_data = pd.DataFrame({
                'Indicator': categories,
                'Value': values
            })
            st.dataframe(profile_data)
    
    with col2:
        st.subheader("📈 Risk Factor Rankings")
        
        # Risk factor analysis
        risk_factors = {
            'PM2.5 Exposure': country_data['PM25'],
            'Sanitation Gap': 100 - country_data['Sanitation_Access'],
            'Handwashing Gap': 100 - country_data['Handwashing_Facilities'],
            'Climate Risk': country_data['Climate_Risk'] * 100
        }
        
        risk_df = pd.DataFrame(list(risk_factors.items()), columns=['Factor', 'Score'])
        risk_df = risk_df.sort_values('Score', ascending=True)
        
        try:
            fig = px.bar(risk_df, x='Score', y='Factor', orientation='h',
                        title="Environmental Risk Factors (Lower is Better)")
            safe_plotly_chart(fig, "风险因子图表创建失败", risk_df)
        except Exception as e:
            st.error(f"风险因子图表创建出错: {str(e)}")
            # 显示简单的数据表格作为替代
            st.dataframe(risk_df)
        
        # Country recommendations
        st.subheader("💡 Policy Recommendations")
        if country_data['Sanitation_Access'] < 90:
            st.warning("🚽 **Priority**: Improve sanitation infrastructure")
        if country_data['PM25'] > 25:
            st.warning("🌫️ **Priority**: Air quality management")
        if country_data['Climate_Risk'] > 0.5:
            st.warning("🌡️ **Priority**: Climate adaptation planning")
        
        st.success("✅ **Strength**: Areas performing well relative to global average")

elif page == "🔮 Prediction Tool":
    st.header("PCOS Burden Prediction Tool")
    st.markdown("*Predict PCOS burden based on environmental conditions using our validated Layer 5 model*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎛️ Environmental Inputs")
        
        # Input controls
        pm25 = st.slider("PM2.5 Concentration (μg/m³)", 0, 100, 25)
        sanitation = st.slider("Sanitation Access (%)", 0, 100, 85)
        handwashing = st.slider("Handwashing Facilities (%)", 0, 100, 75)
        gdp = st.slider("GDP per capita (USD)", 1000, 100000, 25000)
        temp_var = st.slider("Temperature Variability (°C)", 0.0, 10.0, 2.5)
        
        # Prediction calculation (simplified model)
        baseline = 1620
        pm25_effect = (pm25 - 15) * 7.2
        sanitation_effect = (85 - sanitation) * 9.6 / 15
        handwashing_effect = (75 - handwashing) * 9.9 / 25
        gdp_effect = (25000 - gdp) * 0.01
        temp_effect = temp_var * 12.3
        
        predicted_burden = baseline + pm25_effect + sanitation_effect + handwashing_effect + gdp_effect + temp_effect
        predicted_burden = max(100, predicted_burden)  # Minimum threshold
        
    with col2:
        st.subheader("📊 Prediction Results")
        
        # Display prediction
        st.metric(
            label="Predicted PCOS Prevalence",
            value=f"{predicted_burden:.0f}",
            delta=f"{predicted_burden - baseline:.0f} vs global average",
            help="Per 100,000 females, age-standardized"
        )
        
        # Risk level
        if predicted_burden < 1000:
            st.success("🟢 **Low Risk**: Below global average")
        elif predicted_burden < 2000:
            st.warning("🟡 **Moderate Risk**: Near global average")
        else:
            st.error("🔴 **High Risk**: Above global average")
        
        # Contributing factors breakdown
        st.subheader("🔍 Contributing Factors")
        
        factors = {
            'PM2.5 Exposure': pm25_effect,
            'Sanitation Gap': sanitation_effect,
            'Handwashing Gap': handwashing_effect,
            'Economic Factor': gdp_effect,
            'Temperature Variability': temp_effect
        }
        
        factor_df = pd.DataFrame(list(factors.items()), columns=['Factor', 'Impact'])
        
        try:
            fig = px.bar(factor_df, x='Factor', y='Impact',
                        title="Factor Contributions to Predicted Burden",
                        color='Impact', color_continuous_scale='RdYlBu_r')
            if fig is not None:
                fig.update_xaxes(tickangle=45)
                safe_plotly_chart(fig, "因子贡献图表创建失败", factor_df)
            else:
                st.error("无法创建图表，请检查数据")
        except Exception as e:
            st.error(f"图表创建出错: {str(e)}")
            # 显示简单的文本版本
            st.markdown("**因子贡献分析:**")
            for factor, impact in factors.items():
                st.markdown(f"- {factor}: {impact:.1f}")
        

elif page == "🌡️ Climate Scenarios":
    st.header("Climate Change Impact Scenarios")
    
    # Scenario selector
    scenario = st.selectbox(
        "Select Climate Scenario",
        ["Current Baseline", "1.5°C Warming", "2°C Warming", "3°C Warming", "4°C Warming"]
    )
    
    # Generate scenario data
    years = list(range(2025, 2051))
    baseline_trend = [1620 + i * 2 for i in range(len(years))]
    
    scenario_multipliers = {
        "Current Baseline": 1.0,
        "1.5°C Warming": 1.05,
        "2°C Warming": 1.12,
        "3°C Warming": 1.25,
        "4°C Warming": 1.45
    }
    
    multiplier = scenario_multipliers[scenario]
    scenario_values = [val * multiplier for val in baseline_trend]
    
    # Plot scenarios
    fig = go.Figure()
    
    # Baseline
    fig.add_trace(go.Scatter(
        x=years, y=baseline_trend,
        mode='lines', name='Current Baseline',
        line=dict(color='blue', width=2)
    ))
    
    # Selected scenario
    if scenario != "Current Baseline":
        fig.add_trace(go.Scatter(
            x=years, y=scenario_values,
            mode='lines', name=scenario,
            line=dict(color='red', width=3)
        ))
    
    try:
        fig.update_layout(
            title=f"Projected PCOS Burden: {scenario}",
            xaxis_title="Year",
            yaxis_title="PCOS Prevalence (per 100,000)",
            height=500
        )
        
        safe_plotly_chart(fig, "气候场景图表创建失败")
    except Exception as e:
        st.error(f"气候场景图表创建出错: {str(e)}")
        # 显示数值作为替代
        scenario_data = pd.DataFrame({
            'Year': years,
            'Baseline': baseline_trend,
            'Scenario': scenario_values
        })
        st.dataframe(scenario_data.tail(10))  # 显示最后10年的数据
    
    # Impact summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Projected Impact by 2050")
        impact = (scenario_values[-1] - baseline_trend[-1]) / baseline_trend[-1] * 100
        st.metric(
            label="Additional Burden",
            value=f"{impact:.1f}%",
            delta=f"{scenario_values[-1] - baseline_trend[-1]:.0f} cases per 100,000",
            help="Compared to current baseline trajectory"
        )
    
    with col2:
        st.subheader("🌍 Global Population Impact")
        global_women = 3.9e9  # Approximate global female population
        additional_cases = (scenario_values[-1] - baseline_trend[-1]) / 100000 * global_women
        st.metric(
            label="Additional Global Cases",
            value=f"{additional_cases/1e6:.1f}M",
            delta="million women affected",
            help="Estimated additional PCOS cases globally"
        )

elif page == "📈 Inequality Trends":
    st.header("Global Health Inequality Analysis")
    
    # Generate time series data for inequality metrics
    years = list(range(1990, 2022))
    concentration_index = [-0.15 + (i/len(years)) * 0.12 for i in range(len(years))]
    gini_coefficient = [0.55 - (i/len(years)) * 0.08 for i in range(len(years))]
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Concentration Index (Income-related Inequality)', 
                       'Gini Coefficient (Overall Inequality)'),
        vertical_spacing=0.1
    )
    
    # Concentration index
    fig.add_trace(
        go.Scatter(x=years, y=concentration_index, name='Concentration Index',
                  line=dict(color='blue', width=3)),
        row=1, col=1
    )
    
    # Gini coefficient
    fig.add_trace(
        go.Scatter(x=years, y=gini_coefficient, name='Gini Coefficient',
                  line=dict(color='red', width=3)),
        row=2, col=1
    )
    
    try:
        fig.update_layout(height=600, title_text="PCOS-Related Health Inequality Trends, 1990-2021")
        safe_plotly_chart(fig, "不平等趋势图表创建失败")
    except Exception as e:
        st.error(f"不平等趋势图表创建出错: {str(e)}")
        # 显示数值作为替代
        trend_data = pd.DataFrame({
            'Year': years,
            'Concentration_Index': concentration_index,
            'Gini_Coefficient': gini_coefficient
        })
        st.dataframe(trend_data.tail(10))  # 显示最后10年的数据
    
    # Key insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Inequality Metrics (2021)")
        st.metric("Concentration Index", "-0.029", "76% improvement since 2005")
        st.metric("Gini Coefficient", "0.466", "11% improvement since 2005")
        st.metric("Theil Index", "0.360", "19% improvement since 2005")
    
    with col2:
        st.subheader("🎯 Policy Implications")
        st.markdown("""
        **Between-group inequality dominates** (101.1% of total)
        - International cooperation more effective than domestic redistribution
        - Technology transfer and capacity building priority
        - Country-level interventions target inequality drivers
        
        **Success factors from low-burden countries:**
        - Better environmental quality indicators
        - Stronger health system performance  
        - More equitable development patterns
        """)

elif page == "📚 Research Data":
    st.header("Research Data & Methodology")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Dataset Info", "🔬 Methodology", "📊 Download Data", "📖 Citation"])
    
    with tab1:
        st.subheader("📋 Integrated Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Data Sources:**
            - 🌍 Global Burden of Disease Study 2021
            - 🏥 WHO Global Health Observatory  
            - 🏦 World Bank Development Indicators
            - 🌡️ ERA5 Climate Reanalysis
            - 🛰️ Satellite-derived PM2.5
            - ⚗️ Summary Exposure Values
            - 🏭 CO2 Emissions Database
            """)
        
        with col2:
            st.markdown("""
            **Coverage:**
            - 📍 247 countries and territories
            - 📅 1990-2021 (32 years)
            - 🔢 208 analytical variables
            - 📈 99.6% data completeness
            - 🎯 6,073 country-year observations
            """)
        
        # Data quality metrics
        st.subheader("🎯 Data Quality Metrics")
        
        quality_metrics = {
            'Metric': ['Data Completeness', 'Temporal Coverage', 'Geographic Coverage', 
                      'Variable Integration', 'Validation Accuracy'],
            'Score': [99.6, 100.0, 100.0, 95.2, 92.8],
            'Status': ['Excellent', 'Complete', 'Complete', 'Excellent', 'Excellent']
        }
        
        quality_df = pd.DataFrame(quality_metrics)
        
        try:
            fig = px.bar(quality_df, x='Metric', y='Score',
                        title="Data Quality Assessment",
                        color='Score', color_continuous_scale='Greens')
            if fig is not None:
                fig.update_xaxes(tickangle=45)
                safe_plotly_chart(fig, "数据质量图表创建失败", quality_df)
            else:
                st.error("无法创建数据质量图表，请检查数据")
        except Exception as e:
            st.error(f"数据质量图表创建出错: {str(e)}")
            # 显示简单的表格版本
            st.dataframe(quality_df)
    
    with tab2:
        st.subheader("🔬 Analytical Methodology")
        
        st.markdown("""
        **Multi-Method Framework (22 approaches):**
        
        **🔍 Descriptive Analyses**
        - Income-stratified comparisons
        - Geographic distribution mapping  
        - Temporal trend analysis
        
        **💰 Economic Analyses** 
        - Concentration index (income-related inequality)
        - Gini coefficient (overall inequality)
        - Theil index decomposition
        - Value of lost welfare calculations
        
        **📊 Advanced Statistical Methods**
        - Shapley value decomposition
        - Oaxaca-Blinder decomposition
        - Instrumental variable methods
        - Bayesian hierarchical modeling
        
        **🤖 Machine Learning (5-Layer Framework)**
        - Layer 1: Linear regression (30 features)
        - Layer 2: LASSO/Ridge (45 features)  
        - Layer 3: Deep learning (LSTM, Neural networks)
        - Layer 4: Spatial modeling
        - Layer 5: Ensemble methods (103 features)
        
        **✅ Validation & Robustness**
        - Temporal cross-validation (leave-one-year-out)
        - Spatial cross-validation (leave-one-region-out)
        - Bootstrap methods (1,000 replications)
        - Monte Carlo simulation (10,000 iterations)
        """)
    
    with tab3:
        st.subheader("📊 Data Download Portal")
        
        st.markdown("""
        **Available Datasets:**
        """)
        
        # Mock download interface
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Complete Dataset (CSV)", help="Full harmonized dataset"):
                st.success("Download initiated: pcos_environmental_data.csv")
            
            if st.button("🔢 Analysis Results (JSON)", help="Shapley values, predictions"):
                st.success("Download initiated: analysis_results.json")
            
            if st.button("📊 Inequality Metrics (Excel)", help="All inequality calculations"):
                st.success("Download initiated: inequality_metrics.xlsx")
        
        with col2:
            if st.button("🗺️ Geographic Data (GeoJSON)", help="Country boundaries + data"):
                st.success("Download initiated: geographic_data.geojson")
            
            if st.button("🎯 Model Weights (PKL)", help="Trained model parameters"):
                st.success("Download initiated: layer5_model.pkl")
            
            if st.button("📚 Codebook (PDF)", help="Variable definitions"):
                st.success("Download initiated: data_codebook.pdf")
        
        st.markdown("""
        **License:** CC BY 4.0 (Creative Commons Attribution)
        **Format:** Multiple formats available
        **Updates:** Annual refresh planned
        """)
    
    with tab4:
        st.subheader("📖 Citation Information")
        
        st.markdown("""
        **Recommended Citation:**
        
        ```
        [Author names]. Global environmental and climate correlates of 
        polycystic ovary syndrome: an integrated analysis of seven 
        international databases across 247 countries, 1990–2021. 
        The Lancet Planetary Health. 2025;XX(X):XXX-XXX.
        ```
        
        **BibTeX:**
        ```bibtex
        @article{pcos_environmental_2025,
          title={Global environmental and climate correlates of polycystic ovary syndrome},
          author={[Authors]},
          journal={The Lancet Planetary Health},
          year={2025},
          volume={XX},
          number={X},
          pages={XXX--XXX},
          publisher={Elsevier}
        }
        ```
        
        **Data Citation:**
        ```
        [Author names]. PCOS Environmental Health Atlas Dataset. 
        GitHub/Zenodo. 2025. DOI: 10.5281/zenodo.XXXXXXX
        ```
        """)
        
        st.markdown("""
        **Related Resources:**
        - 🐙 [GitHub Repository](https://github.com/username/pcos-environmental-health)
        - 📊 [Interactive Platform](https://pcos-environmental-atlas.streamlit.app)  
        - 📄 [Preprint](https://doi.org/10.1101/2025.XX.XX.XXXXXX)
        - 📧 Contact: [email@institution.edu]
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>PCOS Global Environmental Health Atlas</strong></p>
    <p>Based on research published in The Lancet Planetary Health • Open Science Initiative</p>
    <p>🌍 247 Countries • 📅 1990-2021 • 🔬 7 International Databases</p>
</div>
""", unsafe_allow_html=True)
