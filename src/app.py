import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

# Page config 
st.set_page_config(
    page_title="NSE Stock Risk Clustering",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 NSE Stock Risk Clustering Dashboard")
st.markdown("""
This dashboard helps investors classify Nairobi Stock Exchange stocks into **risk categories** 
using machine learning.
""")

# COLOR SCHEME 

RISK_COLORS = {
    'Low Risk': '#00CC96',      # Green
    'Medium-Low': '#FFA15A',     # Orange
    'Medium-High': '#EF553B',    # Red-Orange
    'High Risk': '#AB63FA'       # Purple
}

# LOADING DATA

@st.cache_data
def load_clustered_data():
    
    path = Path("Data/Processed/nse_clustered.csv")
    
    if not path.exists():
        st.error("❌ File not found: Data/Processed/nse_clustered.csv")
        return None
    
    df = pd.read_csv(path)
    return df

@st.cache_data
def load_features_data():

    path = Path("Data/Processed/nse_features.csv")
    
    if not path.exists():
        return None
    
    return pd.read_csv(path)

@st.cache_data
def load_time_series_data():

    path = Path("Data/Processed/cleaned_nse.csv")
    
    if not path.exists():
        return None
    
    # Load data
    df = pd.read_csv(path)
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date
    df = df.sort_values(['Stock_code', 'Date'])
    
    return df

# Load time series data
df_time = load_time_series_data()

@st.cache_resource
def load_model():

    path = Path("models/stock_clusterer.pkl")
    
    if not path.exists():
        st.warning("⚠️ Model not found. Using cluster labels from CSV.")
        return None
    
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.warning(f"⚠️ Could not load model: {e}")
        return None

# Loading everything
df_clustered = load_clustered_data() 
df_features = load_features_data() 
model = load_model() 

# Stop if no data
if df_clustered is None:
    st.stop()

# SIDEBAR 

st.sidebar.header("🔧 Dashboard Controls")

# Description
st.sidebar.markdown("""
    <small>Filter stocks by risk profile, sector, and key metrics</small>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")

# Risk Profile Selection - Using expander to save space
with st.sidebar.expander("📊 **Risk Profiles**", expanded=True):
    risk_options = df_clustered['Risk_Profile'].unique()
    
    # "Select All" checkbox
    select_all_risk = st.checkbox("Select All Risk Profiles", value=True, key="select_all_risk")
    
    if select_all_risk:
        selected_risks = st.multiselect(
            "Choose risk levels:",
            options=risk_options,
            default=risk_options.tolist(),
            label_visibility="collapsed"
        )
    else:
        selected_risks = st.multiselect(
            "Choose risk levels:",
            options=risk_options,
            default=[],
            label_visibility="collapsed"
        )
    
    # Show count
    st.caption(f"Selected: {len(selected_risks)} of {len(risk_options)}")

# Sector Selection - Using dropdown style
with st.sidebar.expander("🏢 **Sectors**", expanded=True):
    sector_options = df_clustered['Sector'].unique()
    
    # "Select All" checkbox for sectors
    select_all_sector = st.checkbox("Select All Sectors", value=True, key="select_all_sector")
    
    if select_all_sector:
        selected_sectors = st.multiselect(
            "Choose sectors:",
            options=sector_options,
            default=sector_options.tolist(),
            label_visibility="collapsed"
        )
    else:
        selected_sectors = st.multiselect(
            "Choose sectors:",
            options=sector_options,
            default=[],
            label_visibility="collapsed"
        )
    
    st.caption(f"Selected: {len(selected_sectors)} of {len(sector_options)}")

st.sidebar.markdown("---")

# Metric Filters - Using sliders in columns for compactness
st.sidebar.subheader("📈 Metric Ranges")

# Sharpe Ratio in a clean card-like container
with st.sidebar.container():
    st.markdown("**Sharpe Ratio**")
    col1, col2 = st.columns(2)
    with col1:
        sharpe_min = st.number_input(
            "Min",
            min_value=float(df_clustered['sharpe_ratio'].min()),
            max_value=float(df_clustered['sharpe_ratio'].max()),
            value=float(df_clustered['sharpe_ratio'].min()),
            format="%.2f",
            key="sharpe_min"
        )
    with col2:
        sharpe_max = st.number_input(
            "Max",
            min_value=float(df_clustered['sharpe_ratio'].min()),
            max_value=float(df_clustered['sharpe_ratio'].max()),
            value=float(df_clustered['sharpe_ratio'].max()),
            format="%.2f",
            key="sharpe_max"
        )
    
    # Add a range slider for visual
    sharpe_range = st.slider(
        "Range",
        min_value=float(df_clustered['sharpe_ratio'].min()),
        max_value=float(df_clustered['sharpe_ratio'].max()),
        value=(sharpe_min, sharpe_max),
        label_visibility="collapsed",
        key="sharpe_slider"
    )
    sharpe_min, sharpe_max = sharpe_range

# Volatility in a clean card-like container
with st.sidebar.container():
    st.markdown("**Volatility**")
    col1, col2 = st.columns(2)
    with col1:
        vol_min = st.number_input(
            "Min",
            min_value=float(df_clustered['volatility_mean'].min()),
            max_value=float(df_clustered['volatility_mean'].max()),
            value=float(df_clustered['volatility_mean'].min()),
            format="%.3f",
            key="vol_min"
        )
    with col2:
        vol_max = st.number_input(
            "Max",
            min_value=float(df_clustered['volatility_mean'].min()),
            max_value=float(df_clustered['volatility_mean'].max()),
            value=float(df_clustered['volatility_mean'].max()),
            format="%.3f",
            key="vol_max"
        )
    
    vol_range = st.slider(
        "Range",
        min_value=float(df_clustered['volatility_mean'].min()),
        max_value=float(df_clustered['volatility_mean'].max()),
        value=(vol_min, vol_max),
        label_visibility="collapsed",
        key="vol_slider"
    )
    vol_min, vol_max = vol_range

st.sidebar.markdown("---")

# Apply Filters Button
col1, col2 = st.sidebar.columns(2)
with col1:
    apply_filters = st.button("🎯 Apply Filters", type="primary", use_container_width=True)
with col2:
    reset_filters = st.button("🔄 Reset", use_container_width=True)

# Handle reset
if reset_filters:
    selected_risks = risk_options.tolist()
    selected_sectors = sector_options.tolist()
    sharpe_min = float(df_clustered['sharpe_ratio'].min())
    sharpe_max = float(df_clustered['sharpe_ratio'].max())
    vol_min = float(df_clustered['volatility_mean'].min())
    vol_max = float(df_clustered['volatility_mean'].max())
    st.rerun()

# Apply filters (only if button pressed or on initial load)
if apply_filters or 'filters_applied' not in st.session_state:
    st.session_state.filters_applied = True
    filtered_df = df_clustered[
        (df_clustered['Risk_Profile'].isin(selected_risks)) &
        (df_clustered['Sector'].isin(selected_sectors)) &
        (df_clustered['sharpe_ratio'].between(sharpe_min, sharpe_max)) &
        (df_clustered['volatility_mean'].between(vol_min, vol_max))
    ]
else:
    filtered_df = df_clustered[
        (df_clustered['Risk_Profile'].isin(selected_risks)) &
        (df_clustered['Sector'].isin(selected_sectors)) &
        (df_clustered['sharpe_ratio'].between(sharpe_min, sharpe_max)) &
        (df_clustered['volatility_mean'].between(vol_min, vol_max))
    ]

# Summary stats in a card at the bottom
st.sidebar.markdown("---")
with st.sidebar.container():
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px;'>
        <h4 style='margin:0; text-align: center;'>📊 Summary</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric(
            "Total Stocks",
            len(df_clustered),
            delta=None
        )
    with col2:
        st.metric(
            "Filtered",
            len(filtered_df),
            delta=f"{len(filtered_df)-len(df_clustered)}",
            delta_color="off"
        )
    
    # Show risk distribution in a mini chart
    risk_dist = filtered_df['Risk_Profile'].value_counts()
    st.markdown("**Risk Distribution:**")
    for risk in risk_options:
        count = risk_dist.get(risk, 0)
        pct = (count/len(filtered_df)*100) if len(filtered_df) > 0 else 0
        st.markdown(
            f"<span style='color: {RISK_COLORS.get(risk, '#808080')}'>⬤</span> {risk}: {count} ({pct:.0f}%)",
            unsafe_allow_html=True
        )

# Optional: Add a footer with last update
st.sidebar.markdown("---")
st.sidebar.caption("🔄 Last updated: " + pd.Timestamp.now().strftime("%Y-%m-%d"))


# MAIN DASHBOARD TABS 

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Cluster Overview",
    "📈 Stock Analysis",
    "📈 Trends & Insights",  
    "🔬 Feature Explorer",
    "ℹ️ About",
    "👨‍💻 Developer"
])


# TAB 1: CLUSTER OVERVIEW

with tab1:
    st.header("📊 Risk Cluster Distribution")
    
    # Two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart of risk distribution
        risk_counts = filtered_df['Risk_Profile'].value_counts().reset_index()
        risk_counts.columns = ['Risk_Profile', 'Count']
        
        fig_pie = px.pie(
            risk_counts,
            values='Count',
            names='Risk_Profile',
            title='Stocks by Risk Profile',
            color='Risk_Profile',
            color_discrete_map=RISK_COLORS,
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar chart of sectors by risk
        sector_risk = pd.crosstab(
            filtered_df['Sector'],
            filtered_df['Risk_Profile']
        )
        
        fig_bar = px.bar(
            sector_risk,
            title='Sector Distribution by Risk Profile',
            barmode='stack',
            color_discrete_map=RISK_COLORS
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Risk-Return scatter plot
    st.subheader("🎯 Risk vs Return by Cluster")
    
    fig_scatter = px.scatter(
        filtered_df,
        x='volatility_mean',
        y='sharpe_ratio',
        color='Risk_Profile',
        size='volume_volatility' if 'volume_volatility' in filtered_df.columns else None,
        hover_data=['Stock_code', 'Sector', 'Name', 'max_drawdown'],
        title='Sharpe Ratio vs Volatility',
        color_discrete_map=RISK_COLORS,
        labels={
            'volatility_mean': 'Volatility (Risk)',
            'sharpe_ratio': 'Sharpe Ratio (Return/Risk)'
        }
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Summary statistics table
    st.subheader("📈 Key Metrics by Risk Profile")
    
    summary_stats = filtered_df.groupby('Risk_Profile').agg({
        'sharpe_ratio': ['mean', 'min', 'max'],
        'volatility_mean': ['mean', 'min', 'max'],
        'max_drawdown': ['mean', 'min', 'max'],
        'Stock_code': 'count'
    }).round(3)
    
    # Rename columns for display
    summary_stats.columns = ['Sharpe_Mean', 'Sharpe_Min', 'Sharpe_Max',
                            'Vol_Mean', 'Vol_Min', 'Vol_Max',
                            'DD_Mean', 'DD_Min', 'DD_Max',
                            'Count']
    summary_stats = summary_stats.reset_index()
    
    st.dataframe(
        summary_stats,
        use_container_width=True,
        column_config={
            'Risk_Profile': 'Risk Profile',
            'Count': '# Stocks',
            'Sharpe_Mean': st.column_config.NumberColumn('Avg Sharpe', format='%.3f'),
            'Vol_Mean': st.column_config.NumberColumn('Avg Volatility', format='%.3f'),
            'DD_Mean': st.column_config.NumberColumn('Avg Max DD', format='%.3f')
        }
    )

# Tab2: STOCK ANALYSIS
with tab2:
    st.header("📈 Stock Analysis")
    
    # Stock selection
    stock_options = filtered_df['Stock_code'].unique()
    selected_stock = st.selectbox(
        "Select a stock to analyze:",
        options=stock_options,
        index=0
    )
    
    # Get stock details
    stock_details = filtered_df[filtered_df['Stock_code'] == selected_stock].iloc[0]
    
    # Display stock info
    st.subheader(f"📊 {stock_details['Name']} ({stock_details['Stock_code']})")
    st.markdown(f"**Sector:** {stock_details['Sector']}")
    st.markdown(f"**Risk Profile:** {stock_details['Risk_Profile']}")
    
with tab6:
    st.header("Group 3")
    st.markdown("""
    This project was developed by **Group 3** as part of the Data Science Bootcamp at Moringa School.
    """)