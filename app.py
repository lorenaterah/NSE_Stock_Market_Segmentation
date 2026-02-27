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
    <div style='background-color: ##2c3e50; padding: 10px; border-radius: 10px;'>
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

tab1, tab2, tab3, tab4, tab5, = st.tabs([
    "📊 Cluster Overview",
    "📈 Stock Analysis",
    "📈 Trends & Insights",  
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
    st.header("🔍 Individual Stock Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
    # Stock selector card 
        st.markdown("""
    <div style='background-color: #2c3e50; padding: 12px; border-radius: 10px; color: #ffffff;'>
        <h4 style='margin-top:0; color: #ffffff;'>Select a Stock</h4>
    </div>
    """, unsafe_allow_html=True)

        
        # Search/filter options
        search_method = st.radio(
            "Search by:",
            ["Dropdown List", "Search by Code", "Search by Name"],
            horizontal=True
        )
        
        if search_method == "Dropdown List":
            # Sort stocks alphabetically
            stock_list = sorted(df_clustered['Stock_code'].unique())
            selected_stock = st.selectbox(
                "Choose a stock:",
                options=stock_list,
                index=None,
                placeholder="Select a stock..."
            )
        elif search_method == "Search by Code":
            stock_code = st.text_input("Enter stock code (e.g., SCOM):").upper()
            matching_stocks = df_clustered[df_clustered['Stock_code'].str.contains(stock_code, na=False)]
            if len(matching_stocks) == 1:
                selected_stock = matching_stocks.iloc[0]['Stock_code']
            elif len(matching_stocks) > 1:
                selected_stock = st.selectbox(
                    "Multiple matches:",
                    options=matching_stocks['Stock_code'].tolist()
                )
            else:
                selected_stock = None
                if stock_code:
                    st.warning("No matching stock found")
        else:  # Search by Name
            stock_name = st.text_input("Enter company name:").upper()
            matching_stocks = df_clustered[df_clustered['Name'].str.contains(stock_name, na=False, case=False)]
            if len(matching_stocks) == 1:
                selected_stock = matching_stocks.iloc[0]['Stock_code']
            elif len(matching_stocks) > 1:
                selected_stock = st.selectbox(
                    "Multiple matches:",
                    options=matching_stocks['Stock_code'].tolist()
                )
            else:
                selected_stock = None
                if stock_name:
                    st.warning("No matching company found")
    
    with col2:
        if selected_stock:
            # Get stock data
            stock_data = df_clustered[df_clustered['Stock_code'] == selected_stock].iloc[0]
            
            # Risk profile header with color
            risk_color = RISK_COLORS.get(stock_data['Risk_Profile'], '#808080')
            st.markdown(f"""
            <div style='background-color: {risk_color}20; padding: 20px; border-radius: 10px; border-left: 5px solid {risk_color};'>
                <h2 style='margin:0; color: {risk_color};'>{stock_data['Stock_code']}</h2>
                <p style='margin:5px 0; font-size: 18px;'>{stock_data['Name']}</p>
                <p style='margin:0;'><b>Sector:</b> {stock_data['Sector']}</p>
                <p style='margin:10px 0 0 0; font-size: 24px; font-weight: bold; color: {risk_color};'>
                    {stock_data['Risk_Profile']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Key metrics in columns
            st.subheader("📊 Key Risk Metrics")
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric(
                    "Sharpe Ratio",
                    f"{stock_data['sharpe_ratio']:.3f}",
                    help="Risk-adjusted return. Higher is better."
                )
            
            with metric_col2:
                st.metric(
                    "Volatility",
                    f"{stock_data['volatility_mean']:.1%}",
                    help="Annual price volatility. Higher = riskier."
                )
            
            with metric_col3:
                st.metric(
                    "Max Drawdown",
                    f"{stock_data['max_drawdown']:.1%}",
                    help="Worst historical decline."
                )
            
            with metric_col4:
                st.metric(
                    "VaR (95%)",
                    f"{stock_data['var_95']:.1%}",
                    help="Value at Risk - 95% confidence"
                )
            
            # Additional metrics in expander
            with st.expander("📈 Technical Indicators & Advanced Metrics"):
                tech_col1, tech_col2, tech_col3 = st.columns(3)
                
                with tech_col1:
                    st.metric("RSI", f"{stock_data['rsi_mean']:.1f}")
                    st.metric("Momentum (20d)", f"{stock_data['momentum_20d']:.1%}")
                    st.metric("Volume Volatility", f"{stock_data['volume_volatility']:.3f}")
                
                with tech_col2:
                    st.metric("Downside Deviation", f"{stock_data['downside_deviation']:.3f}")
                    st.metric("Return Consistency", f"{stock_data['return_consistency']:.3f}")
                    st.metric("Trend Strength", f"{stock_data['trend_strength']:.3f}")
                
                with tech_col3:
                    st.metric("Bollinger Width", f"{stock_data['bb_width_mean']:.3f}")
                    st.metric("MACD Volatility", f"{stock_data['macd_volatility']:.3f}")
                    st.metric("Illiquidity", f"{stock_data['amihud_illiquidity']:.6f}")
            
            # Comparison with sector and cluster
            st.subheader("📊 Peer Comparison")
            
            # Get sector peers
            sector_peers = df_clustered[
                (df_clustered['Sector'] == stock_data['Sector']) & 
                (df_clustered['Stock_code'] != selected_stock)
            ]
            
            # Get cluster peers
            cluster_peers = df_clustered[
                (df_clustered['Risk_Profile'] == stock_data['Risk_Profile']) & 
                (df_clustered['Stock_code'] != selected_stock)
            ]
            
            comp_col1, comp_col2 = st.columns(2)
            
            with comp_col1:
                st.markdown(f"**Same Sector ({stock_data['Sector']})**")
                if len(sector_peers) > 0:
                    # Calculate sector averages
                    sector_avg_sharpe = sector_peers['sharpe_ratio'].mean()
                    sector_avg_vol = sector_peers['volatility_mean'].mean()
                    
                    # Show comparison
                    comp_data = pd.DataFrame({
                        'Metric': ['Sharpe Ratio', 'Volatility'],
                        'This Stock': [stock_data['sharpe_ratio'], stock_data['volatility_mean']],
                        'Sector Avg': [sector_avg_sharpe, sector_avg_vol]
                    })
                    
                    fig_comp = go.Figure(data=[
                        go.Bar(name='This Stock', x=comp_data['Metric'], y=comp_data['This Stock'], 
                               marker_color=risk_color),
                        go.Bar(name='Sector Avg', x=comp_data['Metric'], y=comp_data['Sector Avg'], 
                               marker_color='lightgray')
                    ])
                    fig_comp.update_layout(barmode='group', height=300, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(fig_comp, use_container_width=True)
                else:
                    st.info(f"No other stocks in {stock_data['Sector']} sector")
            
            with comp_col2:
                st.markdown(f"**Same Risk Profile ({stock_data['Risk_Profile']})**")
                if len(cluster_peers) > 0:
                    # Get top peers by Sharpe
                    top_peers = cluster_peers.nlargest(5, 'sharpe_ratio')[['Stock_code', 'sharpe_ratio', 'volatility_mean']]
                    st.dataframe(
                        top_peers,
                        use_container_width=True,
                        column_config={
                            'Stock_code': 'Stock',
                            'sharpe_ratio': st.column_config.NumberColumn('Sharpe', format='%.3f'),
                            'volatility_mean': st.column_config.NumberColumn('Volatility', format='%.1%')
                        }
                    )
                else:
                    st.info(f"No other stocks in {stock_data['Risk_Profile']} category")
            
            # Radar chart for risk dimensions
            st.subheader("🎯 Risk Profile Radar")
            
            # Select features for radar
            radar_features = {
                'Sharpe Ratio': stock_data['sharpe_ratio'],
                'Return Consistency': stock_data['return_consistency'],
                'Liquidity': 1 - stock_data['amihud_illiquidity'] * 100,  # Inverse for better visualization
                'Trend Strength': stock_data['trend_strength'],
                'Momentum': stock_data['momentum_20d'] + 0.5  # Shift to positive
            }
            
            # Normalize for radar (0-1 scale)
            radar_values = []
            for feature, value in radar_features.items():
                if feature == 'Sharpe Ratio':
                    norm_val = min(max((value + 0.5) / 2, 0), 1)  # Normalize Sharpe to 0-1
                elif feature == 'Liquidity':
                    norm_val = min(max(value, 0), 1)
                else:
                    norm_val = min(max(value, 0), 1)
                radar_values.append(norm_val)
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_values + [radar_values[0]],  # Close the loop
                theta=list(radar_features.keys()) + [list(radar_features.keys())[0]],
                fill='toself',
                name=selected_stock,
                line_color=risk_color,
                fillcolor=f'rgba{tuple(int(risk_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.3,)}'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
        else:
            # Placeholder when no stock selected
            st.info("👈 Select a stock from the left panel to view detailed analysis")
            st.markdown("""
            ### What you'll see:
            - 📊 **Key Risk Metrics** - Sharpe ratio, volatility, max drawdown
            - 📈 **Technical Indicators** - RSI, momentum, trend strength
            - 👥 **Peer Comparison** - How it compares to sector and risk category
            - 🎯 **Risk Radar** - Multi-dimensional risk profile visualization
            """)



# TAB 3: TRENDS & INSIGHTS (TIME SERIES)
with tab3:
    st.header("📈 Trends & Insights (2021-2024)")
    st.markdown("Visualize how stocks and sectors have performed over time")
    
    if df_time is not None and not df_time.empty:
        # Use Day Price instead of Adjusted Price
        price_col = 'Day Price'  # Changed from 'Adjusted Price'
        
        # Date range info
        min_date = pd.to_datetime(df_time['Date']).min()
        max_date = pd.to_datetime(df_time['Date']).max()
        
        st.info(f"📅 Data from {min_date.strftime('%B %Y')} to {max_date.strftime('%B %Y')} | {df_time['Stock_code'].nunique()} stocks | {len(df_time):,} trading days")
        
        # Main controls
        view_type = st.radio(
            "Select View:",
            ["Single Stock", "Compare Stocks", "Sector Performance"],
            horizontal=True
        )
        
        if view_type == "Single Stock":
            st.subheader("📈 Single Stock Analysis")
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                # Stock selector
                stock_list = sorted(df_time['Stock_code'].unique())
                selected_stock = st.selectbox("Select Stock:", stock_list)
                
                # Date range
                date_range = st.date_input(
                    "Date Range:",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                
                if len(date_range) == 2:
                    start_date, end_date = date_range
                else:
                    start_date, end_date = min_date, max_date
                
                # Moving averages
                show_ma = st.checkbox("Show Moving Averages", value=False)
                if show_ma:
                    ma_20 = st.checkbox("20-day MA", value=True)
                    ma_50 = st.checkbox("50-day MA", value=False)
            
            with col2:
                # Get stock data
                mask = (df_time['Stock_code'] == selected_stock) & \
                       (pd.to_datetime(df_time['Date']) >= pd.to_datetime(start_date)) & \
                       (pd.to_datetime(df_time['Date']) <= pd.to_datetime(end_date))
                stock_data = df_time[mask].copy()
                stock_data['Date'] = pd.to_datetime(stock_data['Date'])
                stock_data = stock_data.sort_values('Date')
                
                if not stock_data.empty:
                    # Check if we have valid price data
                    if stock_data[price_col].isna().all() or (stock_data[price_col] == 0).all():
                        st.error(f"⚠️ No valid price data for {selected_stock}")
                        st.write("Please check if this stock has trading data for the selected period.")
                    else:
                        # Get stock name
                        stock_name = stock_data['Name'].iloc[0] if 'Name' in stock_data.columns else selected_stock
                        
                        # Display stock info
                        st.markdown(f"**{selected_stock}** - {stock_name}")
                        
                        # Create line chart
                        fig = go.Figure()
                        
                        # Main price line
                        fig.add_trace(go.Scatter(
                            x=stock_data['Date'],
                            y=stock_data[price_col],
                            mode='lines',
                            name='Price',
                            line=dict(color='#2c3e50', width=2)
                        ))
                        
                        # Add moving averages if selected
                        if show_ma:
                            if ma_20:
                                stock_data['MA20'] = stock_data[price_col].rolling(20).mean()
                                fig.add_trace(go.Scatter(
                                    x=stock_data['Date'],
                                    y=stock_data['MA20'],
                                    mode='lines',
                                    name='20-day MA',
                                    line=dict(color='#e74c3c', width=1.5, dash='dash')
                                ))
                            
                            if ma_50:
                                stock_data['MA50'] = stock_data[price_col].rolling(50).mean()
                                fig.add_trace(go.Scatter(
                                    x=stock_data['Date'],
                                    y=stock_data['MA50'],
                                    mode='lines',
                                    name='50-day MA',
                                    line=dict(color='#27ae60', width=1.5, dash='dash')
                                ))
                        
                        # Update layout
                        fig.update_layout(
                            title=f'{selected_stock} - Price History',
                            xaxis_title='Date',
                            yaxis_title='Price (KES)',
                            height=500,
                            hovermode='x unified',
                            showlegend=True,
                            template='plotly_white'
                        )
                        
                        # Show the plot
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show basic stats
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Current Price", f"KES {stock_data[price_col].iloc[-1]:.2f}")
                        with col2:
                            st.metric("Highest", f"KES {stock_data[price_col].max():.2f}")
                        with col3:
                            st.metric("Lowest", f"KES {stock_data[price_col].min():.2f}")
                        with col4:
                            change = ((stock_data[price_col].iloc[-1] / stock_data[price_col].iloc[0]) - 1) * 100
                            st.metric("Total Return", f"{change:.1f}%")
                else:
                    st.warning("No data available for selected period")
        
        elif view_type == "Compare Stocks":
            st.subheader("📊 Stock Comparison")
            
            # Stock selector
            stock_options = sorted(df_time['Stock_code'].unique())
            selected_stocks = st.multiselect(
                "Select stocks to compare:",
                options=stock_options,
                max_selections=5
            )
            
            if selected_stocks:
                # Date range
                date_range = st.date_input(
                    "Date Range:",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    
                    # Create performance table
                    st.subheader("📈 Period Returns")
                    
                    perf_data = []
                    for stock in selected_stocks:
                        mask = (df_time['Stock_code'] == stock) & \
                               (pd.to_datetime(df_time['Date']) >= pd.to_datetime(start_date)) & \
                               (pd.to_datetime(df_time['Date']) <= pd.to_datetime(end_date))
                        stock_data = df_time[mask].copy()
                        stock_data = stock_data.sort_values('Date')
                        
                        if len(stock_data) > 1 and not stock_data[price_col].isna().all():
                            # Filter out zero values if they exist
                            valid_data = stock_data[stock_data[price_col] > 0]
                            if len(valid_data) > 1:
                                start_price = valid_data[price_col].iloc[0]
                                end_price = valid_data[price_col].iloc[-1]
                                total_return = ((end_price / start_price) - 1) * 100
                                highest = valid_data[price_col].max()
                                lowest = valid_data[price_col].min()
                                
                                perf_data.append({
                                    'Stock': stock,
                                    'Start Price (KES)': f"{start_price:.2f}",
                                    'End Price (KES)': f"{end_price:.2f}",
                                    'Total Return': f"{total_return:.1f}%",
                                    'Highest (KES)': f"{highest:.2f}",
                                    'Lowest (KES)': f"{lowest:.2f}"
                                })
                    
                    if perf_data:
                        df_perf = pd.DataFrame(perf_data)
                        st.dataframe(df_perf, use_container_width=True)
                    else:
                        st.warning("No valid price data available for selected stocks/period")
        
        else:  # Sector Performance
            st.subheader("🏢 Sector Performance")
            
            # Get sectors
            if 'Sector' in df_time.columns:
                sectors = sorted(df_time['Sector'].unique())
            else:
                sectors = ["All Stocks"]
            
            selected_sector = st.selectbox("Select Sector:", sectors)
            
            # Date range
            date_range = st.date_input(
                "Date Range:",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                
                # Get stocks in this sector
                if selected_sector == "All Stocks" or 'Sector' not in df_time.columns:
                    sector_stocks = df_time['Stock_code'].unique()
                else:
                    sector_stocks = df_time[df_time['Sector'] == selected_sector]['Stock_code'].unique()
                
                if len(sector_stocks) > 0:
                    # Filter data for sector stocks
                    mask = (df_time['Stock_code'].isin(sector_stocks)) & \
                           (pd.to_datetime(df_time['Date']) >= pd.to_datetime(start_date)) & \
                           (pd.to_datetime(df_time['Date']) <= pd.to_datetime(end_date))
                    sector_data = df_time[mask].copy()
                    sector_data['Date'] = pd.to_datetime(sector_data['Date'])
                    
                    if not sector_data.empty and not sector_data[price_col].isna().all():
                        # Filter out zero prices
                        sector_data = sector_data[sector_data[price_col] > 0]
                        
                        if not sector_data.empty:
                            # Calculate sector average performance
                            sector_avg = sector_data.groupby('Date')[price_col].mean().reset_index()
                            sector_avg['Normalized'] = (sector_avg[price_col] / sector_avg[price_col].iloc[0]) * 100
                            
                            # Create line chart
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                x=sector_avg['Date'],
                                y=sector_avg['Normalized'],
                                mode='lines',
                                name=f'{selected_sector} Average',
                                line=dict(color='#2c3e50', width=3)
                            ))
                            
                            fig.update_layout(
                                title=f'{selected_sector} Sector Performance (Base 100)',
                                xaxis_title='Date',
                                yaxis_title='Performance (Base 100)',
                                height=500,
                                hovermode='x unified',
                                template='plotly_white'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show sector statistics
                            st.subheader("📊 Sector Statistics")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Number of Stocks", len(sector_stocks))
                            with col2:
                                if sector_avg[price_col].iloc[0] > 0:
                                    total_return = ((sector_avg[price_col].iloc[-1] / sector_avg[price_col].iloc[0]) - 1) * 100
                                    st.metric("Sector Return", f"{total_return:.1f}%")
                            with col3:
                                # Calculate average volatility
                                volatilities = []
                                for stock in sector_stocks[:10]:
                                    stock_prices = sector_data[sector_data['Stock_code'] == stock][price_col]
                                    if len(stock_prices) > 20:
                                        returns = stock_prices.pct_change().dropna()
                                        if len(returns) > 0 and returns.std() > 0:
                                            vol = returns.std() * (252**0.5) * 100
                                            volatilities.append(vol)
                                if volatilities:
                                    st.metric("Avg Volatility", f"{np.mean(volatilities):.1f}%")
                        else:
                            st.warning("No valid price data after filtering zeros")
                    else:
                        st.warning("No data available for selected sector/period")
                else:
                    st.warning("No stocks found in this sector")
    
    else:
        st.warning("⚠️ Time series data not found.")




# TAB 4: ABOUT THE PROJECT

with tab4:
    st.header("ℹ️ About This Project")
    
    # Project Overview - Simple and direct
    st.markdown("""
    ### 🎯 What This Dashboard Does
    This tool clusters **Nairobi Stock Exchange (NSE)** stocks into **4 risk profiles** using machine learning. 
    It helps investors quickly identify stocks that match their risk tolerance.
    
    **Key achievement:** Improved clustering quality from *Silhouette 0.32 → 0.717* through advanced feature engineering.
    """)
    
    # Quick Stats Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Stocks", f"{len(df_clustered)}", "NSE companies")
    with col2:
        st.metric("🔧 Features", "19", "Risk dimensions")
    with col3:
        st.metric("🎯 Clusters", "4", "Risk profiles")
    with col4:
        st.metric("📈 Silhouette", "0.717", "+0.397 improvement")
    
    st.markdown("---")
    
    # The Four Risk Profiles - Visual cards
    st.subheader("📊 The Four Risk Profiles")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background-color: #00CC9620; padding: 15px; border-radius: 10px; border-left: 5px solid #00CC96; margin-bottom: 15px;'>
            <h4 style='color: #00CC96; margin:0;'>🟢 Low Risk</h4>
            <p style='margin:5px 0 0 0;'>Stable, liquid stocks with consistent returns. Ideal for conservative investors.</p>
        </div>
        
        <div style='background-color: #FFA15A20; padding: 15px; border-radius: 10px; border-left: 5px solid #FFA15A; margin-bottom: 15px;'>
            <h4 style='color: #FFA15A; margin:0;'>🟠 Medium-Low Risk</h4>
            <p style='margin:5px 0 0 0;'>Balanced risk/reward profile. Good for moderate risk tolerance.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background-color: #EF553B20; padding: 15px; border-radius: 10px; border-left: 5px solid #EF553B; margin-bottom: 15px;'>
            <h4 style='color: #EF553B; margin:0;'>🔴 Medium-High Risk</h4>
            <p style='margin:5px 0 0 0;'>Growth-oriented stocks with higher volatility. For aggressive investors.</p>
        </div>
        
        <div style='background-color: #AB63FA20; padding: 15px; border-radius: 10px; border-left: 5px solid #AB63FA; margin-bottom: 15px;'>
            <h4 style='color: #AB63FA; margin:0;'>🟣 High Risk</h4>
            <p style='margin:5px 0 0 0;'>Speculative stocks with large price swings. High risk, high potential reward.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Methodology in 3 simple steps
    st.subheader("🔬 How It Works")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **1️⃣ Data Processing**
        - 4 years of daily NSE data
        - Cleaned & validated
        - 57 liquid stocks
        """)
        
    with col2:
        st.markdown("""
        **2️⃣ Feature Engineering** ⭐
        - 19 risk indicators
        - Technical + liquidity metrics
        - Risk-adjusted returns
        """)
        
    with col3:
        st.markdown("""
        **3️⃣ Clustering**
        - K-Means algorithm
        - 4 optimal clusters
        - Silhouette validation
        """)
    
    # Key insight expander
    with st.expander("📈 **What Makes This Different?**"):
        st.markdown("""
        The magic is in **feature engineering**. Instead of just using volatility, we captured multiple risk dimensions:
        
        | Category | Features | What It Tells Us |
        |----------|----------|------------------|
        | **Risk Metrics** | Volatility, Drawdown, VaR | How much can you lose? |
        | **Returns** | Sharpe ratio, Consistency | Is the risk worth it? |
        | **Technical** | RSI, MACD, Momentum | What's the trend? |
        | **Liquidity** | Volume, Amihud ratio | Can you trade it? |
        
        **Result:** Before, two stocks with similar volatility looked the same. Now we can tell apart:
        - 🟢 A stable dividend payer from 
        - 🔴 A distressed speculative stock
        """)
    
    # Simple disclaimer
    st.markdown("---")
    st.subheader("⚠️ Important")
    
    warn_col1, warn_col2 = st.columns(2)
    
    with warn_col1:
        st.warning("**📜 Not Financial Advice**")
        st.caption("This is for **educational purposes only**. Always do your own research before investing.")
        
        st.info("**📊 Data Source**")
        st.caption("Historical data from NSE (2021-2024). Past performance ≠ future results.")
    
    with warn_col2:
        st.warning("**🔧 Model Limitations**")
        st.caption("Based on price data only. Fundamentals (P/E, debt) not included. Use as screening tool, not final decision.")
        
        st.info("**🎯 Best Used For**")
        st.caption("Quick risk screening • Portfolio diversification • Educational purposes")
    
    # Footer with links
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Repo-blue)](https://github.com/lorenaterah/NSE_Stock_Market_Segmentation)")
    with col2:
        st.markdown("[![Docs](https://img.shields.io/badge/Documentation-Guide-green)](PROJECT_GUIDE.md)")
    with col3:
        st.markdown("[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)")
    with col4:
        st.markdown(f"**v2.0** | {pd.Timestamp.now().year}")




# TAB 5: TEAM MORINGA - DEVELOPERS

with tab5:
    st.header("👥 Meet the Team - Moringa School Group 3")
    st.markdown("*Data Science Cohort | Machine Learning Capstone Project*")
    
    # Team introduction
    st.markdown("""
    <div style='background-color: #404E4D; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <p style='font-size: 16px;'>We are Group 3 from Moringa School's Data Science program, DSF-FT14. This NSE Stock Risk Profiler represents our 
        collaborative effort to apply unsupervised learning techniques to real financial data from the Nairobi Securities Exchange.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Team members in columns (2 rows of 3)
    st.subheader("🌟 Team Members")
    
    # Row 1 of team members
    col1, col2, col3 = st.columns(3) 
    
    with col1:
        st.markdown("""
        <div style='background-color: #1e1e1e; padding: 15px; border-radius: 10px; border-left: 5px solid #FF4B4B; color: white; text-align: center;'>
            <h4 style='color: #FF4B4B; margin:0;'>Muema Stephen</h4>
            <p style='color: white; font-size: 14px;'>Data Scientist</p>
            <p style='margin-top: 10px;'>
                <a href="https://github.com/Kaks753" style='color: #FF4B4B; text-decoration: none;'>🔗 @Kaks753</a>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background-color: #1e1e1e; padding: 15px; border-radius: 10px; border-left: 5px solid #3498db; color: white; text-align: center;'>
            <h4 style='color: #3498db; margin:0;'>Sharon Kipruto</h4>
            <p style='color: white; font-size: 14px;'>Data Scientist</p>
            <p style='margin-top: 10px;'>
                <a href="https://github.com/sharonkipruto-code" style='color: #3498db; text-decoration: none;'>🔗 @sharonkipruto-code</a>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background-color: #1e1e1e; padding: 15px; border-radius: 10px; border-left: 5px solid #27ae60; color: white; text-align: center;'>
            <h4 style='color: #27ae60; margin:0;'>Salma Mwende</h4>
            <p style='color: white; font-size: 14px;'>Data Scientist</p>
            <p style='margin-top: 10px;'>
                <a href="https://github.com/salmamwende-code" style='color: #27ae60; text-decoration: none;'>🔗 @salmamwende-code</a>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Row 2 of team members
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown("""
        <div style='background-color: #1e1e1e; padding: 15px; border-radius: 10px; border-left: 5px solid #f39c12; color: white; text-align: center;'>
            <h4 style='color: #f39c12; margin:0;'>Lorena Terah</h4>
            <p style='color: white; font-size: 14px;'>Data Scientist</p>
            <p style='margin-top: 10px;'>
                <a href="https://github.com/lorenaterah" style='color: #f39c12; text-decoration: none;'>🔗 @lorenaterah</a>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div style='background-color: #1e1e1e; padding: 15px; border-radius: 10px; border-left: 5px solid #9b59b6; color: white; text-align: center;'>
            <h4 style='color: #9b59b6; margin:0;'>Edgar Owuor</h4>
            <p style='color: white; font-size: 14px;'>Data Scientist</p>
            <p style='margin-top: 10px;'>
                <a href="https://github.com/edgarowuor-tech" style='color: #9b59b6; text-decoration: none;'>🔗 @edgarowuor-tech</a>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        st.markdown("""
        <div style='background-color: #1e1e1e; padding: 15px; border-radius: 10px; border-left: 5px solid #e74c3c; color: white; text-align: center;'>
            <h4 style='color: #e74c3c; margin:0;'>Dawa Jarso</h4>
            <p style='color: white; font-size: 14px;'>Data Scientist</p>
            <p style='margin-top: 10px;'>
                <a href="https://github.com/Dawa-Jarso" style='color: #e74c3c; text-decoration: none;'>🔗 @Dawa-Jarso</a>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Project Overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Project Overview")
        st.markdown("""
        **NSE Stock Risk Profiler** was developed as our Phase 4 capstone project at Moringa School. 
        
        **Objective:** Cluster NSE stocks into risk profiles (Low/Medium/High) using unsupervised learning 
        on 4 years of historical data (2021-2024).
        
        **Key Achievements:**
        - ✅ 19 financial features engineered from raw data
        - ✅ Silhouette score improved from 0.32 → 0.717
        - ✅ 57+ NSE stocks successfully clustered
        - ✅ Interactive dashboard with 6 functional tabs
        """)
    
    with col2:
        st.subheader("🔧 Tools & Technologies")
        st.markdown("""
        <div style='background-color: #222525; padding: 15px; border-radius: 10px;'>
            <p><strong>Languages:</strong> Python</p>
            <p><strong>Libraries:</strong> pandas, numpy, scikit-learn, plotly, matplotlib</p>
            <p><strong>Framework:</strong> Streamlit</p>
            <p><strong>Version Control:</strong> Git, GitHub</p>
            <p><strong>Development:</strong> VS Code</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Skills demonstrated - as a grid
    st.subheader("💡 Skills Demonstrated")
    
    skill_cols = st.columns(4)
    skills = [
        ("🐍 Python", "pandas, numpy, sklearn"),
        ("📊 Visualization", "plotly, matplotlib"),
        ("🤖 ML", "K-Means, PCA, Scaling"),
        ("🌐 Web App", "streamlit"),
        ("📈 Finance", "Risk metrics, technical analysis"),
        ("🔧 MLOps", "Model persistence, caching"),
        ("👥 Collaboration", "Git, GitHub, Agile"),
        ("📝 Documentation", "Technical writing")
    ]
    
    for i, (skill, desc) in enumerate(skills):
        with skill_cols[i % 4]:
            st.markdown(f"""
            <div style='background-color: #1e1e1e; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                <p style='color: #FF4B4B; font-weight: bold; margin:0;'>{skill}</p>
                <p style='color: white; font-size: 12px; margin:0;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Acknowledgments
    st.subheader("🙏 Special Thanks")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='text-align: center;'>
            <p>👨‍🏫 <strong>Technical Mentors</strong></p>
            <p style='color: gray;'>Diana Mongina<br>Samuel Karu</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center;'>
            <p>🏢 <strong>Data Source</strong></p>
            <p style='color: gray;'>Nairobi Securities Exchange</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center;'>
            <p>💻 <strong>Tools</strong></p>
            <p style='color: gray;'>VS Code<br>GitHub<br>Streamlit Cloud</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # GitHub repository link
    st.markdown(f"""
    <div style='background-color: #2c3e50; padding: 20px; border-radius: 10px; text-align: center;'>
        <h4 style='color: white; margin:0;'>🔗 View Project on GitHub</h4>
        <a href="https://github.com/lorenaterah/NSE_Stock_Market_Segmentation" target="_blank" style='text-decoration: none;'>
            <p style='color: #0000ff; font-size: 16px;'>https://github.com/lorenaterah/NSE_Stock_Market_Segmentation</p>
        </a>
        <p style='color: white; font-size: 14px;'>⭐ Star us if you find this project useful!</p>
    </div>
    """, unsafe_allow_html=True)    
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
        <p><strong>Moringa School Group 3</strong> | Data Science Cohort-DSFT-FT14 | February 2026</p>
    </div>
    """, unsafe_allow_html=True)