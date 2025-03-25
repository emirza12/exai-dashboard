import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="LifeSure E-commerce Insights",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a cleaner look with sustainability colors
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        padding-bottom: 1.5rem;
    }
    .section-title {
        font-size: 1.8rem;
        color: #388E3C;
        padding-top: 1rem;
        padding-bottom: 0.5rem;
    }
    .chart-title {
        font-size: 1.3rem;
        color: #43A047;
        padding-top: 0.5rem;
    }
    .insight-box {
        background-color: #E8F5E9;
        border-left: 5px solid #43A047;
        padding: 10px 15px;
        margin-bottom: 20px;
        border-radius: 0 5px 5px 0;
    }
    .about-box {
        background-color: #000000;
        color: #FFFFFF;
        border-left: 5px solid #43A047;
        padding: 15px 20px;
        margin-bottom: 20px;
        border-radius: 0 5px 5px 0;
    }
    .about-box h3 {
        color: #4CAF50;
        margin-top: 0;
    }
    .policy-box {
        background-color: #C8E6C9;
        padding: 15px;
        margin-bottom: 20px;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f9f9f9;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #388E3C;
    }
    .metric-label {
        font-size: 1rem;
        color: #616161;
    }
</style>
""", unsafe_allow_html=True)

# Define consistent color palettes for the dashboard
def get_color_palettes():
    # Main brand colors for LifeSure (green theme)
    primary_color = '#2E7D32'  # Dark green - main color
    
    # Single-hue sequential palette (greens) for continuous variables
    sequential_greens = ['#e8f5e9', '#c8e6c9', '#a5d6a7', '#81c784', '#66bb6a', '#4caf50', '#43a047', '#388e3c', '#2e7d32', '#1b5e20']
    
    # Green-focused categorical palette for discrete variables
    categorical_palette = ['#388E3C', '#4CAF50', '#66BB6A', '#81C784', '#A5D6A7', '#C8E6C9', '#3E5F22', '#558B2F', '#33691E', '#1B5E20']
    
    # Green-focused diverging palette (light to dark green)
    diverging_palette = ['#E8F5E9', '#C8E6C9', '#A5D6A7', '#81C784', '#66BB6A', '#4CAF50', '#43A047', '#388E3C', '#2E7D32', '#1B5E20']
    
    # Two-color categorical palette (lighter and darker green)
    binary_palette = ['#4CAF50', '#2E7D32']  # Medium green and dark green
    
    return {
        'primary': primary_color,
        'sequential': sequential_greens,
        'categorical': categorical_palette,
        'diverging': diverging_palette,
        'binary': binary_palette
    }

def load_data():
    """Load the customer dataset"""
    try:
        df = pd.read_csv('exported_dataset.csv')
        return df
    except FileNotFoundError:
        st.error("Error: 'exported_dataset.csv' file not found in the current directory.")
        return None

def prepare_data(df):
    """Prepare and clean the data for analysis"""
    # Frequency mapping for calculations
    frequency_mapping = {
        'Weekly': 52,
        'Fortnightly': 26,
        'Bi-Weekly': 26,
        'Monthly': 12,
        'Quarterly': 4,
        'Every 3 Months': 4,
        'Annually': 1
    }
    
    # Convert frequencies to numbers if needed
    if 'Frequency_per_Year' not in df.columns or df['Frequency_per_Year'].isna().any():
        if 'Frequency of Purchases' in df.columns:
            df['Frequency_per_Year'] = df['Frequency of Purchases'].map(frequency_mapping)
    
    # Calculate spending metrics if not already present
    if 'Estimated_Annual_Spend' not in df.columns:
        if 'Purchase Amount (USD)' in df.columns and 'Frequency_per_Year' in df.columns:
            df['Estimated_Annual_Spend'] = df['Purchase Amount (USD)'] * df['Frequency_per_Year']
    
    if 'Estimated_LTV' not in df.columns:
        if 'Estimated_Annual_Spend' in df.columns:
            df['Estimated_LTV'] = df['Estimated_Annual_Spend'] * 5  # 5-year customer lifecycle
    
    # Create age groups if not already present
    if 'Age Group' not in df.columns:
        if 'Age' in df.columns:
            bins = [0, 25, 40, 60, 120]
            labels = ['18-24', '25-39', '40-59', '60+']
            df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    
    # Create spending segments for customer value analysis
    if 'Estimated_Annual_Spend' in df.columns:
        spend_quantiles = df['Estimated_Annual_Spend'].quantile([0.33, 0.67]).tolist()
        conditions = [
            df['Estimated_Annual_Spend'] <= spend_quantiles[0],
            (df['Estimated_Annual_Spend'] > spend_quantiles[0]) & (df['Estimated_Annual_Spend'] <= spend_quantiles[1]),
            df['Estimated_Annual_Spend'] > spend_quantiles[1]
        ]
        choices = ['Low Spender', 'Medium Spender', 'High Spender']
        df['Spending_Segment'] = np.select(conditions, choices, default='Unknown')
    
    # Clean up Yes/No columns for consistency
    for col in ['Subscription Status', 'Discount Applied', 'Promo Code Used']:
        if col in df.columns:
            df[col] = df[col].map({True: 'Yes', False: 'No', 'Yes': 'Yes', 'No': 'No'})
    
    return df

def create_dashboard(df):
    # Get the color palettes first thing in the dashboard creation
    colors = get_color_palettes()
    
    # Main title
    st.markdown('<h1 class="main-title">🌿 E-commerce Insights for LifeSure Insurance</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This dashboard analyzes consumer behavior from e-commerce data to help LifeSure Insurance 
    identify patterns relevant to designing modern, sustainable insurance policies. While 
    the data comes from retail shopping, many consumer preferences and trends provide valuable 
    insights for insurance product development.
    """)
    
    # Sidebar filters with explanations
    st.sidebar.markdown('## 🔍 Explore Customer Segments')
    st.sidebar.markdown('Filter the e-commerce data to understand different customer groups:')
    
    # Gender filter
    if 'Gender' in df.columns:
        gender_options = ['All'] + sorted(df['Gender'].unique().tolist())
        selected_gender = st.sidebar.selectbox('Gender', gender_options)
    else:
        selected_gender = 'All'
    
    # Age group filter
    if 'Age Group' in df.columns:
        age_options = ['All'] + sorted(df['Age Group'].unique().tolist())
        selected_age = st.sidebar.selectbox('Age Group', age_options)
    else:
        selected_age = 'All'
    
    # Category filter
    if 'Category' in df.columns:
        category_options = ['All'] + sorted(df['Category'].unique().tolist())
        selected_category = st.sidebar.selectbox('Product Category', category_options)
    else:
        selected_category = 'All'
    
    # Subscription status filter
    if 'Subscription Status' in df.columns:
        subscription_options = ['All'] + sorted(df['Subscription Status'].unique().tolist())
        selected_subscription = st.sidebar.selectbox('Subscription Status', subscription_options)
    else:
        selected_subscription = 'All'
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_gender != 'All':
        filtered_df = filtered_df[filtered_df['Gender'] == selected_gender]
    
    if selected_age != 'All':
        filtered_df = filtered_df[filtered_df['Age Group'] == selected_age]
    
    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['Category'] == selected_category]
    
    if selected_subscription != 'All':
        filtered_df = filtered_df[filtered_df['Subscription Status'] == selected_subscription]
    
    # Check if filtered data is empty
    if filtered_df.empty:
        st.warning("No customers match your selected filters. Please try different filter options.")
        return
    
    # Display current filter selections
    active_filters = []
    if selected_gender != 'All': active_filters.append(f"Gender: {selected_gender}")
    if selected_age != 'All': active_filters.append(f"Age Group: {selected_age}")
    if selected_category != 'All': active_filters.append(f"Category: {selected_category}")
    if selected_subscription != 'All': active_filters.append(f"Subscription Status: {selected_subscription}")
    
    if active_filters:
        st.markdown(f"**Currently viewing:** {' | '.join(active_filters)}")
    
    # --- SECTION 1: KEY BUSINESS METRICS --- #
    st.markdown('<h2 class="section-title">Key Consumer Insights for Insurance Innovation</h2>', unsafe_allow_html=True)
    
    # Key metrics with explanations
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'Age' in filtered_df.columns:
            avg_age = round(filtered_df['Age'].mean(), 1)
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{avg_age}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Average Age</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if 'Estimated_Annual_Spend' in filtered_df.columns:
            avg_spend = round(filtered_df['Estimated_Annual_Spend'].mean(), 2)
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">${avg_spend:,.2f}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Avg. Annual Spend</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        if 'Subscription Status' in filtered_df.columns:
            subscription_rate = round((filtered_df['Subscription Status'] == 'Yes').mean() * 100, 1)
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{subscription_rate}%</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Subscription Rate</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        if 'Review Rating' in filtered_df.columns:
            avg_rating = round(filtered_df['Review Rating'].mean(), 1)
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{avg_rating}/5.0</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Avg. Rating</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Insights from key metrics
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("""
    💡 **Consumer Behavior Insights for Insurance:**  
    The e-commerce data reveals patterns in consumer spending, subscription preferences, and satisfaction that parallel insurance behaviors.
    
    **Insurance Application:** These metrics suggest that modern consumers are increasingly comfortable with subscription models and 
    digital engagement. LifeSure can apply these insights by developing flexible, subscription-based insurance products 
    with transparent customer satisfaction metrics.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # --- SECTION 2: CUSTOMER DEMOGRAPHICS --- #
    st.markdown('<h2 class="section-title">Consumer Demographics Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Gender' in filtered_df.columns:
            st.markdown('<h3 class="chart-title">Gender Distribution</h3>', unsafe_allow_html=True)
            
            gender_counts = filtered_df['Gender'].value_counts().reset_index()
            gender_counts.columns = ['Gender', 'Count']
            
            fig = px.pie(
                gender_counts,
                values='Count',
                names='Gender',
                color='Gender',
                color_discrete_sequence=colors['categorical'][:len(gender_counts)],
                hole=0.4
            )
            fig.update_traces(
                textposition='outside', 
                textinfo='percent+label',
                marker=dict(line=dict(color='#FFFFFF', width=2))
            )
            fig.update_layout(
                margin=dict(t=0, b=0, l=0, r=0),
                legend_title="Gender",
                legend=dict(orientation="h", yanchor="bottom", y=0, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'Age Group' in filtered_df.columns:
            st.markdown('<h3 class="chart-title">Age Group Distribution</h3>', unsafe_allow_html=True)
            
            age_counts = filtered_df['Age Group'].value_counts().reset_index()
            age_counts.columns = ['Age Group', 'Count']
            
            # Sort age groups in logical order
            age_order = ['18-24', '25-39', '40-59', '60+']
            age_counts['Age Group'] = pd.Categorical(age_counts['Age Group'], categories=age_order, ordered=True)
            age_counts = age_counts.sort_values('Age Group')
            
            fig = px.bar(
                age_counts,
                x='Age Group',
                y='Count',
                color='Age Group',
                color_discrete_sequence=colors['categorical'][:len(age_counts)],
                text_auto=True
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=10, b=30),
                xaxis_title="",
                yaxis_title="Number of Customers",
                legend_title="Age Group",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Demographics insights
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("""
    💡 **Demographic Insights for Insurance Targeting:**  
    E-commerce customer demographics reveal distinct age and gender patterns in purchasing behavior.
    
    **Insurance Application:** LifeSure can develop targeted, sustainable insurance products for specific demographic groups:
    • Younger customers (18-39) may prefer digital-first policies with mobile app integration
    • Middle-aged customers (40-59) often value family coverage with sustainability components
    • Senior customers (60+) may respond to simple, transparent policies with legacy planning options
    
    Consider demographic-specific marketing and educational materials about your sustainable insurance options.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # --- SECTION 3: PRODUCT PREFERENCES --- #
    st.markdown('<h2 class="section-title">Product Preferences & Consumer Interest</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Category' in filtered_df.columns:
            st.markdown('<h3 class="chart-title">Product Category Distribution</h3>', unsafe_allow_html=True)
            
            category_counts = filtered_df['Category'].value_counts().reset_index()
            category_counts.columns = ['Category', 'Count']
            category_counts = category_counts.sort_values('Count', ascending=False)
            
            fig = px.bar(
                category_counts,
                x='Category',
                y='Count',
                color='Category',
                color_discrete_sequence=colors['categorical'][:len(category_counts)],
                text_auto=True
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(
                height=400,
                margin=dict(l=10, r=10, t=10, b=40),
                xaxis_title="",
                yaxis_title="Number of Purchases",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'Category' in filtered_df.columns and 'Age Group' in filtered_df.columns:
            st.markdown('<h3 class="chart-title">Category Preference by Age Group</h3>', unsafe_allow_html=True)
            
            cat_age = pd.crosstab(filtered_df['Age Group'], filtered_df['Category']).apply(lambda x: x / x.sum(), axis=1)
            
            # Convert to long format for plotting
            cat_age_long = cat_age.reset_index().melt(
                id_vars=['Age Group'],
                var_name='Category',
                value_name='Percentage'
            )
            
            # Ensure age groups are in logical order
            order = ['18-24', '25-39', '40-59', '60+']
            cat_age_long['Age Group'] = pd.Categorical(cat_age_long['Age Group'], categories=order, ordered=True)
            cat_age_long = cat_age_long.sort_values('Age Group')
            
            # Only show top categories for readability
            top_categories = filtered_df['Category'].value_counts().nlargest(5).index.tolist()
            cat_age_long_filtered = cat_age_long[cat_age_long['Category'].isin(top_categories)]
            
            fig = px.bar(
                cat_age_long_filtered,
                x='Age Group',
                y='Percentage',
                color='Category',
                barmode='group',
                text_auto='.0%',
                color_discrete_sequence=colors['categorical'][:len(top_categories)]
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(
                height=400,
                margin=dict(l=10, r=10, t=10, b=40),
                xaxis_title="",
                yaxis_title="Percentage (%)",
                legend_title="Category"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Product preference insights
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    if 'Category' in filtered_df.columns:
        top_category = filtered_df['Category'].value_counts().index[0]
        
        st.markdown(f"""
        💡 **Consumer Interest Patterns for Insurance Products:**  
        The most popular e-commerce category is **{top_category}**, with clear age-based preference patterns.
        
        **Insurance Application:** Just as consumers show distinct product preferences by age group, LifeSure can:
        • Design tiered insurance offerings with different sustainability features
        • Create age-appropriate coverage bundles that reflect life stage priorities
        • Develop modular policies where customers can select relevant coverages
        
        The product preferences in e-commerce indicate how insurance offerings should be tailored for different customer segments.
        """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # --- SECTION 4: PURCHASE BEHAVIOR --- #
    st.markdown('<h2 class="section-title">Purchasing Behavior Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Purchase Amount (USD)' in filtered_df.columns:
            st.markdown('<h3 class="chart-title">Purchase Amount Distribution</h3>', unsafe_allow_html=True)
            
            # Create purchase amount bins
            purchase_bins = [0, 25, 50, 75, 100]
            purchase_labels = ['$0-25', '$25-50', '$50-75', '$75-100']
            filtered_df['Purchase Range'] = pd.cut(filtered_df['Purchase Amount (USD)'], bins=purchase_bins, labels=purchase_labels, right=True)
            
            purchase_dist = filtered_df['Purchase Range'].value_counts().reset_index()
            purchase_dist.columns = ['Purchase Amount', 'Count']
            
            # Ensure purchase ranges are in logical order
            purchase_dist['Purchase Amount'] = pd.Categorical(purchase_dist['Purchase Amount'], categories=purchase_labels, ordered=True)
            purchase_dist = purchase_dist.sort_values('Purchase Amount')
            
            fig = px.bar(
                purchase_dist,
                x='Purchase Amount',
                y='Count',
                color='Count',
                color_continuous_scale='Greens',
                text_auto=True
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=10, b=30),
                xaxis_title="Purchase Amount Range",
                yaxis_title="Number of Purchases",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'Frequency of Purchases' in filtered_df.columns:
            st.markdown('<h3 class="chart-title">Purchase Frequency Analysis</h3>', unsafe_allow_html=True)
            
            freq_counts = filtered_df['Frequency of Purchases'].value_counts().reset_index()
            freq_counts.columns = ['Purchase Frequency', 'Count']
            
            # Define order for frequency
            freq_order = ['Weekly', 'Bi-Weekly', 'Fortnightly', 'Monthly', 'Quarterly', 'Every 3 Months', 'Annually']
            freq_counts['Purchase Frequency'] = pd.Categorical(freq_counts['Purchase Frequency'], categories=freq_order, ordered=True)
            freq_counts = freq_counts.sort_values('Purchase Frequency')
            
            fig = px.bar(
                freq_counts,
                x='Purchase Frequency',
                y='Count',
                color='Count',
                color_continuous_scale='Greens',
                text_auto=True
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=10, b=30),
                xaxis_title="Purchase Frequency",
                yaxis_title="Number of Customers",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Purchase behavior insights
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("""
    💡 **E-commerce Purchase Patterns & Insurance Premium Insights:**  
    This data reveals how consumers distribute spending and their preferred payment frequencies.
    
    **Insurance Application:** LifeSure can apply these insights by:
    • Offering flexible premium payment schedules that match customers' natural payment rhythms
    • Creating tiered insurance coverage options at different price points
    • Providing micro-insurance options for specific needs with lower entry costs
    • Developing usage-based insurance with transparent pricing
    
    The way consumers spend in e-commerce offers clues about premium pricing and payment frequency preferences.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # --- SECTION 5: SUBSCRIPTION & LOYALTY --- #
    st.markdown('<h2 class="section-title">Subscription & Customer Loyalty Analysis</h2>', unsafe_allow_html=True)
    
    if 'Subscription Status' in filtered_df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="chart-title">Subscription Status Comparison</h3>', unsafe_allow_html=True)
            
            sub_counts = filtered_df['Subscription Status'].value_counts().reset_index()
            sub_counts.columns = ['Subscription Status', 'Count']
            
            fig = px.pie(
                sub_counts,
                values='Count',
                names='Subscription Status',
                color='Subscription Status',
                color_discrete_map={'Yes': '#43A047', 'No': '#E8F5E9'},
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Estimated_LTV' in filtered_df.columns:
                st.markdown('<h3 class="chart-title">Customer Lifetime Value by Subscription</h3>', unsafe_allow_html=True)
                
                subscription_ltv = filtered_df.groupby('Subscription Status')['Estimated_LTV'].mean().reset_index()
                
                fig = px.bar(
                    subscription_ltv,
                    x='Subscription Status',
                    y='Estimated_LTV',
                    color='Subscription Status',
                    color_discrete_sequence=colors['binary'],
                    text_auto='$.2f'
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(
                    height=350,
                    margin=dict(l=10, r=10, t=10, b=30),
                    xaxis_title="",
                    yaxis_title="Average Lifetime Value ($)"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate subscription impact
                sub_yes_ltv = subscription_ltv[subscription_ltv['Subscription Status'] == 'Yes']['Estimated_LTV'].values[0]
                sub_no_ltv = subscription_ltv[subscription_ltv['Subscription Status'] == 'No']['Estimated_LTV'].values[0]
                sub_impact = (sub_yes_ltv / sub_no_ltv - 1) * 100
        
        # Subscription insights
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        if 'Estimated_LTV' in filtered_df.columns:
            st.markdown(f"""
            💡 **Subscription Model Insights for Insurance Products:**  
            E-commerce data shows subscribers have a **{sub_impact:.1f}% higher** lifetime value than non-subscribers.
            
            **Insurance Application:** LifeSure can leverage subscription insights by:
            • Implementing loyalty programs that reward long-term customers
            • Creating subscription-based insurance bundles with regular adjustments
            • Offering premium discounts for policy renewals and multi-policy holders
            • Developing sustainable "green rewards" programs for loyal customers
            
            The success of e-commerce subscription models suggests similar approaches would work for modern insurance products.
            """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # --- SECTION 6: PAYMENT METHODS --- #
    st.markdown('<h2 class="section-title">Payment Method Analysis</h2>', unsafe_allow_html=True)
    
    if 'Payment Method' in filtered_df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="chart-title">Popular Payment Methods</h3>', unsafe_allow_html=True)
            
            payment_counts = filtered_df['Payment Method'].value_counts().reset_index()
            payment_counts.columns = ['Payment Method', 'Count']
            
            # Sort by count
            payment_counts = payment_counts.sort_values('Count', ascending=False)
            
            # Get top payment methods for better readability
            top_payments = payment_counts.head(5)['Payment Method'].tolist()
            
            fig = px.pie(
                payment_counts.head(5),  # Just show top 5
                values='Count',
                names='Payment Method',
                color='Payment Method',
                color_discrete_sequence=colors['categorical'][:len(payment_counts.head(5))],
                hole=0.4
            )
            fig.update_traces(
                textposition='outside', 
                textinfo='percent+label',
                marker=dict(line=dict(color='#FFFFFF', width=2))
            )
            fig.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=10, b=10),
                legend_title="Payment Method"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<h3 class="chart-title">Payment Methods by Age Group</h3>', unsafe_allow_html=True)
            
            if 'Age Group' in filtered_df.columns:
                # Filter for just the top methods
                payment_age_df = filtered_df[filtered_df['Payment Method'].isin(top_payments)]
                
                payment_age = pd.crosstab(payment_age_df['Age Group'], payment_age_df['Payment Method'])
                payment_age_pct = payment_age.div(payment_age.sum(axis=1), axis=0) * 100
                
                payment_age_long = payment_age_pct.reset_index().melt(
                    id_vars=['Age Group'],
                    var_name='Payment Method',
                    value_name='Percentage'
                )
                
                # Ensure age groups are in logical order
                order = ['18-24', '25-39', '40-59', '60+']
                payment_age_long['Age Group'] = pd.Categorical(payment_age_long['Age Group'], categories=order, ordered=True)
                payment_age_long = payment_age_long.sort_values('Age Group')
                
                fig = px.bar(
                    payment_age_long,
                    x='Age Group',
                    y='Percentage',
                    color='Payment Method',
                    barmode='group',
                    text_auto='.1f',
                    color_discrete_sequence=colors['categorical'][:len(payment_age_long['Payment Method'].unique())]
                )
                fig.update_traces(texttemplate='%{text}%', textposition='outside')
                fig.update_layout(
                    height=400,
                    margin=dict(l=10, r=10, t=10, b=40),
                    xaxis_title="",
                    yaxis_title="Percentage (%)",
                    legend_title="Payment Method"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Payment insights
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    if 'Payment Method' in filtered_df.columns:
        top_payment = payment_counts.iloc[0]['Payment Method']
        
        st.markdown(f"""
        💡 **E-commerce Payment Preferences for Insurance Billing:**  
        **{top_payment}** is the preferred payment method in e-commerce, with significant age-based variations.
        
        **Insurance Application:** LifeSure can enhance customer experience by:
        • Offering multiple digital payment options to reduce paper waste
        • Implementing auto-pay systems with sustainability impact tracking
        • Creating age-specific payment method recommendations
        • Providing premium discounts for electronic billing and payments
        
        The diverse payment preferences seen in e-commerce should inform insurance billing strategies.
        """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # --- SECTION 7: SATISFACTION ANALYSIS --- #
    if 'Review Rating' in filtered_df.columns:
        st.markdown('<h2 class="section-title">Customer Satisfaction Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="chart-title">Customer Rating Distribution</h3>', unsafe_allow_html=True)
            
            # Round ratings to nearest 0.5 for better visualization
            filtered_df['Rounded Rating'] = round(filtered_df['Review Rating'] * 2) / 2
            rating_counts = filtered_df['Rounded Rating'].value_counts().sort_index().reset_index()
            rating_counts.columns = ['Rating', 'Count']
            
            fig = px.bar(
                rating_counts,
                x='Rating',
                y='Count',
                color='Rating',
                color_continuous_scale=colors['diverging'],
                text_auto=True
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=10, b=30),
                xaxis_title="Customer Rating",
                yaxis_title="Number of Customers",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<h3 class="chart-title">Average Rating by Product Category</h3>', unsafe_allow_html=True)
            
            if 'Category' in filtered_df.columns:
                category_ratings = filtered_df.groupby('Category')['Review Rating'].mean().sort_values(ascending=False).reset_index()
                category_ratings.columns = ['Category', 'Average Rating']
                
                fig = px.bar(
                    category_ratings,
                    x='Category',
                    y='Average Rating',
                    color='Average Rating',
                    color_continuous_scale=colors['diverging'],
                    text_auto='.2f'
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(
                    height=350,
                    margin=dict(l=10, r=10, t=10, b=30),
                    xaxis_title="",
                    yaxis_title="Average Rating",
                    yaxis=dict(range=[0, 5]),
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Satisfaction insights
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("""
        💡 **Customer Satisfaction Patterns for Insurance Products:**  
        E-commerce ratings show how product satisfaction varies by category and features.
        
        **Insurance Application:** LifeSure can improve customer satisfaction by:
        • Implementing transparent rating systems for policies and claims experience
        • Creating simplified policy documents and clear explanations
        • Offering sustainable options that align with customer values
        • Developing continuous feedback mechanisms for product improvement
        
        The satisfaction patterns in e-commerce highlight the importance of customer-centric insurance product design.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # --- SECTION 8: GEOGRAPHIC DISTRIBUTION & RISK MAPPING --- #
    st.markdown('<h2 class="section-title">Geographic Distribution Analysis</h2>', unsafe_allow_html=True)

    if 'Location' in filtered_df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="chart-title">Customer Distribution by Region</h3>', unsafe_allow_html=True)
            
            # Get top 10 locations - CHANGE THIS TO SHOW MORE LOCATIONS
            location_counts = filtered_df['Location'].value_counts().reset_index()
            location_counts.columns = ['Location', 'Count']
            location_counts = location_counts.sort_values('Count', ascending=False)
            
            # Increase from top 10 to top 25 locations
            top_n = 25  # Changed from 10 to 25
            
            if len(location_counts) > top_n:
                other_count = location_counts.iloc[top_n:]['Count'].sum()
                location_counts = location_counts.iloc[:top_n]
                if other_count > 0:
                    location_counts = pd.concat([
                        location_counts,
                        pd.DataFrame({'Location': ['Other'], 'Count': [other_count]})
                    ])
            
            # Adjust the chart to handle more locations
            fig = px.bar(
                location_counts,
                x='Count',
                y='Location',
                orientation='h',
                color='Count',
                color_continuous_scale=colors['sequential'],
                text_auto=True,
                height=600  # Increased height to accommodate more locations
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=30),
                xaxis_title="Number of Customers",
                yaxis_title="",
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Review Rating' in filtered_df.columns and 'Location' in filtered_df.columns:
                st.markdown('<h3 class="chart-title">Customer Satisfaction by Region</h3>', unsafe_allow_html=True)
                
                # Calculate average rating by location for locations with more than 5 customers
                location_counts_dict = filtered_df['Location'].value_counts().to_dict()
                valid_locations = [loc for loc, count in location_counts_dict.items() if count >= 5]
                
                if valid_locations:
                    location_df = filtered_df[filtered_df['Location'].isin(valid_locations)]
                    location_ratings = location_df.groupby('Location')['Review Rating'].mean().reset_index()
                    location_ratings = location_ratings.sort_values('Review Rating', ascending=False)
                    
                    # Show up to 25 regions instead of just 10 for consistency
                    top_n = 25  # Match the number in the adjacent chart
                    if len(location_ratings) > top_n:
                        location_ratings = location_ratings.head(top_n)
                    
                    fig = px.bar(
                        location_ratings,
                        x='Review Rating',
                        y='Location',
                        orientation='h',
                        color='Review Rating',
                        color_continuous_scale=colors['diverging'],
                        text_auto='.2f',
                        height=600  # Match the height of the adjacent chart
                    )
                    fig.update_traces(textposition='outside')
                    fig.update_layout(
                        margin=dict(l=10, r=10, t=10, b=30),
                        xaxis_title="Average Rating",
                        xaxis=dict(range=[0, 5]),
                        yaxis_title="",
                        yaxis=dict(autorange="reversed"),
                        coloraxis_showscale=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough data to show satisfaction by region.")

        # Geographic insights for insurance
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown(f"""
        💡 **Geographic Insights for Insurance Risk Assessment:**  
        The e-commerce data reveals significant geographic variation in customer distribution and satisfaction.
        
        **Insurance Application:** LifeSure can apply these insights by:
        • Developing region-specific insurance products that address local needs and risks
        • Implementing localized sustainability initiatives in high-customer-concentration areas
        • Prioritizing digital transformation efforts in regions with lower satisfaction scores
        • Creating community-based environmental programs in top customer locations
        
        Geographic patterns in consumer behavior can guide targeted insurance coverage options and risk assessment models.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- SECTION 9: SEASONAL TRENDS & RISK TIMING --- #
    if 'Season' in filtered_df.columns:
        st.markdown('<h2 class="section-title">Seasonal Purchasing Behavior</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="chart-title">Purchase Distribution by Season</h3>', unsafe_allow_html=True)
            
            season_counts = filtered_df['Season'].value_counts().reset_index()
            season_counts.columns = ['Season', 'Count']
            
            # Ensure proper season order
            season_order = ['Winter', 'Spring', 'Summer', 'Fall']
            season_counts['Season'] = pd.Categorical(season_counts['Season'], categories=season_order, ordered=True)
            season_counts = season_counts.sort_values('Season')
            
            fig = px.bar(
                season_counts,
                x='Season',
                y='Count',
                color='Season',
                text_auto=True,
                color_discrete_sequence=colors['categorical'][:4]  # Just 4 colors for seasons
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=10, b=30),
                xaxis_title="",
                yaxis_title="Number of Purchases"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Purchase Amount (USD)' in filtered_df.columns and 'Season' in filtered_df.columns:
                st.markdown('<h3 class="chart-title">Average Purchase Amount by Season</h3>', unsafe_allow_html=True)
                
                season_amount = filtered_df.groupby('Season')['Purchase Amount (USD)'].mean().reset_index()
                season_amount.columns = ['Season', 'Average Amount']
                
                # Ensure proper season order
                season_amount['Season'] = pd.Categorical(season_amount['Season'], categories=season_order, ordered=True)
                season_amount = season_amount.sort_values('Season')
                
                fig = px.line(
                    season_amount,
                    x='Season',
                    y='Average Amount',
                    markers=True,
                    text='Average Amount'
                )
                fig.update_traces(texttemplate='$%{text:.2f}', textposition='top center')
                fig.update_layout(
                    height=350,
                    margin=dict(l=10, r=10, t=10, b=30),
                    xaxis_title="",
                    yaxis_title="Average Purchase Amount (USD)"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal insights for insurance
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("""
        💡 **Seasonal Insights for Insurance Planning:**  
        E-commerce data shows significant seasonal variations in consumer spending patterns and preferences.
        
        **Insurance Application:** LifeSure can apply seasonal insights by:
        • Developing seasonal insurance offerings that align with changing customer needs
        • Creating flexible coverage options that can be adjusted quarterly
        • Implementing seasonal risk assessment models for more accurate pricing
        • Designing promotional campaigns aligned with natural customer spending cycles
        
        Seasonal patterns reveal how customer needs and behaviors change throughout the year, which can inform both 
        product development and sustainable business operations.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- SECTION 10: CUSTOMER LOYALTY ANALYSIS --- #
    if 'Previous Purchases' in filtered_df.columns:
        st.markdown('<h2 class="section-title">Customer Loyalty & Retention Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="chart-title">Customer Distribution by Previous Purchases</h3>', unsafe_allow_html=True)
            
            # Create buckets for previous purchases
            bins = [0, 5, 15, 30, 100]
            labels = ['New (0-5)', 'Returning (6-15)', 'Loyal (16-30)', 'VIP (31+)']
            filtered_df['Loyalty Segment'] = pd.cut(filtered_df['Previous Purchases'], bins=bins, labels=labels, right=False)
            
            loyalty_counts = filtered_df['Loyalty Segment'].value_counts().reset_index()
            loyalty_counts.columns = ['Loyalty Segment', 'Count']
            
            # Sort in meaningful order
            loyalty_counts['Loyalty Segment'] = pd.Categorical(
                loyalty_counts['Loyalty Segment'], 
                categories=labels,
                ordered=True
            )
            loyalty_counts = loyalty_counts.sort_values('Loyalty Segment')
            
            fig = px.pie(
                loyalty_counts,
                values='Count',
                names='Loyalty Segment',
                color='Loyalty Segment',
                color_discrete_sequence=colors['categorical'][:len(loyalty_counts)]
            )
            fig.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                marker=dict(line=dict(color='#FFFFFF', width=2))
            )
            fig.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=10, b=10),
                legend_title="Customer Loyalty"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Previous Purchases' in filtered_df.columns and 'Estimated_LTV' in filtered_df.columns:
                st.markdown('<h3 class="chart-title">Customer Value by Loyalty</h3>', unsafe_allow_html=True)
                
                loyalty_value = filtered_df.groupby('Loyalty Segment')['Estimated_LTV'].mean().reset_index()
                loyalty_value.columns = ['Loyalty Segment', 'Average LTV']
                
                # Sort in meaningful order
                loyalty_value['Loyalty Segment'] = pd.Categorical(
                    loyalty_value['Loyalty Segment'], 
                    categories=labels,
                    ordered=True
                )
                loyalty_value = loyalty_value.sort_values('Loyalty Segment')
                
                fig = px.bar(
                    loyalty_value,
                    x='Loyalty Segment',
                    y='Average LTV',
                    color='Average LTV',
                    color_continuous_scale='Greens',
                    text_auto='$.2f'
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(
                    height=350,
                    margin=dict(l=10, r=10, t=10, b=30),
                    xaxis_title="",
                    yaxis_title="Average Lifetime Value (USD)"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Calculate the LTV multiplier for VIP vs New customers
        if 'Loyalty Segment' in filtered_df.columns and 'Estimated_LTV' in filtered_df.columns:
            vip_ltv = filtered_df[filtered_df['Loyalty Segment'] == 'VIP (31+)']['Estimated_LTV'].mean()
            new_ltv = filtered_df[filtered_df['Loyalty Segment'] == 'New (0-5)']['Estimated_LTV'].mean()
            if new_ltv > 0:
                ltv_multiplier = vip_ltv / new_ltv
            else:
                ltv_multiplier = 0
        
        # Loyalty insights for insurance
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        if 'Loyalty Segment' in filtered_df.columns and 'Estimated_LTV' in filtered_df.columns:
            st.markdown(f"""
            💡 **Customer Loyalty Insights for Insurance Retention:**  
            The e-commerce data reveals that your most loyal customers (VIP) have a **{ltv_multiplier:.1f}x higher** lifetime value 
            than new customers.
            
            **Insurance Application:** LifeSure can enhance customer retention by:
            • Creating multi-year policy discounts that grow with customer tenure
            • Developing a loyalty program with sustainability-focused rewards
            • Implementing special services for long-term policyholders
            • Designing retention campaigns targeting customers approaching policy renewal milestones
            
            The significant value difference between new and loyal customers highlights the importance of retention-focused 
            insurance products and services.
            """)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- SECTION 11: DISCOUNT & PROMO IMPACT ANALYSIS --- #
    if 'Discount Applied' in filtered_df.columns and 'Promo Code Used' in filtered_df.columns:
        st.markdown('<h2 class="section-title">Discount & Promotion Impact Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="chart-title">Impact of Discounts on Purchase Amount</h3>', unsafe_allow_html=True)
            
            if 'Purchase Amount (USD)' in filtered_df.columns:
                discount_amount = filtered_df.groupby('Discount Applied')['Purchase Amount (USD)'].mean().reset_index()
                discount_amount.columns = ['Discount Applied', 'Average Purchase']
                
                fig = px.bar(
                    discount_amount,
                    x='Discount Applied',
                    y='Average Purchase',
                    color='Discount Applied',
                    text_auto='$.2f',
                    color_discrete_sequence=colors['binary']
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(
                    height=350,
                    margin=dict(l=10, r=10, t=10, b=30),
                    xaxis_title="Discount Applied",
                    yaxis_title="Average Purchase Amount (USD)"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<h3 class="chart-title">Effect of Discounts on Customer Satisfaction</h3>', unsafe_allow_html=True)
            
            if 'Review Rating' in filtered_df.columns:
                discount_rating = filtered_df.groupby(['Discount Applied', 'Promo Code Used'])['Review Rating'].mean().reset_index()
                discount_rating.columns = ['Discount Applied', 'Promo Code Used', 'Average Rating']
                
                fig = px.bar(
                    discount_rating,
                    x='Discount Applied',
                    y='Average Rating',
                    color='Promo Code Used',
                    barmode='group',
                    text_auto='.2f',
                    color_discrete_sequence=colors['binary']
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(
                    height=350,
                    margin=dict(l=10, r=10, t=10, b=30),
                    xaxis_title="Discount Applied",
                    yaxis_title="Average Rating",
                    yaxis=dict(range=[0, 5])
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Discount insights for insurance
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("""
        💡 **Discount Strategy Insights for Insurance Pricing:**  
        E-commerce data shows how discounts and promotions influence customer behavior and satisfaction.
        
        **Insurance Application:** LifeSure can optimize pricing strategies by:
        • Implementing transparent "green discounts" for environmentally conscious choices
        • Creating targeted promotional campaigns at key decision points
        • Developing bundled policy packages with clear value communication
        • Designing loyalty-based discount structures that increase with tenure
        
        The impact of pricing incentives in e-commerce demonstrates how strategic discounts can drive both 
        acquisition and retention in insurance products.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Sustainability implementation strategy
    st.markdown('<h2 class="section-title">Applying E-commerce Insights to Sustainable Insurance</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="policy-box">', unsafe_allow_html=True)
    st.markdown("""
    ### Three-Phase Implementation Plan for LifeSure
    
    **Phase 1: Digital Transformation** (Immediate)
    - Convert all policy documents to digital format based on e-commerce digital engagement patterns
    - Implement paperless claims processing with user-friendly interfaces
    - Launch mobile app with sustainable living tips and policy management features
    
    **Phase 2: Sustainable Product Development** (6-12 months)
    - Introduce green policy options with environmental benefits, targeting segments identified in the data
    - Develop usage-based insurance with sustainability incentives based on consumer purchase patterns
    - Create community environmental protection programs aligned with consumer values
    
    **Phase 3: Full Sustainability Integration** (12-24 months)
    - Build comprehensive carbon neutral operations, highlighting this in marketing
    - Implement transparent ESG investment options for all policies
    - Establish sustainability measurement and reporting framework for policy impact
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Data exploration
    with st.expander("Explore E-commerce Data Details"):
        st.dataframe(filtered_df.head(10), height=300)
    
    # Dashboard explanation with black background
    st.markdown("""
    <div class="about-box">
    <h3>About This Dashboard</h3>
    <p>This dashboard analyzes e-commerce customer behavior to provide insights for LifeSure Insurance's 
    sustainability initiatives. While the data comes from retail shopping patterns, the consumer preferences, 
    payment behaviors, and engagement patterns can inform insurance product development, digital transformation 
    strategies, and sustainability initiatives.</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Load the data
    df = load_data()
    
    if df is not None:
        # Prepare the data
        clean_df = prepare_data(df)
        
        # Create the dashboard (don't define colors here)
        create_dashboard(clean_df)

if __name__ == '__main__':
    main()
