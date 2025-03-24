import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="LifeSure Sustainability Insights",
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
    # Main title
    st.markdown('<h1 class="main-title">🌿 LifeSure Customer Insights Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This dashboard analyzes customer behavior patterns to help LifeSure Insurance design modern, 
    sustainable insurance policies. The visualizations reveal key insights about customer preferences, 
    spending patterns, and engagement that can inform your sustainability initiatives.
    """)
    
    # Sidebar filters with explanations
    st.sidebar.markdown('## 🔍 Explore Customer Segments')
    st.sidebar.markdown('Use these filters to focus on specific customer segments:')
    
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
    st.markdown('<h2 class="section-title">Key Customer Insights</h2>', unsafe_allow_html=True)
    
    # Key metrics with explanations
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{filtered_df.shape[0]:,}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Customers</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if 'Purchase Amount (USD)' in filtered_df.columns:
            avg_purchase = filtered_df['Purchase Amount (USD)'].mean()
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">${avg_purchase:.2f}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Avg. Purchase Amount</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        if 'Estimated_LTV' in filtered_df.columns:
            avg_ltv = filtered_df['Estimated_LTV'].mean()
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">${avg_ltv:.2f}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Avg. Customer Lifetime Value</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        if 'Review Rating' in filtered_df.columns:
            avg_rating = filtered_df['Review Rating'].mean()
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{avg_rating:.1f}/5.0</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Avg. Customer Rating</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Insurance company context explanation
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("""
    💡 **Insurance Sustainability Implication:**  
    Understanding customer lifetime value and satisfaction helps design sustainable policies with appropriate pricing
    and benefits. Customers who show loyalty and high satisfaction are more likely to adopt green insurance options.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # --- SECTION 2: CUSTOMER DEMOGRAPHICS --- #
    st.markdown('<h2 class="section-title">Customer Demographics</h2>', unsafe_allow_html=True)
    
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
                color_discrete_sequence=px.colors.sequential.Greens,
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'Age Group' in filtered_df.columns:
            st.markdown('<h3 class="chart-title">Age Distribution</h3>', unsafe_allow_html=True)
            
            age_counts = filtered_df['Age Group'].value_counts().reset_index()
            age_counts.columns = ['Age Group', 'Count']
            
            # Ensure age groups are in logical order
            order = ['18-24', '25-39', '40-59', '60+']
            age_counts['Age Group'] = pd.Categorical(age_counts['Age Group'], categories=order, ordered=True)
            age_counts = age_counts.sort_values('Age Group')
            
            fig = px.bar(
                age_counts,
                x='Age Group',
                y='Count',
                color='Age Group',
                text_auto=True
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="",
                yaxis_title="Number of Customers",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Demographics insights
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("""
    💡 **Demographic Insights for Sustainable Insurance:**  
    Different age groups have varying sustainability priorities. Younger customers (18-39) typically value 
    digital experiences and environmental impact, while older customers (40+) may prioritize long-term stability 
    and community benefits. Design policy options that address these different sustainability priorities.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # --- SECTION 3: SUBSCRIPTION ANALYSIS --- #
    if 'Subscription Status' in filtered_df.columns and 'Estimated_LTV' in filtered_df.columns:
        st.markdown('<h2 class="section-title">Subscription Impact Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="chart-title">Subscription Status Distribution</h3>', unsafe_allow_html=True)
            
            sub_counts = filtered_df['Subscription Status'].value_counts().reset_index()
            sub_counts.columns = ['Subscription Status', 'Count']
            
            fig = px.pie(
                sub_counts,
                values='Count',
                names='Subscription Status',
                color_discrete_sequence=['#43A047', '#C8E6C9']
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<h3 class="chart-title">Average LTV by Subscription Status</h3>', unsafe_allow_html=True)
            
            sub_ltv = filtered_df.groupby('Subscription Status')['Estimated_LTV'].mean().reset_index()
            
            fig = px.bar(
                sub_ltv,
                x='Subscription Status',
                y='Estimated_LTV',
                color='Subscription Status',
                text_auto='.2f',
                color_discrete_sequence=['#43A047', '#C8E6C9']
            )
            fig.update_traces(texttemplate='$%{text}', textposition='outside')
            fig.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="",
                yaxis_title="Average Customer Lifetime Value ($)",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Calculate the LTV difference
        if len(sub_ltv) == 2:
            yes_ltv = sub_ltv[sub_ltv['Subscription Status'] == 'Yes']['Estimated_LTV'].values[0]
            no_ltv = sub_ltv[sub_ltv['Subscription Status'] == 'No']['Estimated_LTV'].values[0]
            ltv_diff_pct = ((yes_ltv - no_ltv) / no_ltv) * 100 if no_ltv > 0 else 0
        
            # Subscription insights
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown(f"""
            💡 **Subscription Model Insights:**  
            Subscribers have a **{ltv_diff_pct:.1f}%** higher lifetime value than non-subscribers. 
            
            For sustainable insurance policies, consider a subscription-based model that provides ongoing coverage 
            with automatic renewals. This encourages long-term relationships and enables continuous improvement of 
            sustainable features based on customer feedback and changing environmental needs.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # --- SECTION 4: PRODUCT PREFERENCES --- #
    if 'Category' in filtered_df.columns:
        st.markdown('<h2 class="section-title">Product Category Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="chart-title">Popular Product Categories</h3>', unsafe_allow_html=True)
            
            category_counts = filtered_df['Category'].value_counts().reset_index()
            category_counts.columns = ['Category', 'Count']
            
            # Sort by count
            category_counts = category_counts.sort_values('Count', ascending=False)
            
            fig = px.bar(
                category_counts,
                x='Category',
                y='Count',
                color='Category',
                text_auto=True
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(
                height=400,
                margin=dict(l=10, r=10, t=10, b=30),
                xaxis_title="",
                yaxis_title="Number of Purchases",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<h3 class="chart-title">Average Purchase Amount by Category</h3>', unsafe_allow_html=True)
            
            if 'Purchase Amount (USD)' in filtered_df.columns:
                category_spend = filtered_df.groupby('Category')['Purchase Amount (USD)'].mean().reset_index()
                category_spend = category_spend.sort_values('Purchase Amount (USD)', ascending=False)
                
                fig = px.bar(
                    category_spend,
                    x='Category',
                    y='Purchase Amount (USD)',
                    color='Category',
                    text_auto='.2f'
                )
                fig.update_traces(texttemplate='$%{text}', textposition='outside')
                fig.update_layout(
                    height=400,
                    margin=dict(l=10, r=10, t=10, b=30),
                    xaxis_title="",
                    yaxis_title="Average Purchase Amount ($)",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Category insights
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        top_category = category_counts.iloc[0]['Category']
        highest_spend = category_spend.iloc[0]['Category']
        
        st.markdown(f"""
        💡 **Product Category Insights for Insurance:**  
        The most popular category is **{top_category}**, while **{highest_spend}** has the highest average spending.
        
        For sustainable insurance design, consider offering a tiered approach:
        • **Basic Coverage** - Essential protection with eco-friendly options
        • **Add-On Modules** - Flexible supplementary coverage aligned with customer values
        • **Premium Protection** - Comprehensive coverage with enhanced sustainability features
        
        This mimics retail purchasing behavior where customers shop across different categories based on their needs.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # --- SECTION 5: GEOGRAPHIC DISTRIBUTION --- #
    if 'Location' in filtered_df.columns:
        st.markdown('<h2 class="section-title">Geographic Distribution</h2>', unsafe_allow_html=True)
        
        top_locations = filtered_df['Location'].value_counts().nlargest(10).reset_index()
        top_locations.columns = ['Location', 'Number of Customers']
        
        fig = px.bar(
            top_locations,
            x='Number of Customers',
            y='Location',
            orientation='h',
            color='Number of Customers',
            color_continuous_scale='Greens',
            text='Number of Customers'
        )
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_layout(
            height=500,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Number of Customers",
            yaxis_title="",
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Location insights
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        top_loc = top_locations.iloc[0]['Location']
        top_loc_pct = (top_locations.iloc[0]['Number of Customers'] / filtered_df.shape[0]) * 100
        
        st.markdown(f"""
        💡 **Geographic Insights for Sustainable Insurance:**  
        Your top customer location is **{top_loc}** with {top_locations.iloc[0]['Number of Customers']} customers 
        ({top_loc_pct:.1f}% of your customer base).
        
        Consider region-specific sustainable insurance policies that address local environmental challenges.
        For example, wildfire coverage in western states, flood protection in coastal areas, or renewable
        energy incentives in states with strong solar/wind potential.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # --- SECTION 6: PAYMENT PREFERENCES --- #
    st.markdown('<h2 class="section-title">Payment Method Analysis</h2>', unsafe_allow_html=True)
    
    if 'Payment Method' in filtered_df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="chart-title">Popular Payment Methods</h3>', unsafe_allow_html=True)
            
            payment_counts = filtered_df['Payment Method'].value_counts().reset_index()
            payment_counts.columns = ['Payment Method', 'Count']
            
            # Sort by count
            payment_counts = payment_counts.sort_values('Count', ascending=False)
            
            # Only show top 5 payment methods if there are many
            if len(payment_counts) > 5:
                other_count = payment_counts.iloc[5:]['Count'].sum()
                payment_counts = payment_counts.iloc[:5]
                payment_counts = pd.concat([
                    payment_counts,
                    pd.DataFrame({'Payment Method': ['Other'], 'Count': [other_count]})
                ])
            
            fig = px.pie(
                payment_counts,
                values='Count',
                names='Payment Method',
                color_discrete_sequence=px.colors.sequential.Greens
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                height=400,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<h3 class="chart-title">Payment Methods by Age Group</h3>', unsafe_allow_html=True)
            
            if 'Age Group' in filtered_df.columns:
                # Get top payment methods
                top_payments = payment_counts['Payment Method'].iloc[:4].tolist()
                if 'Other' in top_payments:
                    top_payments.remove('Other')
                
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
                    text_auto='.1f'
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
        💡 **Payment Insights for Sustainable Insurance:**  
        **{top_payment}** is your customers' preferred payment method, but preferences vary significantly by age group.
        
        For sustainable insurance operations:
        • Offer digital payment options to reduce paper waste
        • Implement auto-pay systems with transparent sustainability impact metrics
        • Consider micro-insurance payments that align with customer cash flow
        • Provide premium discounts for electronic billing and payments
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
                color_continuous_scale='Greens',
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
                    color_continuous_scale='Greens',
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
    
    # Sustainability implementation strategy
    st.markdown('<h2 class="section-title">Sustainability Strategy for LifeSure Insurance</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="policy-box">', unsafe_allow_html=True)
    st.markdown("""
    ### Three-Phase Implementation Plan
    
    **Phase 1: Digital Transformation** (Immediate)
    - Convert all policy documents to digital format
    - Implement paperless claims processing
    - Launch mobile app with sustainable living tips
    
    **Phase 2: Sustainable Product Development** (6-12 months)
    - Introduce green policy options with environmental benefits
    - Develop usage-based insurance with sustainability incentives
    - Create community environmental protection programs
    
    **Phase 3: Full Sustainability Integration** (12-24 months)
    - Build comprehensive carbon neutral operations
    - Implement transparent ESG investment options for all policies
    - Establish sustainability measurement and reporting for policy impact
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Data exploration
    with st.expander("Explore Data Details"):
        st.dataframe(filtered_df.head(10), height=300)
    
    # Dashboard explanation
    st.markdown("""
    <div class="insight-box">
    <h3>About This Dashboard</h3>
    <p>This dashboard provides insights from customer behavior data to help LifeSure Insurance develop and 
    implement sustainable insurance policies. These visualizations can inform product design, customer 
    engagement strategies, and digital transformation initiatives that align with sustainability goals.</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Load the data
    df = load_data()
    
    if df is not None:
        # Prepare the data
        clean_df = prepare_data(df)
        
        # Create the dashboard
        create_dashboard(clean_df)

if __name__ == '__main__':
    main()
