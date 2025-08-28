import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("🧵 Cloth Shop Sales Dashboard")

# File uploader
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)

    # Show data shape
    st.subheader("🔍 Dataset Preview")
    st.write("Shape:", df.shape)
    st.dataframe(df.head())

    # Info and describe
    st.subheader("📄 Dataset Info & Statistics")
    buffer = df.info(buf=None)
    st.text(df.info(verbose=True))
    st.dataframe(df.describe())

    # Convert 'date' column
    df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=False, errors='coerce')
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Clean missing values
    st.subheader("🧹 Missing Value Summary")
    st.write("Before filling:", df.isnull().sum())
    df.fillna(method='ffill', inplace=True)
    st.write("After forward fill:", df.isnull().sum())

    # 📈 Sales Trend - Mean of Last 30 Days
    st.subheader("📊 30-Day Sales Trend (Mean Aggregated)")
    daily_mean_sales = df.groupby(df['date'].dt.date)['sales'].mean().reset_index()
    daily_mean_sales = daily_mean_sales.sort_values('date').tail(30)

    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(daily_mean_sales['date'], daily_mean_sales['sales'], marker='o', label='Mean Daily Sales')
    ax1.set_title("30-Day Daily Sales Trend (Mean Aggregated)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Average Sales")
    ax1.grid(True)
    ax1.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    # 🎂 Age Distribution
    st.subheader("👥 Age Distribution of Customers")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Age'], kde=True, bins=30, color='skyblue', ax=ax2)
    ax2.set_title('Age Distribution')
    ax2.set_xlabel('Age')
    ax2.set_ylabel('Frequency')
    st.pyplot(fig2)


    # 📦 Boxplot to check Outliers in Age
    st.subheader("📦 Outliers in Age (Boxplot)")

    if 'Age' in df.columns:
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        if df['Age'].dropna().empty:
            st.warning("⚠️ 'Age' column has no numeric values to plot.")
        else:
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            sns.boxplot(x=df['Age'].dropna(), color='lightgreen', ax=ax3)
            ax3.set_title('Boxplot to check for Outliers in Age')
            ax3.set_xlabel('Age')
            st.pyplot(fig3)
    else:
        st.warning("⚠️ Column 'Age' not found in dataset.")


    # 👗 Fashion Influence Distribution
    st.subheader("👗 Fashion Influence Distribution")
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    sns.countplot(x='Fashion Influence', data=df, palette='Set2', ax=ax4)
    ax4.set_title('Fashion Distribution')
    ax4.set_xlabel('Fashion')
    ax4.set_ylabel('Count')
    st.pyplot(fig4)

    # 💰 Purchasing Power Distribution
    st.subheader("💰 Purchasing Power Distribution")
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    sns.countplot(x='Purchasing Power', data=df, palette='Set3', ax=ax5)
    ax5.set_title('Purchasing Power Distribution')
    ax5.set_xlabel('Purchasing Power')
    ax5.set_ylabel('Count')
    st.pyplot(fig5)

    # 👥 Age Distribution by Fashion Categories
    st.subheader("👥 Age Distribution across Fashion Categories")
    fig6, ax6 = plt.subplots(figsize=(8, 6))
    sns.boxplot(x='Fashion Influence', y='Age', data=df, palette='Set1', ax=ax6)
    ax6.set_title('Age Distribution across Fashion Categories')
    ax6.set_xlabel('Fashion')
    ax6.set_ylabel('Age')
    st.pyplot(fig6)

    # 🎯 Customer Segmentation
    st.subheader("🎯 Customer Segmentation")

    # Define segmentation labels
    def segment_label(row):
        if row['Cluster'] == 0:
            return "Price-Sensitive Young Fashionistas"
        elif row['Cluster'] == 1:
            return "Festival-Driven High Spenders"
        elif row['Cluster'] == 2:
            return "Comfort-First Traditional Buyers"
        else:
            return "Other"

    df['Segment'] = df.apply(segment_label, axis=1)

    # Segment vs Age (Boxplot)
    st.markdown("**📊 Age Distribution by Segment**")
    fig7, ax7 = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='Segment', y='Age', palette='Set2', ax=ax7)
    ax7.set_title("Age Distribution by Customer Segment")
    ax7.set_xticklabels(ax7.get_xticklabels(), rotation=20)
    st.pyplot(fig7)

    # Segment vs Gender (Countplot)
    st.markdown("**👫 Gender Distribution by Segment**")
    fig8, ax8 = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x='Segment', hue='Gender', ax=ax8)
    ax8.set_title("Gender Distribution by Segment")
    ax8.set_xticklabels(ax8.get_xticklabels(), rotation=15)
    st.pyplot(fig8)

        # 🎯 Segment Labeling
    st.subheader("📌 Customer Segment Labeling")

    def segment_label(row):
        if row['Cluster'] == 0:
            return "Price-Sensitive Young Fashionistas"
        elif row['Cluster'] == 1:
            return "Festival-Driven High Spenders"
        elif row['Cluster'] == 2:
            return "Comfort-First Traditional Buyers"
        elif row['Cluster'] == 3:
            return "Seasonal Trend Followers"
        else:
            return "Other"

    if 'Cluster' in df.columns:
        df['Segment'] = df.apply(segment_label, axis=1)
    else:
        st.warning("⚠️ 'Cluster' column not found. Segments cannot be labeled.")

    # 🧵 Material Preference by Segment with Color Palette Options
    st.subheader("🎨 Preferred Material Type by Segment")
    palette_choice = st.selectbox("Choose Color Palette", ['Paired', 'Set2', 'Pastel1'])

    if 'Segment' in df.columns and 'Material' in df.columns:
        fig_mat, ax_mat = plt.subplots(figsize=(12, 6))
        sns.countplot(data=df, x='Segment', hue='Material', palette=palette_choice, ax=ax_mat)
        ax_mat.set_title("Preferred Material Type by Segment", fontsize=14)
        ax_mat.set_xlabel("Customer Segment")
        ax_mat.set_ylabel("Count")
        ax_mat.legend(title="Material Type")
        plt.xticks(rotation=15)
        plt.tight_layout()
        st.pyplot(fig_mat)
    else:
        st.warning("⚠️ 'Segment' or 'Material' column not found for plotting.")

    # 🔍 Clustering & Elbow Method
    st.subheader("📊 K-Means Clustering (Elbow Method)")

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    # Select numeric columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[numerical_cols])

        # Sliders for Elbow Method Range
        k_min = st.slider("Minimum K", 1, 10, 1)
        k_max = st.slider("Maximum K", 10, 100, 20)

        if k_min < k_max:
            inertia = []
            for k in range(k_min, k_max + 1):
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(scaled_data)
                inertia.append(kmeans.inertia_)

            fig_elbow, ax_elbow = plt.subplots(figsize=(8, 6))
            ax_elbow.plot(range(k_min, k_max + 1), inertia, marker='o', color='blue')
            ax_elbow.set_title('Elbow Method for Optimal K')
            ax_elbow.set_xlabel('Number of Clusters')
            ax_elbow.set_ylabel('Inertia')
            st.pyplot(fig_elbow)
        else:
            st.warning("⚠️ Minimum K must be less than Maximum K.")
    else:
        st.warning("⚠️ No numeric columns available for clustering.")


# st_app_model.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
st.subheader("📅 Feature Engineering on Date Column")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['date'])
    st.success("File uploaded successfully!")
    

    # Ensure 'date' column is in datetime format
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
    else:
        st.warning("The 'date' column is not present in the dataset.")

    # Feature engineering
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    st.write("✅ Date Features Added: `day_of_week` & `month`")

    # Encode categorical features
    st.subheader("🔣 Label Encoding of Categorical Columns")
    label_cols = ['Avg_Spend', 'Family_Size', 'inventory', 'promotion',
                  'Fashion_Trend', 'Purchasing Power', 'Preferred_Clothing_Type']

    for col in label_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    
    st.write("✅ Categorical features encoded.")

    # Train/Test split
    st.subheader("🧪 Train/Test Split")
    features = ['Avg_Spend', 'Family_Size', 'inventory', 'promotion',
                'Fashion_Trend', 'Purchasing Power', 'Preferred_Clothing_Type',
                'day_of_week', 'month']
    X = df[features]
    y = df['sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    st.write("✅ Train/Test Split Completed")


# Impute missing values
    
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

# Train model
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)

# Predict
    y_pred = model_lr.predict(X_test)


    # Linear Regression
    st.subheader("📈 Linear Regression Model")
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    y_pred_lr = model_lr.predict(X_test)

    st.write("**R² Score:**", round(r2_score(y_test, y_pred_lr), 4))
    st.write("**RMSE:**", round(np.sqrt(mean_squared_error(y_test, y_pred_lr)), 4))

    # Random Forest
    st.subheader("🌲 Random Forest Model")
    model_rf = RandomForestRegressor(max_depth=2, min_samples_split=10, n_estimators=50, random_state=42)
    model_rf.fit(X_train, y_train)
    y_pred_rf = model_rf.predict(X_test)

    st.write("**MAE:**", round(mean_absolute_error(y_test, y_pred_rf), 4))

    # XGBoost
    st.subheader("🚀 XGBoost Model")
    model_xgb = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.01)
    model_xgb.fit(X_train, y_train)
    y_pred_xgb = model_xgb.predict(X_test)

    st.write("**XGB MSE:**", round(mean_squared_error(y_test, y_pred_xgb), 4))

else:
    st.warning("📎 Please upload a CSV file with 'date' and 'sales' columns to continue.")

st.subheader("📈 Future Sales Prediction (Next 10 Data Points)")
if 'X_test' in locals():
    future_data = X_test[:10]  # assuming it's a NumPy array
    future_sales = model_rf.predict(future_data)

    future_df = pd.DataFrame({
        'Index': range(len(future_sales)),
        'Predicted Sales': future_sales.round(2)
    })
    st.dataframe(future_df)
else:
    st.warning("X_test not available. Ensure model and train/test split are executed above.")

# Section: Scenario Analysis
st.subheader("🔮 Scenario Analysis: What-if Simulation")

def simulate_sales(income, season, online):
    base = income * 0.1
    season_multiplier = {"Summer": 1.2, "Winter": 0.9}.get(season, 1.0)
    online_multiplier = {"Strong": 1.5, "Moderate": 1.2}.get(online, 0.8)
    return base * season_multiplier * online_multiplier

income_input = st.number_input("Enter Monthly Income (₹)", min_value=1000, value=50000, step=1000)
season_input = st.selectbox("Select Season", ["Summer", "Winter", "Other"])
online_input = st.selectbox("Online Presence", ["Strong", "Moderate", "None"])

simulated_forecast = simulate_sales(income_input, season_input, online_input)
st.success(f"💰 Simulated Forecasted Sales = ₹{simulated_forecast:.2f}")

# Section: Forecasting for Next 7 Days
st.subheader("📆 Forecasting Next 7 Days")

features = ['day_of_week', 'month', 'promotion', 'inventory']
target = 'sales'

if all(col in df.columns for col in features + [target]):
    df_ml = df.dropna(subset=features + [target])
    X = df_ml[features]
    y = df_ml[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    y_pred = model_rf.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f"📉 Validation MAE: `{mae:.2f}` sales units")

    # Future prediction
    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)
    last_inventory = df['inventory'].iloc[-1]
    last_promotion = df['promotion'].iloc[-1]
    
    X_future = pd.DataFrame({
            'day_of_week': [d.dayofweek for d in future_dates],
            'month': [d.month for d in future_dates],
            'promotion': [last_promotion] * 7,
            'inventory': [last_inventory] * 7
        }   )
    
    sales_forecast = model_rf.predict(X_future)
    
    markup = st.slider('📊 Markup (%)', min_value=10, max_value=100, value=35, step=5)
    profit_forecast = sales_forecast * markup / 100

    forecast_df = pd.DataFrame({
         'Date': future_dates.strftime('%Y-%m-%d'),
            'Predicted Sales': sales_forecast.round(2),
            'Predicted Profit': profit_forecast.round(2)
        }   )

    st.dataframe(forecast_df)
else:
        st.error("Required columns not found in dataframe. Please check the column names and data.")

    # Upload data
st.title("Sales & Profit Dashboard")

# Load your existing dataframe
# Ensure your 'date' column is in datetime format
df['date'] = pd.to_datetime(df['date'])

# Display key stats
avg_sales = df['sales'].mean()
total_sales = df['sales'].sum()
avg_inventory = df['inventory'].mean()

st.metric("Average Daily Sales", f"{avg_sales:.2f}")
st.metric("Total Sales", f"{total_sales:.0f}")
st.metric("Average Inventory Level", f"{avg_inventory:.0f}")

# Sidebar slider for markup
markup = st.sidebar.slider("Markup (%)", min_value=10, max_value=100, value=40, step=5)

# Calculate and plot sales vs profit
df['profit'] = df['sales'] * markup / 100

fig1 = px.bar(df, x='date', y=['sales', 'profit'], barmode='group',
                labels={'value': 'Amount', 'date': 'Date'}, title=f"Daily Sales and Profit (Markup = {markup}%)")
st.plotly_chart(fig1, use_container_width=True)

# Line graph aggregation
st.subheader("Resampled Sales & Profit (Line Graph)")

resample_option = st.selectbox(
        "Choose Resampling Interval",
        options=[("7 Days", "7D"), ("15 Days", "15D"), ("30 Days", "30D")],
        index=2
    )

# Grouping and aggregating
df.set_index('date', inplace=True)
resampled_df = df[['sales', 'profit']].resample(resample_option[1]).mean().reset_index()

fig2 = px.line(resampled_df, x='date', y=['sales', 'profit'], title=f"Aggregated Sales & Profit Every {resample_option[0]}")
st.plotly_chart(fig2, use_container_width=True)


#st.set_page_config(layout="wide")
st.title("📈 Local Cloth Shop Sales Dashboard")

# Simulate or read CSV
@st.cache_data
def load_data():
    np.random.seed(42)
    sales = np.random.normal(20000, 3000, 36) + np.linspace(1000, 6000, 36)
    dates = pd.date_range(start='2022-01-01', periods=36, freq='M')
    df = pd.DataFrame({'date': dates, 'sales': sales})
    df['inventory'] = np.random.randint(5000, 10000, size=len(df))
    df['promotion'] = np.random.choice([0, 1], size=len(df))
    return df

df = load_data()
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# -------------------------
# KPIs Section
# -------------------------
st.subheader("📊 Sales Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Avg Daily Sales", f"₹{df['sales'].mean():.2f}")
col2.metric("Total Sales", f"₹{df['sales'].sum():.0f}")
col3.metric("Avg Inventory", f"{df['inventory'].mean():.0f} units")

# -------------------------
# Interactive Profit Graph
# -------------------------
st.subheader("💰 Interactive Profit Simulation")
markup = st.slider("Select Markup (%)", min_value=10, max_value=100, step=5, value=40)
resample_freq = st.selectbox("Select Resampling Frequency", ['7D', '15D', '30D'])

df['profit'] = df['sales'] * markup / 100
df_agg = df.resample(resample_freq).mean()

fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(df_agg.index, df_agg['profit'], marker='o', color='green', label='Profit')
ax1.set_title(f"{resample_freq} Avg Profit with {markup}% Markup", fontsize=16)
ax1.set_ylabel("Profit (₹)")
ax1.grid(True)
ax1.legend()
st.pyplot(fig1)

# -------------------------
# Forecasting
# -------------------------
st.subheader("🔮 Sales Forecast (ARIMA & Exponential Smoothing)")
future_steps = st.slider("Forecast Months Ahead", 1, 12, 6)

ts = df['sales']

# ARIMA
arima_model = ARIMA(ts, order=(1, 1, 1)).fit()
arima_forecast = arima_model.forecast(steps=future_steps)

# Exp Smoothing
es_model = ExponentialSmoothing(ts, trend='add', seasonal=None).fit()
es_forecast = es_model.forecast(future_steps)

# Forecast Plot
fig2, ax2 = plt.subplots(figsize=(12, 4))
ts.plot(label="Actual", ax=ax2)
arima_forecast.plot(label="ARIMA Forecast", ax=ax2)
es_forecast.plot(label="Exponential Smoothing", ax=ax2)
ax2.set_title(f"{future_steps}-Month Sales Forecast")
ax2.set_ylabel("Sales (₹)")
ax2.legend()
st.pyplot(fig2)

# -------------------------
# Final Insights
# -------------------------
st.subheader("🧠 Growth Insights for Local Shops")
st.markdown("""
- 📢 **Use promotions wisely**: Boosts visibility and increases sales.
- 📦 **Maintain healthy inventory**: Avoid stockouts and missed revenue.
- 🎯 **Track customer demand trends**: Focus on seasonal/clothing patterns.
- 🌐 **Strengthen online presence**: Loyalty and repeat customers = higher profit.
""")