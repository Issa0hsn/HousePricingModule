from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

#Load the data 
df=pd.read_csv("cleaned_data.csv") # if it does not work you can put the full path 

#Selecting the features and target column 
x=df.drop('SalePrice', axis=1)  
y=df['SalePrice'] 

# splitting the data into training and testing sets (80, 20)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 3. initializing the XGBoost regressor with GPU support
model = xgb.XGBRegressor(
    n_estimators=1500,#trees in the forest
    learning_rate=0.05,
    max_depth=5,    
    tree_method='hist',# algorithm for building trees (histogram-based, efficient for large datasets)
    device='cuda',           # use GPU for training    
)
# 4. you maybe hear a fan noise, it's the GPU working hard! 

# training the model (this may take a few seconds to a couple of minutes depending on your GPU)
print("Training started... 🌪️")
model.fit(x_train, y_train,verbose=False,eval_set=[(x_train,y_train)])
print("Training finished successfully!")

# 5. testing the model (making predictions)
predictions = model.predict(x_test)

# 6. Evaluating the model's performance
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)
os.system("cls")
print("-" * 30)
print(f"✅ RMSE (Root Mean Squared Error): ${rmse:,.2f}")
print(f"🎯 R-squared (Accuracy): {r2 * 100:.2f}%")

# 1. sorting features by importance (gain) and selecting the top 10
importances = model.feature_importances_
feature_names = x.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)

# 3. visualizing the top 10 features influencing house prices
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='magma')
plt.title('Top 10 Features Influencing House Prices')
plt.xlabel('Importance Score (Gain)')
plt.ylabel('Feature Name')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()
residuals = y_test - predictions
plt.figure(figsize=(10, 5))
sns.scatterplot(x=predictions, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals Plot: Are we missing something?')
plt.xlabel('Predicted Prices')
plt.ylabel('Error (Actual - Predicted)')
plt.show()
