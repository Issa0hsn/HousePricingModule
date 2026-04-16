from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
df=pd.read_csv("cleaned_data_training.csv") # if it does not work you can put the full path  
x=df.drop('SalePrice', axis=1)  
y=df['SalePrice']  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(
    n_estimators=1500,
    learning_rate=0.05,
    max_depth=5,    
    tree_method='hist',
    device='cuda',           
)

print("Training started... 🌪️")
model.fit(x_train, y_train,verbose=False,eval_set=[(x_train,y_train)])
print("Training finished successfully!")

predictions = model.predict(x_test)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)
os.system("cls")
print("-" * 30)
print(f"✅ RMSE (Root Mean Squared Error): ${rmse:,.2f}")
print(f"🎯 R-squared (Accuracy): {r2 * 100:.2f}%")
importances = model.feature_importances_
feature_names = x.columns

feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)

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
