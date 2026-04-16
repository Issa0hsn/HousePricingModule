from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
df=pd.read_csv("C:\\Users\\ALAMEEN\\PythonWork\\MlProject\\cleaned__data_training.csv")  # تأكد إن المسار صحيح
# 1. فصل الهدف (السعر) عن باقي الميزات (المدخلات)
# تأكد إن اسم عمود السعر عندك هو 'SalePrice'
x=df.drop('SalePrice', axis=1)  # كل الأعمدة ما عدا السعر
y=df['SalePrice']  # عمود السعر فقط

# 2. تقسيم البيانات: 80% لتدريب الموديل، و 20% لاختباره
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 3. إعداد الـ XGBoost وتشغيل كرت الشاشة 🚀
model = xgb.XGBRegressor(
    n_estimators=5000,       # عدد الأشجار (غابة ضخمة)
    learning_rate=0.05,      # سرعة التعلم (خطوات ثابتة ومدروسة)
    max_depth=5,            # عمق كل شجرة 
    
    tree_method='hist',      # خوارزمية بناء الأشجار (الأسرع حالياً)
    device='cuda',           # السحر هون: توجيه الأمر لكرت الشاشة!
    # ملاحظة: إذا نسختك قديمة من XGBoost وعطاك خطأ على السطر اللي فوق،
    # احذفه واستخدم هاد السطر بداله: tree_method='gpu_hist'
)

# 4. بدء التدريب (هون رح تسمع صوت مراوح اللابتوب!)
print("Training started... 🌪️")
model.fit(x_train, y_train,verbose=False,eval_set=[(x_train,y_train)])
print("Training finished successfully!")
#print(f"best iteration : {model.best_iteration}")

# 5. التوقع على بيانات الاختبار (الـ 20% اللي ما شافها الموديل)
predictions = model.predict(x_test)

# 6. التقييم الرياضي (امتحان الموديل)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)
os.system("cls")
print("-" * 30)
print(f"✅ RMSE (Root Mean Squared Error): ${rmse:,.2f}")
print(f"🎯 R-squared (Accuracy): {r2 * 100:.2f}%")
# 1. الحصول على أهمية الميزات
importances = model.feature_importances_
feature_names = x.columns

# 2. ترتيبهم وأخذ أول 10
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)

# 3. الرسم
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