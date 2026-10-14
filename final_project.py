
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files


uploaded = files.upload()
df = pd.read_csv("train.csv")


print("First 5 rows:")
print(df.head())

# 2. Dataset information
print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nStatistics:")
print(df.describe())

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)

# 4. Survival count
sns.countplot(data=df, x="Survived")
plt.title("Titanic Survival Count")
plt.show()


sns.countplot(data=df, x="Sex", hue="Survived")
plt.title("Survival by Gender")
plt.show()


sns.countplot(data=df, x="Pclass", hue="Survived")
plt.title("Survival by Passenger Class")
plt.show()


sns.histplot(data=df, x="Age", bins=20, kde=True)
plt.title("Passenger Age Distribution")
plt.show()


print("\nSurvival Rate:")
print(df["Survived"].mean() * 100, "%")


sns.heatmap(
    df[["Survived", "Pclass", "Age", "SibSp",
        "Parch", "Fare"]].corr(),
    annot=True, cmap="coolwarm"
)
plt.title("Feature Correlation")
plt.show()


df.to_csv("cleaned_titanic.csv", index=False)
print("Cleaned dataset saved!")
