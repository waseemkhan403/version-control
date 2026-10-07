"""
Train a Decision Tree classifier on the Iris dataset. 

Demonstrates a minimal but complete training routine: 
* load data 
* split into train/test 
* fit the model    
* evaluate and report multiple metrics 
""" 

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, precision_score, recall_score   

def main():     
    # 1. Load the dataset     
    X, y = load_iris(return_X_y=True)       
    
    # 2. Hold out 20% of the data for testing     
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)       
    
    # 3. Build and train the model      
    model = DecisionTreeClassifier(max_depth=3, random_state=42)     
    model.fit(X_train, y_train)       
    
    # 4. Predict and evaluate     
    y_pred = model.predict(X_test)     
    acc = accuracy_score(y_test, y_pred)     
    prec = precision_score(y_test, y_pred, average="macro")     
    rec = recall_score(y_test, y_pred, average="macro")

    print(f"Accuracy : {acc:.2f}")     
    print(f"Precision: {prec:.2f}")     
    print(f"Recall   : {rec:.2f}")   

if __name__ == "__main__":     
    main() 