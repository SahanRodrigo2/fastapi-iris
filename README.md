Iris Flower Classification
Problem Description
The task is to classify iris flowers into three species: (Setosa, Versicolor, Virginica)  
The model takes four input features:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

This is a multi-class classification problem.

Model Choice
We used Logistic Regression with StandardScaler inside a Scikit-Learn pipeline:
- Simple and interpretable
- Works well on small datasets like Iris
- Achieved 96% accuracy on the test set

The trained model is saved as “model.pkl”.




How to Run the Application
Clone the repository:
bash
git clone https://github.com/SahanRodrigo2/fastapi-iris.git
cd fastapi-iris




