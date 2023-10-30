from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the Iris dataset
iris = load_iris()
X_train, _, y_train, _ = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Train a simple Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    try:
        # Get the uploaded file
        file = request.files['file']

        # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(file, index_col=0)
        df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

        # Make predictions for each entry in the CSV
        predictions = []
        for index, row in df.iterrows():
            input_data = np.array(
                [row['sepal_length'], row['sepal_width'], row['petal_length'], row['petal_width']]).reshape(1, -1)
            prediction = model.predict(input_data)[0]
            predicted_class = iris.target_names[prediction]
            predictions.append(predicted_class)

        return render_template('result.html', data=list(zip(df.index, df['sepal_length'], df['sepal_width'], df['petal_length'], df['petal_width'], predictions)))

    except Exception as e:
        return render_template('error.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True)
