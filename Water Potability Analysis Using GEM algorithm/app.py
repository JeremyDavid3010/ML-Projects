from flask import Flask, render_template, request
import torch

app = Flask(__name__)

# Load your trained model and define preprocessing functions here
# Example:
class WaterQualityClassifierImproved(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(WaterQualityClassifierImproved, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size1)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(hidden_size2, hidden_size3)
        self.relu3 = torch.nn.ReLU()
        self.fc4 = torch.nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

# Load the trained model
model = WaterQualityClassifierImproved(input_size=13, hidden_size1=64, hidden_size2=32, hidden_size3=16, output_size=1)
model.load_state_dict(torch.load('Water_quality.h5'))
model.eval()

# Function to make predictions
def predict_output(input_data):
    with torch.no_grad():
        inputs = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
        outputs = model(inputs)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        if predicted.item() == 0:
            return 'NO, You cannot Drink this Water !'
        elif predicted.item() == 1:
            return 'YES, You can Consume this Water !'

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    if request.method == 'POST':
        # Get the input values from the form
        input_data = [float(request.form.get(feat)) for feat in ['ph', 'iron', 'nitrate', 'chloride', 'lead', 'zinc', 'turbidity', 'fluoride', 'copper', 'sulfate', 'chlorine', 'manganese', 'total_dissolved_solids']]
        # Make prediction
        prediction = predict_output(input_data)
        return render_template('result2.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)