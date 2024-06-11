import torch

# Define your neural network model class
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
model.load_state_dict(torch.load('C:/Users/jerem/OneDrive/Desktop/Final Year Project - Water Analysis/Final Project/Water_quality.h5'))
model.eval()

# Sample input data for prediction
sample_input = torch.tensor([[7.0, 0.2, 10.0, 20.0, 0.1, 30.0, 50.0, 0.4, 25.0, 15.0, 1.5, 0.03, 300.0]], dtype=torch.float32)

# Make prediction
with torch.no_grad():
    output = model(sample_input)
    predicted_label = torch.sigmoid(output) > 0.5
    if predicted_label.item() == 0:
        prediction = 'NO'
    elif predicted_label.item() == 1:
        prediction = 'YES'

# Print the prediction
print("Prediction:", prediction)
print("jeremy")