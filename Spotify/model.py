from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import rand
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Initialize Spark session with MongoDB support
spark = SparkSession \
    .builder \
    .appName("Music Recommendation System") \
    .config("spark.mongodb.input.uri", "mongodb+srv://user:1234@cluster0.sserygh.mongodb.net/Spotify_Recommendation_System.Meta_Data") \
    .config("spark.mongodb.output.uri", "mongodb+srv://user:1234@cluster0.sserygh.mongodb.net/Spotify_Recommendation_System.Model") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
    .getOrCreate()

# Load data from MongoDB
df = spark.read.format("mongo").load()
df = df.selectExpr("cast(track_id as int) as track_id", "cast(folder as int) as folder")
df_with_ratings = df.withColumn("rating", (rand() * 4 + 1))  # Random ratings between 1 and 5

# ALS model for collaborative filtering
als = ALS(maxIter=10, regParam=0.01, userCol="folder", itemCol="track_id", ratingCol="rating", coldStartStrategy="drop")
model = als.fit(df_with_ratings)
predictions = model.transform(df_with_ratings)

# Evaluate the model
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = ", rmse)

# Save the ALS model to the local file system
als_model_path = "/path/to/save/als_model"  # Specify the path where you want to save the ALS model
model.write().overwrite().save(als_model_path)
print("ALS model saved to:", als_model_path)

# Prepare data for PyTorch (example with random data for demonstration)
features = torch.randn(100, 15)  # Simulated feature vectors
labels = torch.randint(0, 2, (100, 1)).float()  # Simulated binary labels

# PyTorch dataset and dataloader
class MusicDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

dataset = MusicDataset(features, labels)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Define and train the PyTorch model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.linear1 = nn.Linear(15, 64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return self.sigmoid(x)

model = SimpleNN()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop for the PyTorch model
for epoch in range(5):  # Reduced number of epochs for quick demonstration
    for data, target in dataloader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Save the PyTorch model to the local file system
torch_model_path = "file:///home/maemoon/Documents/BDA_Project/torch_model.pth" 
 # Specify the path where you want to save the PyTorch model

torch.save(model.state_dict(), torch_model_path)
print("PyTorch model saved to:", torch_model_path)

# Clean up Spark session
spark.stop()
