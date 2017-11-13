# QuestionAnsweringSystem
A system for automatically answering questions. This project is part of CS5224 class and works on the dataset provided in the Kaggle competition for the class. 

# Getting started
1. Start by downloading the data and copying it in the raw_data/ directory. 
2. Run get_started.sh, which will preprocess the data and download the required GloVe files
3. Run qa_data to create word vector representation for the dataset
4. Run train.py, pass it the directory to the prepocessed data files and GloVe. 
5. The model will be trained and save to model.h5
6. For testing the model on a seperate dataset, load the model using Keras and evaluate it on the dataset. 


