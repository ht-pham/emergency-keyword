
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import torch
from torch.utils.data import DataLoader

from cnn import CNN
from cnn_lstm import CNN_LSTM
from dataset import MFCCDataset
from tflite_conversion import convert_pytorch_to_tflite

def load_data():
    X = np.load('./../../data/processed/labels/X.npy')
    y = np.load('./../../data/processed/labels/y.npy')
    return X,y

def split_data(X, y, test_size=0.2, random_state=42):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,stratify=y)
    print(f"Training set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test

def normalize(X):
    # Normalization before converting to tensor
    return (X-X.mean())/(X.std()+1e-6)

def toTensor(X,y):
    # Combine normalized X and y into one Dataset object
    dataset = MFCCDataset(X=X,y=y)

    # use DataLoader -- special util for batching tensors
    dataloader = DataLoader(dataset,batch_size=16,shuffle=True)

    return dataloader

def load_optimizers(model):
    import torch
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005)
    if str(model) == 'CNN_LSTM':
        weights = torch.tensor([1.4,1.2,1.2]) # best: [1.2,1.5,1.0]
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    # print(optimizer)
    # print(criterion)
    return optimizer, criterion

def build(model,dataloader,epochs=101):
    opt, criterion = load_optimizers(model)
    model.train()
    for epoch in range(epochs):
        for X_batch,y_batch in dataloader:
            predictions = model(X_batch)
            loss = criterion(predictions,y_batch)

            opt.zero_grad()
            loss.backward()
            opt.step()

    print(f"Finished training {model}")

def evaluate(model,testloader):
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for inputs,labels in testloader:
            outputs = model(inputs)
            _,predicted = torch.max(outputs,1)

            print(f"Pred: {predicted[:10]}")
            print(f"True: {labels[:10]}")

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    
    print("Classification Report")
    print(classification_report(all_labels,all_preds))
    print("=====================")
    print("Confusion Matrix")
    print(confusion_matrix(all_labels,all_preds))


               
if __name__ == "__main__":
    #======== STEP 1 - LOAD & PROCESS DATASET ========
    #-----------  (1) load data   --------------------
    X,y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    #-----------  (2) normalize data   ---------------
    X_train = normalize(X=X_train)
    X_test = normalize(X=X_test)
    #-----------  (3) convert to Tensor   ------------
    dataloader = toTensor(X=X_train,y=y_train)
    testloader = toTensor(X=X_test,y=y_test)

    #======== STEP 2 - LOAD MODELS ===================
    #-----------  (1) load models   ------------------
    base_model = CNN()
    lstm_model = CNN_LSTM()
    #-----------  (2) load parameters   --------------
    cnn_opt, cnn_criterion = load_optimizers(base_model)
    lstm_opt, lstm_criterion = load_optimizers(lstm_model)
    
    #======== STEP 3 - TRAIN & EVALUATE MODELS =======
    #----------- (1) Train models --------------------
    ## train the models
    build(base_model,dataloader,80)
    build(lstm_model,dataloader,150)

    #----------- (2) Evaluate models -----------------
    #----------- Baseline Model: CNN -----------------
    print("\n\t\tMODEL EVALUATION\n\n")
    print("=====Training set=======")
    evaluate(base_model,dataloader)
    print("\n\t\tMODEL EVALUATION\n\n")
    print("======Testing set=======")
    evaluate(base_model,testloader)

    #----------- Model: CNN + LSTM  ------------------
    print("\n\t\tMODEL EVALUATION\n\n")
    print("=====Training set=======")
    evaluate(lstm_model,dataloader)
    print("\n\t\tMODEL EVALUATION\n\n")
    print("======Testing set=======")
    evaluate(lstm_model,testloader)

    #======== STEP 4 - CONVERT MODELS TO TFLITE =====
    convert_pytorch_to_tflite(base_model,"cnn",(1,12,298))
    convert_pytorch_to_tflite(lstm_model,"cnn_lstm",(1,12,298))
    

    



