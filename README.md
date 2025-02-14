"# chess_model" 
```docker run --name my-redis -p 6379:6379 -d redis```
```pip install -r requirments.txt```
```python main.py```
Adams.pgn:
5/4/23
Test loss: 0.47839900851249695
Test accuracy: 0.7689734101295471

5/5/23
Test loss: 0.4322868287563324
Test accuracy: 0.8030847311019897

5/28/23
Test loss: 0.4277477264404297
Test accuracy: 0.811082124710083

01/07/24 - large tweeks and set seed
Test loss: 0.5457004308700562
Test accuracy: 0.8052128553390503

01/08/24
Test loss: 0.37744206190109253
Test accuracy: 0.8561661839485168

01/10/24
Test loss: 0.4207151234149933
Test accuracy: 0.8471933007240295

01/19/24 - implemented batch processing to save memory
Test loss: 0.44103720784187317
Test accuracy: 0.8269000053405762

Wow it's been a whole year
01/08/25 - changed literally everything
Class 0: Precision: 0.7391, Recall: 0.6430, F1-Score: 0.6877
Class 1: Precision: 0.6695, Recall: 0.6921, F1-Score: 0.6806
Class 2: Precision: 0.6551, Recall: 0.7159, F1-Score: 0.6841

Accuracy: 75.01%, Val Loss: 0.7252, Val Accuracy: 68.05%

01/11/25 - upped to 64 epochs

Class 0: Precision: 0.6256, Recall: 0.6888, F1-Score: 0.6557
Class 1: Precision: 0.6873, Recall: 0.6018, F1-Score: 0.6417
Class 2: Precision: 0.6531, Recall: 0.6769, F1-Score: 0.6648

Train Loss: 0.6155, Train Accuracy: 71.75%, Val Loss: 0.7769, Val Accuracy: 65.24%

01/22/25 - 64 epoch with relative player scoring
Class 0: Precision: 0.6744, Recall: 0.6317, F1-Score: 0.6523
Class 1: Precision: 0.6029, Recall: 0.6741, F1-Score: 0.6365
Class 2: Precision: 0.6775, Recall: 0.6325, F1-Score: 0.6542

02/13/25 - 1000 epoch autoencoder with 1000 epoch DeepChess model
Class 0: Precision: 0.5456, Recall: 0.4714, F1-Score: 0.5058
Class 1: Precision: 0.5314, Recall: 0.6585, F1-Score: 0.5881
Class 2: Precision: 0.5377, Recall: 0.4555, F1-Score: 0.4932

Train Loss: 0.6301, Train Accuracy: 71.31%, Val Loss: 0.7762, Val Accuracy: 65.01%

docker tag chess_model:v1 ethancruz/chess_model
docker push ethancruz/chess_model:latest

