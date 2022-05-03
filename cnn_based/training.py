import numpy as np
from model import *
import matplotlib.pyplot as plt

import torch.optim as optim


from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

import seaborn as sns
from sklearn.metrics import confusion_matrix
import pickle

train_x = pickle.load(open("train_x.p", "rb"))
test_x = pickle.load(open("test_x.p", "rb"))
train_y = pickle.load(open("train_y.p", "rb"))
test_y = pickle.load(open("test_y.p", "rb"))

train_x = torch.from_numpy(train_x)
train_y = torch.from_numpy(train_y)
test_x = torch.from_numpy(test_x)
test_y = torch.from_numpy(test_y)
net = AnomalyModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

for epoch in range(300):

    optimizer.zero_grad()

    outputs = net(train_x.float())
    loss = criterion(outputs, train_y)
    loss.backward()
    optimizer.step()

    print(f'[{epoch + 1}] loss: {loss.item() :.3f}')

print('Finished Training')
torch.save(net.state_dict(), "model.pt")
y_true = []
for element in test_y:
    y_true.append(np.argmax(element))
optimizer.zero_grad()

prediction_proba = net.forward(test_x.float())
prediction = np.argmax(prediction_proba.detach().numpy(), axis=1)


custCnnConfMat = confusion_matrix(y_true, prediction)
sns.heatmap(custCnnConfMat / np.sum(custCnnConfMat), annot=True, fmt='.3%', cmap='Blues')
plt.show()
plt.savefig("confusion_matrix.png")
print('Precision: %.3f' % precision_score(y_true, prediction, average='micro'))
print('F1 Score: %.3f' % f1_score(y_true, prediction, average='micro'))
print('Recall: %.3f' % recall_score(y_true, prediction, average='micro'))
print('Accuracy: %.3f' % accuracy_score(y_true, prediction))
