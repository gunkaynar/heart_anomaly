import numpy as np
from model2 import *
import matplotlib.pyplot as plt

import torch.optim as optim

from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pickle

train_x = pickle.load(open("train_x.p", "rb"))
test_x = pickle.load(open("test_x.p", "rb"))
train_y = pickle.load(open("train_y.p", "rb"))
test_y = pickle.load(open("test_y.p", "rb"))

train_x = np.reshape(train_x, (20000, 360))
train_y = np.reshape(train_y, (20000, 5))
train_x, validation_x, train_y, validation_y = train_test_split(train_x, train_y, test_size=0.2)

train_x = np.reshape(train_x, (16000, 360, 1))
train_y = np.reshape(train_y, (16000, 5))
validation_x = np.reshape(validation_x, (4000, 360, 1))
validation_y = np.reshape(validation_y, (4000, 5))

train_x = torch.from_numpy(train_x)
train_y = torch.from_numpy(train_y)
test_x = torch.from_numpy(test_x)
test_y = torch.from_numpy(test_y)
validation_x = torch.from_numpy(validation_x)
validation_y = torch.from_numpy(validation_y)

epoch_num = 100
input_dim = 1    # input dimension
hidden_dim = 16  # hidden layer dimension
layer_dim = 1     # number of hidden layers
output_dim = 5   # output dimension
net = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
train_losses = []
validation_losses = []
for epoch in range(epoch_num):
    for i, (x, y) in enumerate(batch_generator_torch(train_x, train_y, batch_size=64, shuffle=True)):
        train_loss = []
        optimizer.zero_grad()
        output = net(x.float())
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    train_losses.append(np.mean(train_loss))

    for i, (x, y) in enumerate(batch_generator_torch(validation_x, validation_y, batch_size=len(validation_x),
                                                     shuffle=False)):
        validation_loss = []
        output = net(x.float())
        loss_val = criterion(output, y)
        validation_loss.append(loss_val.item())
    validation_losses.append(np.mean(validation_loss))
    print(f'[{epoch + 1}] training loss: {np.mean(train_loss) :.3f} validation loss {np.mean(validation_loss) :.3f}')

print('Finished Training')
torch.save(net.state_dict(), "model.pt")
epochs = np.arange(1, len(train_losses) + 1)
plt.plot(epochs, train_losses, 'r', label='Training loss')
plt.plot(epochs, validation_losses, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.png')
plt.show()

plt.clf()

y_true = []
for element in test_y:
    y_true.append(np.argmax(element))

prediction_proba = net.forward(test_x.float())
prediction = np.argmax(prediction_proba.detach().numpy(), axis=1)

custCnnConfMat = confusion_matrix(y_true, prediction)
sns.heatmap(custCnnConfMat / np.sum(custCnnConfMat), annot=True, fmt='.3%', cmap='Blues')
plt.savefig("confusion_matrix.png")
plt.show()

print('Precision: %.3f' % precision_score(y_true, prediction, average='macro'))
print('F1 Score: %.3f' % f1_score(y_true, prediction, average='macro'))
print('Recall: %.3f' % recall_score(y_true, prediction, average='macro'))
print('Accuracy: %.3f' % accuracy_score(y_true, prediction))
