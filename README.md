# Experiment-2--Implementation-of-Perceptron
## AIM:

To implement a perceptron for classification using Python

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:
A Perceptron is a basic learning algorithm invented in 1959 by Frank Rosenblatt. It is meant to mimic the working logic of a biological neuron. The human brain is basically a collection of many interconnected neurons. Each one receives a set of inputs, applies some sort of computation on them and propagates the result to other neurons.
A Perceptron is an algorithm used for supervised learning of binary classifiers.Given a sample, the neuron classifies it by assigning a weight to its features. To accomplish this a Perceptron undergoes two phases: training and testing. During training phase weights are initialized to an arbitrary value. Perceptron is then asked to evaluate a sample and compare its decision with the actual class of the sample.If the algorithm chose the wrong class weights are adjusted to better match that particular sample. This process is repeated over and over to finely optimize the biases. After that, the algorithm is ready to be tested against a new set of completely unknown samples to evaluate if the trained model is general enough to cope with real-world samples.
The important Key points to be focused to implement a perceptron:
Models have to be trained with a high number of already classified samples. It is difficult to know a priori this number: a few dozen may be enough in very simple cases while in others thousands or more are needed.
Data is almost never perfect: a preprocessing phase has to take care of missing features, uncorrelated data and, as we are going to see soon, scaling.
Perceptron requires linearly separable samples to achieve convergence.
The math of Perceptron
If we represent samples as vectors of size n, where ‘n’ is the number of its features, a Perceptron can be modeled through the composition of two functions. The first one 
f(x) maps the input features  ‘x’  vector to a scalar value, shifted by a bias ‘b’

A threshold function, usually Heaviside or sign functions, maps the scalar value to a binary output:

Indeed if the neuron output is exactly zero it cannot be assumed that the sample belongs to the first sample since it lies on the boundary between the two classes. Nonetheless for the sake of simplicity,ignore this situation.


## ALGORITHM:
1) Importing the libraries.

2) Importing the dataset.

3) Plot the data to verify the linear separable dataset and consider only two classes.

4) Convert the data set to scale the data to uniform range by using Feature scaling

Split the dataset for training and testing
Define the input vector ‘X’ from the training dataset
Define the desired output vector ‘Y’ scaled to +1 or -1 for two classes C1 and C2
Assign Initial Weight vector ‘W’ as 0 as the dimension of ‘X’
Assign the learning rate
For ‘N ‘ iterations ,do the following:
        v(i) = w(i)*x(i)
         
        W (i+i)= W(i) + learning_rate*(y(i)-t(i))*x(i)
Plot the error for each iteration 
Print the accuracy

## PROGRAM:
## LIBRARIES:
```import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
```
## CLASS PERCEPTRON :

```
class Perceptron:
  def __init__(self, learning_rate=0.1):
    self.learning_rate = learning_rate
    self._b = 0.0
    self._w = None
    self.misclassified_samples = []
  def fit(self, x: np.array, y: np.array, n_iter=10):
    self._b = 0.0
    self._w = np.zeros(x.shape[1])
    self.misclassified_samples = []
    for _ in range(n_iter):
      errors = 0
      for xi,yi in zip(x,y):
        update = self.learning_rate * (yi-self.predict(xi))
        self._b += update
        self._w += update*xi
        errors += int(update !=0)
      self.misclassified_samples.append(errors)
  def f(self,x:np.array) -> float:
    return np.dot(x,self._w) + self._b
  def predict(self, x:np.array):
    return np.where(self.f(x) >= 0,1,-1) 
    df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Semester 3/Neural Networks/Exp_2/IRIS.csv")
df.head()

y = df.iloc[:,-1].values
y

x = df.iloc[:,:-1].values
x

x = x[0:100,0:2]
y = y[0:100]

plt.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='Setosa')

plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='x',
            label='Versicolour')

plt.xlabel("Sepal length")
plt.ylabel("Petal length")
plt.legend(loc='upper left')

plt.show()

y = np.where(y=="Iris-setosa",1,-1)
y

x[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
x[:, 1] = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,
                                                    random_state=0)

classifier = Perceptron(learning_rate=0.01)
classifier.fit(x_train, y_train)


plt.plot(range(1, len(classifier.misclassified_samples) + 1),
         classifier.misclassified_samples, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Errors')
plt.show()

print("accuracy = " , accuracy_score(classifier.predict(x_test), y_test)*100) 

```
## OUTPUT :
<img width="493" alt="NN 2" src="https://user-images.githubusercontent.com/114279259/197104832-025fd78e-09c2-4bdc-880b-0b7168750253.png">

<img width="320" alt="nn 1" src="https://user-images.githubusercontent.com/114279259/197104863-9d4ec35e-4028-4c6c-b8a9-53a38ca7bc71.png">

<img width="483" alt="nn3" src="https://user-images.githubusercontent.com/114279259/197104897-224b6cb8-3fdb-4c3b-9f72-867a301cd940.png">

<img width="314" alt="nn4" src="https://user-images.githubusercontent.com/114279259/197104921-13b5a436-7bf1-4bb6-8977-c1a4a1616cb8.png">

<img width="211" alt="nn 5" src="https://user-images.githubusercontent.com/114279259/197104941-d9b98a36-7b58-4877-bdb6-c14b97b7ff42.png">

## RESULT :

Thus a perceptron for classification is implemented using python.



