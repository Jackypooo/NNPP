import numpy as np
import matplotlib.pyplot as plt

class Layer:
    #Initializes a layer and assignes weights and biases to each neuron in the layer
    def __init__(self, numInputs, numNeurons):
        self.weights = 0.01 * np.random.randn(numNeurons, numInputs)
        self.weights = self.weights.T
        self.biases = np.zeros((1, numNeurons))
    
    #Forward pass through the layer determinging output
    def forward(self, inputs):
        self.inputs = inputs

        self.output = np.dot(inputs, self.weights) + self.biases

    #Backward pass through the layer determining derivatives of weights, biases, and inputs
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    #Takes input and applies the ReLU activation function (0 if negative, otherwise x)
    def forward(self, input):
        self.inputs = input
        self.output = np.maximum(0, input)

    #Finds the derivative of the ReLU activation function for use in determining gradient
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0     

class SoftmaxandCategoricalCrossEntropy():
    #Calculates Loss and applies Softmax activation function to output layer
    def forward(self, pred, true):
        #Softmax Activation Function Calculation
        expValues = np.exp(pred - np.max(pred, axis=1, keepdims=True))
        self.activationOutput = expValues / np.sum(expValues, axis=1, keepdims=True)

        #Calcualtion of Loss Function for Categorization (Categorical Cross Entropy)
        clipped = np.clip(self.activationOutput, 1e-7, 1-1e-7)

        result = -np.log(np.sum(clipped * true))
        loss = np.mean(result)
    
        return loss

    #Calcualtes the derivative of softmax activation function and loss function, later applied to gradient
    def backward(self, dvalues, true):
        samples = len(dvalues)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), true] -= 1
        self.dinputs = self.dinputs / samples

class SGD:
    #Initializes Gradient Decent algorithm
    def __init__(self, learningrate=1.0, decay=0):
        self.learningrate = learningrate
        #Learnging Rate Decay Implementation
        self.decay = decay
        self.i = 0

    #Updates each layers weights and baises based on the gradient (Network Learning)
    def update(self, layer):
        learningrate = self.learningrate - (self.decay * self.i)

        layer.weights += -learningrate * layer.dweights
        layer.biases += - learningrate * layer.dbiases

        self.i += 1

#Input and expected output data
X = np.array([[[1,0]], [[0,1]], [[0,0]], [[1,1]]])
y = np.array([[[0,1]], [[0,1]], [[0,0]], [[1,0]]])

#Actaul neural network code including learning
def NN():
    #Initializes the first layer and its corrisponding activation funciton
    Layer1 = Layer(2,4)
    Activation1 = Activation_ReLU()

    #Initializes second layer and its corrisponding activation function
    Layer2 = Layer(4,2)
    lossActivation = SoftmaxandCategoricalCrossEntropy()

    #Initializes Optimizer and applies decay
    optimizer = SGD(0.85, 0.1)

    #Sets limits to stop infinite loops
    loss = 1
    count = 1
    iterationlimit = 100

    #Loop through applying and learning 
    while (loss > 1e-6 and count < iterationlimit):
        #Uses random sample
        inputNum = np.random.randint(0,4)

        #Calculates first layer
        Layer1.forward(X[inputNum])
        Activation1.forward(Layer1.output)

        #Calculates second layer and loss
        Layer2.forward(Layer1.output)
        loss = lossActivation.forward(Layer2.output, y[inputNum])

        #Uses Loss to calculate gradient
        lossActivation.backward(lossActivation.activationOutput, y[inputNum])
        Layer2.backward(lossActivation.dinputs)
        Activation1.backward(Layer2.dinputs)
        Layer1.backward(Activation1.dinputs)

        #Uses Gradient to "teach" the network
        optimizer.update(Layer1)
        optimizer.update(Layer2)

        count += 1

    if count == iterationlimit:
        return(0)
    else:
        return(1)

#Runs network multiple times to determine it's success so we can continue to tweak
XX = []

for i in range(100):
    successes = 0
    for x in range(10):
        successes += NN()
    XX.append(successes)

#Basic MatPlotLib to graph data
fig, ax = plt.subplots()
ax.hist(XX)
plt.show()