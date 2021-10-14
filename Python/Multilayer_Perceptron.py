def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def layer_sizes(X, Y):
    n_x = X.shape[0] # size of input layer
    n_h = 4
    n_y = Y.shape[0] # size of output layer
    return (n_x, n_h, n_y)

def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache

def compute_cost(A2, Y, parameters):
    
    m = Y.shape[1]
    
    logprobs = np.multiply(Y, np.log(A2)) + np.multiply(1 - Y, np.log(1 - A2))
    cost = -np.sum(logprobs) * (1 / m)
    
    cost = float(np.squeeze(cost))
    
    assert(isinstance(cost, float))
    
    return cost

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2,A1.T) * (1/m)
    db2 = np.sum(dZ2, axis = 1, keepdims = True) * (1/m)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T) * (1/m)
    db1 = np.sum(dZ1, axis = 1, keepdims = True) * (1/m)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

def update_parameters(parameters, grads, learning_rate = 0.7):
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    
    parameters =initialize_parameters(n_x, n_h, n_y)

    for i in range(0, num_iterations):
         
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)
        
        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y, parameters)
 
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)
 
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads)
        
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

def predict(parameters, X, Y):
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)
    
    answer = []
    for i in range(8):
        for j in range(8):
            if(predictions[0][i] == Y[0][j] and predictions[1][i] == Y[1][j] and predictions[2][i] == Y[2][j]):
                answer.append(j)
    return answer

##############################################################################################################################
import numpy as np

X = np.array([[1,1,1,1,1,1, 1,1,1,1,1,1, 0,0,0,0,1,1, 0,0,0,0,1,1, 0,0,0,0,1,1, 0,0,0,1,1,1, 1,1,1,1,1,0, 1,1,1,0,0,0],
              [1,1,0,0,0,0, 1,1,0,0,0,0, 1,1,0,0,0,0, 1,1,0,0,0,0, 1,1,0,0,0,0, 1,1,0,0,0,0, 1,1,1,1,1,1, 1,1,1,1,1,1],
              [1,1,1,1,1,1, 1,1,1,1,1,1, 1,1,0,0,0,0, 1,1,0,0,0,0, 1,1,0,0,0,0, 1,1,0,0,0,0, 1,1,1,1,1,1, 1,1,1,1,1,1],
              [1,1,1,1,1,1, 1,1,1,1,1,1, 0,0,0,0,1,1, 1,1,1,1,1,1, 1,1,1,1,1,1, 1,1,0,0,0,0, 1,1,1,1,1,1, 1,1,1,1,1,1],
              [1,1,1,1,1,1, 1,1,1,1,1,1, 1,1,0,0,1,1, 1,1,0,0,1,1, 1,1,0,0,1,1, 1,1,0,0,1,1, 1,1,1,1,1,1, 1,1,1,1,1,1],
              [1,1,0,0,1,1, 1,1,0,0,1,1, 1,1,1,1,1,1, 1,1,1,1,1,1, 1,1,0,0,1,1, 1,1,0,0,1,1, 1,1,1,1,1,1, 1,1,1,1,1,1],
              [0,0,1,1,0,0, 0,1,1,1,1,0, 0,1,1,1,1,0, 1,1,1,1,1,1, 1,1,1,1,1,1, 1,1,0,0,1,1, 1,1,0,0,1,1, 1,0,0,0,0,1],
              [0,0,1,1,0,0, 0,1,1,1,1,0, 1,1,1,1,1,1, 1,1,0,0,1,1, 1,1,0,0,1,1, 1,1,1,1,1,1, 0,1,1,1,1,0, 0,0,1,1,0,0]])
X = X.T

Y = np.array([[0,0,0],
              [0,0,1],
              [0,1,0],
              [0,1,1],
              [1,0,0],
              [1,0,1],
              [1,1,0],
              [1,1,1]])
Y = Y.T

X_test = np.array([[1,0,1,1,0,1, 1,0,1,1,1,1, 0,0,0,0,1,1, 0,0,0,0,1,1, 0,0,0,0,1,1, 0,0,0,1,0,1, 1,1,1,1,1,0, 1,1,1,0,0,0],
                   [1,0,0,0,0,0, 1,0,0,0,0,0, 1,1,0,0,0,0, 1,1,0,0,0,0, 1,1,0,0,0,0, 1,1,0,0,0,0, 1,0,1,1,1,0, 1,1,1,1,1,1],
                   [1,0,1,1,0,1, 1,0,1,1,1,1, 1,1,1,0,0,0, 1,1,1,0,0,0, 1,1,1,0,0,0, 1,1,0,0,1,0, 1,1,1,1,1,1, 1,1,1,1,1,1],
                   [1,1,1,1,0,1, 1,1,1,1,1,1, 0,0,0,0,1,1, 1,1,0,1,1,1, 1,1,0,1,1,1, 1,1,0,0,1,0, 1,1,1,1,1,1, 1,1,1,1,1,1],
                   [1,1,1,1,1,1, 1,0,1,1,1,1, 1,1,0,0,1,1, 1,1,0,0,1,1, 1,1,1,0,1,1, 1,1,0,0,0,1, 1,1,1,1,1,1, 1,1,1,1,1,1],
                   [1,0,0,0,0,1, 1,0,0,0,1,1, 1,1,0,1,1,1, 1,1,0,1,1,1, 1,1,1,0,1,1, 1,1,0,0,0,1, 1,1,1,1,1,1, 1,1,1,1,1,1],
                   [0,1,1,1,1,0, 0,0,1,1,1,0, 0,1,0,1,1,0, 1,1,0,1,1,1, 1,1,0,1,1,1, 1,1,0,0,0,1, 1,1,0,0,1,1, 1,0,0,0,0,1],
                   [0,1,1,1,1,0, 0,0,1,1,1,0, 1,1,0,1,1,1, 1,1,1,0,1,1, 1,1,1,0,1,1, 1,1,1,1,0,1, 0,1,1,1,1,0, 0,0,1,1,0,0]])

X_test = X_test.T

###############################################################################################################################

parameters = nn_model(X, Y, 4, num_iterations=20000, print_cost=True)

answer = predict(parameters, X_test, Y)

print("\n----------------------------------------학습 패턴----------------------------------------")
for i in range(0, 8):
    for T in range(0, 8):
        for j in range(6 * i, 6 * i + 6):
            print(("■" if (X.T[T][j]) else "□"), end='');
        print("\t", end=' ')
    print()
    
print("\n---------------------------------------- 입력패턴 ----------------------------------------")
for i in range(0, 8):
    for T in range(0, 8):
        for j in range(6 * i, 6 * i + 6):
            print(("■" if (X_test.T[T][j]) else "□"), end='');
        print("\t", end=' ')
    print()
    
print("\n----------------------------------------    결과     ----------------------------------------")
for i in range(0, 8):
    for T in answer:
        for j in range(6 * i, 6 * i + 6):
            print(("■" if (X.T[T][j]) else "□"), end='');
        print("\t", end=' ')
    print()
