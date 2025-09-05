import numpy as np

class LinearRegression():
    def __init__(self, learning_rate=0.0001, epochs=1000):
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.loss=[]
        pass

    def fit(self, X, y):
        n=X.shape[0]
        self.weights=np.zeros(X.shape[1]) 
        self.bias=4.2
        for i in range(self.epochs):
            y_pred=np.dot(X, self.weights)+self.bias #Lineær regresjonsmodell
            losses=np.mean((y-y_pred)**2) #Mean squared error
            self.loss.append(losses) #Tracker tapet

            dweights=(2/n)*np.dot(np.transpose(X), (y-y_pred))
            dbias=(2/n)*np.sum(y-y_pred)

            self.weights += dweights*self.learning_rate
            self.bias += dbias*self.learning_rate
        return self
    
    def predict(self, X):
        y_pred=np.dot(X, self.weights)+self.bias
        return y_pred