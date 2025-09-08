import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression():
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.loss=[]
        pass

    def sigmoid(self, z): 
        return 1.0/(1.0+np.exp(-z)) #Sigmoid-funksjonen
    
    def loss_calc(self, y, y_pred): #Cross-entropy loss
        eps=1e-15 #Legger til denne slik at vi ikke trenger å tenke på log(0)
        y_pred=np.clip(y_pred, eps, 1 - eps) #Fjerne eps fra begge sider av y_pred
        return -np.mean(y*np.log(y_pred)+(1-y)*np.log(1-y_pred))

            #losses=self.loss_calc(y, p) #Beregner loss
            #self.loss.append(losses) #Tracker tapet

    def fit(self, X, y):
        X=np.asarray(X, dtype=float)
        y=np.asarray(y, dtype=float)

        n_samples, n_features=X.shape
        self.weights=np.zeros(n_features) 
        self.bias=0

        for i in range(self.epochs):
            lin_mod=np.dot(X, self.weights) + self.bias #Lineær regresjonsmodell
            p=self.sigmoid(lin_mod) #Bruker en lineær regresjonsmodell i sigmoid-funksjonen

            dweights=(1/n_samples)*np.dot(np.transpose(X), (p-y))
            dbias=(1/n_samples)*np.sum(p-y)

            self.weights -= dweights*self.learning_rate
            self.bias -= dbias*self.learning_rate

        return self
    
    def predict_probability(self, X):
        lin_mod = np.dot(X, self.weights) + self.bias
        return self.sigmoid(lin_mod)

    def predict(self, X):
        return (self.predict_probability(X) >= 0.5).astype(int)