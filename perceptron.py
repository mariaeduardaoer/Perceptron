import numpy as np

class Perceptron:
    
    def __init__(self, input_values, output_values, learning_rate, activation_function):
        ones_column = np.ones((len(input_values), 1)) * (-1)
        self.input_values = np.append(ones_column, input_values, axis=1)
        self.output_values = output_values
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.W = np.random.rand(self.input_values.shape[1])
        print(f'Inicial W: {self.W}')
    
    def train(self):
        epochs = 1
        error = True
 
        while error:
            error = False
            # print(f'> EPOCH {epochs} <')
            for x, d in zip(self.input_values, self.output_values):
                
                u = np.dot(x, self.W)
                y = self.activation_function.g(u)
                # print(f'Input: {x}, Output: {y}, Expected: {d}')
               
                if y != d:
                    #print('Output is different from Expected, recalculating W ...')
                    #print('')
                    #print(f'Actual W: {self.W}')

                   
                    self.W = self.W + self.learning_rate * (d - y) * x
                    error = True
                    #print(f'New W: {self.W}')
                    
                    break
                    
            epochs += 1

        print(f'EPOCHS: {epochs}')    
        print(f'Final W: {self.W}')
        print('')


    def evaluate(self, input_values):
        input_values= np.append([[-1]],[input_values], axis=1) # -1 na primeira posição
        u = np.dot(input_values, self.W)
        return self.activation_function.g(u)