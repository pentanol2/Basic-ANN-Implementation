'''
    An implemtation from the sratch of a feeback  ANN
    using Stochastic Gradient Descent algorithm
    By: YOUSSEF AIDANI
    Year: 2018
'''
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from math import exp
from math import floor
from random import random
import pandas as pd


file_name = 'input.csv'
data = None
data_copy =None
training_set = pd.DataFrame()
test_set = pd.DataFrame()
class_vector = None
class_index = 0
network = list()
attribute_count = 0
test_vs_predicted = None
recall = None
precision = None
f1score = None
cmatrix = None

# we load the dataset from the csv file 
def get_dataset(file='',delimiter = ''):
    global file_name
    global data
    global data_copy
    file_name = file
    data = pd.read_csv(file_name,delimiter)
    data_copy = data.copy()
    # we shuffle the dataset before getting the labels
    data_copy = shuffle_dataframe(data_copy)
    train_class_vector = None
    test_class_vector = None

def shuffle_dataframe(dataframe):
    dataframe = dataframe.copy()
    dataframe = dataframe.sample(frac=1)
    return dataframe

# we select the class/ target specify its index
def get_class(index):
    global data_copy
    global class_vector
    global class_index
    global attribute_count
    attribute_count = len(data_copy.iloc[0])-1
    class_index = index
    try:
        class_vector = data_copy.iloc[:,index]        
    except Exception as e:
        print('The class index is not found!')

# we normalize the dataset by squashing all the values between 0 and 1 
# for faster processing
def normalize_data():
    global class_index
    global data_copy
    global data
    current_max = 255
    x = 0 
    while x<len(data.iloc[0]):
        if(x == class_index):
            break
        data_copy.iloc[:,x] = data_copy.iloc[:,x]/current_max
        x+=1

# we split the dataset into training according a specefic training ratio
def split_dataset(training_ratio):
    global training_set
    global test_set
    global data_copy
    global train_class_vector
    global test_class_vector
    # copy the column names of the main dataset 
    training_set = training_set.reindex_like(data)
    training_set = training_set.dropna()
    test_set = test_set.reindex_like(data)
    test_set = test_set.dropna()
    training_size = floor(len(data)*training_ratio)
        
    # we associate the rest of records to the test set  
    train_class_vector = data_copy.iloc[:training_size,-1]
    training_set = data_copy.iloc[:training_size,:-1]    
    test_class_vector = data_copy.iloc[training_size:,-1] 
    test_set = data_copy.iloc[training_size:,:-1]    

# execute all in the preprocessing unit                
def preprocessing_unit():
    global data
    get_dataset(file_name,';')    
    get_class(len(data.iloc[0])-1)
    normalize_data()
    split_dataset(0.8)

# we calculate the accuracy percentage
def get_accuracy(actual,predicted):
    corrent=0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct +=1
    return correct / float(len(actual))*100.0

# configure our netowrk
 # we set up the network
def setup_network(hiddens, outputs):
    global attribute_count
    global network
    inputs = attribute_count
    # the input layer is just arow from the dataset so we dont see it here as a separate list
    # we randomly set weights for the input layer
    hidden_layer = [{'weights':[random() for i in range(inputs + 1)]} for i in range(hiddens)]
    network.append(hidden_layer)
    #we randomly set weights for the output layer
    output_layer = [{'weights':[random() for i in range(hiddens + 1)]} for i in range(outputs)]
    network.append(output_layer)
    # for both cases a bias weight was added +1
   

# We calculate the weighted sum for input
def weighted_sum(weights,row):
    activation = weights[-1]
    for i in range(len(row)):
        activation += weights[i] * row[i]
    return activation

# Transfer neuron activation
def sigmoid(weighted_sum):
    return 1.0 / (1.0 + exp(-weighted_sum))

# Forward propagate input to a network output
def forward_propagate(row):
    global network
    inputs = row
    #print('row: '+str(row))
    # we foreward the input through all the netwrok layers
    for layer in network:
        new_inputs = [] # This is the current input that will be fed to the next layer
        for neuron in layer:
            activation = weighted_sum(neuron['weights'], inputs)
            neuron['output'] = sigmoid(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    #print('Final input: '+str(inputs))
    return inputs

# We calculate the derivative of an neuron output
#because we use the sigmoid function the derivative is calculated as follows
def derivate_sigmoid(output):
	return output * (1.0 - output)

# we backpropagate the error
def backward_propagate_error(actual):
    global network
    #because this is backpropagation we begin looping from the last layer
    for i in reversed(range(len(network))):                
        layer = network[i]
        #we store the error corresponding to each neuron here
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['error'])
                errors.append(error)
        else:
            for j in range(len(layer)):
               neuron = layer[j]
               errors.append(actual - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['error'] = errors[j] * derivate_sigmoid(neuron['output'])

# we update nerurons weights after calculating the corresponding errors
def new_weights(row, learning_rate):
    global network
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += learning_rate * neuron['error'] * inputs[j]
            neuron['weights'][-1] += learning_rate * neuron['error']

# Train a network for a fixed number of epochs
def train_network(learning_rate, epochs):
    global training_set
    global train_class_vector
    x= 0
    row = None
    for epoch in range(epochs):
        print('epoch: '+str(epoch)+'\n')
        x=0
        while x <len(training_set)-1:
            row = training_set.iloc[x]
            outputs = forward_propagate(row)
            backward_propagate_error(train_class_vector.iloc[x])
            new_weights(row, learning_rate)
            x +=1


# Make a test our network by predicting outputs of text values
def predict_output(row):
    outputs = forward_propagate(row)
	#]return outputs.index(max(outputs))
    return outputs
def test_model(decision_boundary = 0.5):
    global test_set
    global test_class_vector
    global test_vs_predicted
    tested_value = 1
    test_vs_predicted = pd.DataFrame()
    x =0
    while x < len(test_set):
        print(predict_output(test_set.iloc[x]))
        if predict_output(test_set.iloc[x])[0] < decision_boundary:
            tested_value = 0
        else:
            tested_value = 1
        test_vs_predicted = test_vs_predicted.append({'actual':test_class_vector.iloc[x],'predicted':tested_value},ignore_index=True)
        x+=1        
def processing_unit():
    setup_network(2,1)
    train_network(0.6, 500)
def measurements():
    global recall
    global precision 
    global f1score
    global test_vs_predicted 
    global cmatrix
    y_test = test_vs_predicted['actual'];   
    y_pred = test_vs_predicted['predicted']    
    recall = recall_score(y_test,y_pred)
    precision =  precision_score(y_test,y_pred)
    f1score = f1_score(y_test,y_pred)
    cmatrix = confusion_matrix(y_test,y_pred)
    print('Recall :'+str(recall))
    print('Precision :'+str(precision))
    print('F1 score :'+str(f1score))
    print('Confusion Matrix :\n'+str(cmatrix))
    

def write_results_to_file():
    global recall
    global precision 
    global f1score
             
    f = open('results.txt','w+')
    f.write('For 0.5 decision boundary the results are the following:\n')
    f.write('Recall :'+str(recall)+'\n')
    f.write('Precision :'+str(precision)+'\n')
    f.write('F1 score :'+str(f1score)+'\n')

preprocessing_unit()
processing_unit()
test_model()
measurements()
write_results_to_file()
    