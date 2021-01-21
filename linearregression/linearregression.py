import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
data = pd.read_csv("student-mat.csv", sep=';')

data = data[["G1","G2","G3","failures","age","absences"]]

predict = "G3" #what we are trying to get

x = np.array(data.drop([predict],1)) #everything except predict values
y = np.array(data[predict]) #only want g3 value
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size=0.1)#x_train and y _train is some portion, x_test and y_test splits 10 percent because the information is not seen before
best = 0
# for _ in range(30):
#     linear = linear_model.LinearRegression() #makes a linear regression
#     linear.fit(x_train, y_train)
#     accuracy = linear.score(x_test, y_test)
#     print(accuracy)
#     if accuracy > best:
#         best = accuracy
#         with open("studentmodle.pickle", "wb") as f: #this basically saves the pickle file in our directory. Saving is used when a lot of data is used
#             pickle.dump(linear, f)

pickle_in = open("studentmodle.pickle", "rb") #this basically reads the pickle file
linear = pickle.load(pickle_in); #this loads the model into the variable linear
print("Co: ",linear.coef_) #coefficiant   m coefficinats
print("Intercept: ", linear.intercept_) #intercept

predictions = linear.predict(x_test)

for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i])
style.use("ggplot")
p = 'G1'
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
