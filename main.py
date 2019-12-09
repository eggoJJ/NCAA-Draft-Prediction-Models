import numpy as np
import pandas as pd
import pydotplus
import pydot

from IPython.display import Image  
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
from subprocess import check_call
from sklearn.model_selection import KFold
from sklearn import metrics
 

def read():
    playerData = []
    with open('tmp_file.txt','r') as f:
        for line in f:
            playerData.append(line)
                #runScrapy(word)
    return playerData 



'''
for i in range(len(myData)):
    if(len(myData[i]) ==1):
        myData.pop(i)
    new_str = myData[i].split(',', 1)[1]
    new_str = new_str.rsplit("\n",2)[0]
    myData[i] = new_str
'''

dataset = pd.read_csv("tmp_file.txt")

dataset.columns = ['name','games','games_started','minutes_played',
                  'field_goals','field_goals_attempted','field_goal_percent','two_pointers','two_pointers_attempted','two_pointers_percent','three_pointers',
                  'three_pointers_attempted','three_pointers_percent','free_throws','free_throws_attempted','free_throws_percent','offensive_rebounds',
                  'defensive_rebounds','total_rebounds','assists','steals','blocks','turn_overs','personal_fouls','points','classification',]
X = dataset.drop(['name', 'classification'], axis = 1)


y = dataset['classification']

pd.DataFrame(X).fillna(0, inplace = True)
pd.DataFrame(y).fillna(0, inplace = True)

kf = KFold(n_splits=10, random_state=None, shuffle=True) # 5 Fold split 
#KFold(n_splits=5, random_state=None, shuffle=False)

#X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 1)

treeIndex = 0
for train_index, test_index in kf.split(X): #loop for the splits
    X_train, X_test = X.iloc[train_index], X.iloc[test_index] # defining the training/testing data for this iteration
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    #model = tree.DecisionTreeClassifier(criterion="entropy",max_depth=7)
    model = tree.DecisionTreeClassifier(max_depth=10) # Decision Tree CART model
    model.fit(X_train,y_train) # fitting out data to the selected model

    #Assessing the Accuracy 
    y_predict = model.predict(X_test) 
    print(accuracy_score(y_test,y_predict))

    #Displaying Correct/Uncorrect classified points
    displayMatrix = pd.DataFrame(
        confusion_matrix(y_test, y_predict),
        columns=['Predicted Not Drafted', 'Predicted Drafted'],
        index=['True Not Drafted', 'True Drafted']
    )

    #scores = cross_val_score(model, dataset, y, cv=6)
    #print ('Cross-validated scores:', scores)
    print(displayMatrix)
    
    dotfile = open("tree.dot", 'w+')
    tree.export_graphviz(model, out_file = dotfile, feature_names = X.columns)
    dotfile.close()
    #myDotTree = tree.export_graphviz(model,feature_names=X.columns, filled = True, rounded = True)
    #check_call(['dot', '-T', 'png', 'tree.dot', '-o', 'tree.png'])
    (graph,) = pydot.graph_from_dot_file('tree.dot')
    graph.write_png('tree-%d.png' % treeIndex)
    treeIndex+=1
#graph = pydotplus.graph_from_dot_data(myDotTree)  

# Show graph
#Image(graph.create_png())
