# libraries :
from builtins import print
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import minmax_scale
import numpy
import re

# PRE PROCESSING :
# Data Integration :
f = open("20150801.as2types.txt", "r")      # reads the Containing AS_TYPES; 01/08/2018
if f.mode == 'r':
    contents = f.read()
content = contents[205:len(contents)-1]       # the main values start at 205th character
s = [len(content)]
s = content.split('|')        # file pattern : [AS ID]|[Source Data Base Name]|[As type]

i = 0
s2 = []
for x in range(0, len(s), 2):    # discarding [Source Data Base Name]
    s2.append(s[x])
s3 = ''.join(s2)

s3_cleand = s3.replace("\n", "")     # removing "next lines"
tmp = re.split('(\d+)', s3_cleand)  # splits by values and saves in a temp list pattern : list[AS id, AS type]


AS_features = pd.read_csv("result.csv")    # reads the AS features from "AS RANK DATA BASE"
AS_features.describe()
AS_types = list()
as_id = AS_features['data__id']   # column "data__id" is containing AS IDs
asid = list()
for x in range(0, len(as_id)):
    asid.append(str(as_id[x]))

for i in range(0, len(asid)):
    for j in range(1, len(tmp), 2):
        if tmp[j] == asid[i]:        # matches AS IDs and creates AS types list by the same order
            AS_types.append(tmp[j+1])
    if len(AS_types)-1 < i:    # if doesnt match by any, put type = "Transit/Access"  {DATA CLEANING}
        AS_types.append('Transit/Access')
# all AS features :
Columns = ['total', 'data__longitude', 'data__cone__asns', 'data__degree__providers', 'data__cone__prefixes',
           'data__cone__addresses', 'data__degree__globals', 'data__degree__peers', 'data__degree__siblings',
           'data__degree__customers', 'data__degree__transits', 'data__latitude', 'data__id', 'data__name',
           'data__source', 'data__org__name', 'data__org__id', 'data__country', 'data__rank', 'data__clique',
           'data__country_name']
# target features = ["providers", "prefixes", "addresses", "globals", "peers", "customers", "transits"]

# Data Cleaning :
AS_features.dtypes.index
AS_features.fillna(AS_features.mean(), inplace=True)    # fills the NULL (NONE) values by the average of feature(column)

# Data Reduction :
# removing extra data (features)
AS_features.drop(['total', 'data__longitude', 'data__cone__asns', 'data__degree__siblings', 'data__latitude',
                  'data__id', 'data__name', 'data__source', 'data__org__name', 'data__org__id', 'data__country',
                  'data__rank', 'data__clique', 'data__country_name'], axis=1, inplace=True)
# print(binary.describe())

# Data Normalization : min-max algorithm
df = minmax_scale(AS_features, feature_range=(0, 100))  # type: numpy.ndarray
# numpy.savetxt("foo.csv", df, delimiter=",")

# DATA MINING :

# creating the classifier (machine learning) :
X_train, X_test, y_train, y_test = train_test_split(df, AS_types, test_size=0.3, random_state=58)
# training set : 70%  and test set : 30%   of data set
# dividing data set randomly

# building the decision tree :
dt_train_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=6, min_samples_leaf=3)
dt_train_gini.fit(X_train, y_train)
print("classes : ", dt_train_gini.classes_, "features_name : ", ['data__degree__provider',
                                                                 'data__cone__prefixes', 'data__cone__addresses',
                                                                 'data__degree__globals', 'data__degree__peers',
                                                                 'data__degree__customers', 'data__degree__transits'])

# building the Classification rules (text file):
with open("Classification Rules.txt", "w") as f:
    f = tree.export_graphviz(dt_train_gini, out_file=f, feature_names=['Provider_degree', 'AS_prefixes', 'AS_addresses',
                                                                       'Global_degree', 'Peer_degree',
                                                                       'Customer_degree', 'Transit_degree'],
                             class_names=dt_train_gini.classes_)

# POST PROCESSING:
# the model evaluation :

# actual classes : y_test , predicted classes : dt_train_gini.predict(X_test)
# accuracy:
print("Accuracy is ", accuracy_score(y_test, dt_train_gini.predict(X_test))*100)
# confusion matrix :
results = confusion_matrix(y_test, dt_train_gini.predict(X_test))
print('Confusion Matrix :')
print(results)
# other evaluation metrics (precision, recall, f1-score, support) :
print(classification_report(y_test, dt_train_gini.predict(X_test)))
