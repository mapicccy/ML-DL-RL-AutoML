import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import base
from sklearn.neighbors import KNeighborsClassifier

scikit_iris = base.load_iris()
iris = pd.DataFrame(data=np.c_[scikit_iris['data'], scikit_iris['target']],
                    columns=np.append(scikit_iris.feature_names, ['y']))

iris.head(3)
