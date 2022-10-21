import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier


def k_folder_result(X, Y, model, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    result = cross_val_score(model, X, Y, cv=kf, scoring='accuracy')
    print(f'K = {k} Accuracy: %.3f' % result.mean())
    return result


iris = load_iris()
iris_x = iris['data']
iris_y = iris['target']
dsTree = DecisionTreeClassifier()
k_folder_result(iris_x, iris_y, dsTree, 10)  # 十折交叉验证 (k=10)
k_folder_result(iris_x, iris_y, dsTree, 8)   # 归一法      (k=8)

mnist = load_digits()
mnist_x = mnist['images']
mnist_y = mnist['target']
mnist_slide = mnist['data']
k_folder_result(mnist_slide, mnist_y, dsTree, 10)  # 十折交叉验证 & 留一法 (k=10)
k_folder_result(mnist_slide, mnist_y, dsTree, 5)  # 五折交叉验证         (k=5)

fig, ax = plt.subplots(2, 2)
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(mnist_x[i], cmap='gray')
    ax[i // 2][i % 2].set_title(mnist_y[i])
    plt.axis(False)
plt.show()
