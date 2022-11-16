iris = load_iris()

X = iris["data"]
Y = iris["target"]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=RANDOM_STATUS)
