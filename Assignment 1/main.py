from helper import *
from solution import *


#Use for testing the training and testing processes of a model
def train_test_a_model(modelname, train_data, train_label, test_data, test_label, max_iter, learning_rate):
	print(modelname+" testing...")
	logireg(modelname, train_data, train_label, test_data, test_label, max_iter, learning_rate)
	

def logireg(modelname, train_data, train_label, test_data, test_label, max_iter, learning_rate):
        # max iteration test cases
        m_iter_count = 1
        for m_iter in max_iter:
                w = logistic_regression(train_data, train_label, m_iter, learning_rate[1])
                ATrain, ATest = accuracy(train_data, train_label, w), accuracy(test_data, test_label, w)
                print ("Testcase For Max Iteration %d" %m_iter_count)
                print ("Accuracy for Training: %f" %ATrain)
                print ("Accuracy for Testing: %f \n" %ATest)
                m_iter_count += 1

        l_rate_case_count = 1
        # learning rate test cases
        for l_rate in learning_rate:
                w = logistic_regression(train_data, train_label, max_iter[3], l_rate)
                ATrain, ATest = accuracy(train_data, train_label, w), accuracy(test_data, test_label, w)
                print ("Testcase For Learning %d" %l_rate_case_count)
                print ("Accuracy for Training: %f" %ATrain)
                print ("Accuracy for Testing: %f \n" %ATest)
                l_rate_case_count += 1
        print(modelname+" test done. \n")


def test_logistic_regression():
	max_iter = [100, 200, 300, 400]
	learning_rate = [0.1, 0.3, 0.5, 0.7]
	traindata,testdata = "../data/train.txt", "../data/test.txt"
	train_data,train_label = load_features(traindata)
	test_data, test_label = load_features(testdata)
	
	train_test_a_model("logistic regression", train_data, train_label, test_data, test_label, max_iter, learning_rate)
	

def test_thirdorder_logistic_regression():
	max_iter = [100, 200, 300, 400]
	learning_rate = [0.1, 0.3, 0.5, 0.7]
	traindata,testdata = "../data/train.txt", "../data/test.txt"
	train_data,train_label = load_features(traindata)
	test_data, test_label = load_features(testdata)
	new_train_data = thirdorder(train_data)
	new_test_data = thirdorder(test_data)
	train_test_a_model("3rd order logistic regression", new_train_data, train_label, new_test_data, test_label, max_iter, learning_rate)


if __name__ == '__main__':

	test_logistic_regression()
	test_thirdorder_logistic_regression()
