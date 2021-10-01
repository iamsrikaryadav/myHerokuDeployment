from flask import Flask, render_template, jsonify
import os, json
import pandas as pd
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

print("Top print executed")
@app.route("/")
def index():
  return render_template("index.html")

@app.route("/charts/<value>")
def loadHtml(value):
  print("here")
  print(value)
  return render_template(value)

@app.route("/api/JobPostings/<company>/")
def getCompanyYearDetails(company):
  print("executed")
  with open('source/result.json') as json_file:
      data = json.load(json_file)
      # print("*****",data)
  result=data[str(company)]
  # print(result)
  data = str(result)
  data = eval(data)
  # true=True
  return jsonify({"data" : data})

@app.route("/api/JobPostings/<company>/<year>")
def getCompanyMonthDetails(company,year):
  print("getCompanyMonthDetails executed")
  with open('source/hiringMonthRes.json') as json_file:
      data = json.load(json_file)
      # print("*****",data)
  result=data[str(company)][str(year)]
  # print(result)
  data = str(result)
  data = eval(data)
  # true=True
  return jsonify({"data" : data})

@app.route('/<filename>/<predfile>/deeplearning')
def deeplearning(filename,predfile):
    try:
        data=pd.read_csv(filename, header=None)
    except:
        data=pd.read_excel(filename, header=None)
    
    print("---------------------------------")

    try:
        pred=pd.read_csv(predfile, header=None)
    except:
        pred=pd.read_excel(predfile, header=None)
        
    
    from sklearn.model_selection import train_test_split
    # from sklearn.linear_model import LogisticRegression
    import numpy as np

    X = np.array(data.iloc[:,:-1].values)  
    y = np.array(data.iloc[:,-1].values)
    # X=data.iloc[:,:-1].values # print(len(X)) y=df.iloc[:,-1].values
    
    dictin={}
    # Splitting the data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25) 
    print("*************************")
    X_train = MinMaxScaler().fit_transform(X_train)
    X_test = MinMaxScaler().fit_transform(X_test)

    len_y = len(y_train)
    # y_train = y_train.values
    y_train = y_train.reshape(len_y, 1)

    len_test_y = len(y_test)
    # y_test = y_test.values
    y_test = y_test.reshape(len_test_y, 1)

    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()

    def ann_mlp():
        features=data.shape[1]-1
        X = tf.placeholder(shape=[None, features], dtype=tf.float32)
        Y = tf.placeholder(tf.float32, [None, 1])

        # input
        W1 = tf.Variable(tf.random_normal([features,features*2], seed=0), name='weight1')
        b1 = tf.Variable(tf.random_normal([features*2], seed=0), name='bias1')
        layer1 = tf.nn.sigmoid(tf.matmul(X,W1) + b1)

        # hidden1
        W2 = tf.Variable(tf.random_normal([features*2,features*2], seed=0), name='weight2')
        b2 = tf.Variable(tf.random_normal([features*2], seed=0), name='bias2')
        layer2 = tf.nn.sigmoid(tf.matmul(layer1,W2) + b2)

        # hidden2
        result=features*2
        W3 = tf.Variable(tf.random_normal([features*2,result*2], seed=0), name='weight3')
        b3 = tf.Variable(tf.random_normal([result*2], seed=0), name='bias3')
        layer3 = tf.nn.sigmoid(tf.matmul(layer2,W3) + b3)

        # output
        W4 = tf.Variable(tf.random_normal([result*2,1], seed=0), name='weight4')
        b4 = tf.Variable(tf.random_normal([1], seed=0), name='bias4')
        logits = tf.matmul(layer3,W4) + b4
        hypothesis = tf.nn.sigmoid(logits)

        cost_i = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=Y)
        cost = tf.reduce_mean(cost_i)

        train = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

        prediction = tf.cast(hypothesis > 0.5, dtype=tf.float32)
        correct_prediction = tf.equal(prediction, Y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

        # print("\n============Processing============")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for step in range(10001):
                sess.run(train, feed_dict={X: X_train, Y: y_train})
                if step % 1000 == 0:
                    loss, acc = sess.run([cost, accuracy], feed_dict={X: X_train, Y: y_train})
                    print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))

            train_acc = sess.run(accuracy, feed_dict={X: X_train, Y: y_train})
            test_acc,test_predict,test_correct = sess.run([accuracy,prediction,correct_prediction], feed_dict={X: X_test, Y: y_test})
            
            # print("\n============Results============")
            # print("test_predict values: ",test_predict)
            print("Model Prediction =", train_acc)
            print("Test Prediction =", test_acc)
            
            return train_acc,test_acc,test_predict
        
    ann_mlp_train_acc, ann_mlp_test_acc, test_predict = ann_mlp()
    res = fbeta_score(y_test, test_predict, average='binary', beta=0.5)
    # res_1=fbeta_score(y_test, test_predict, average='binary', beta=0.5)
    dictin["test"] = res
    return jsonify(dictin)

if __name__=="__main__":
  port = int(os.environ.get("PORT",5000))
  app.run(host="0.0.0.0",port=port)