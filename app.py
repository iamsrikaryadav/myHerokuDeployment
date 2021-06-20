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

@app.route("/api/JobPostings/<company>")
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

if __name__=="__main__":
  port = int(os.environ.get("PORT",5000))
  app.run(host="0.0.0.0",port=port)