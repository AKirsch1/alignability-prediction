#!/usr/bin/python3
import os
import sys
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def classify(mean, median, standardDeviation, minValue, maxValue):
    classifier = joblib.load(os.path.join(BASE_DIR, '../models/model_alignability_prediction.joblib'))
    y_predict = classifier.predict([[mean, median, standardDeviation, minValue, maxValue]])
    return y_predict[0]


def main():
    if (len(sys.argv) != 6):
        print('Usage: python3 alignability_prediction.py <mean> <median> '
              '<standardDeviation> <minValue> <maxValue>')
        exit(0)
    print(classify(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]),
                   float(sys.argv[4]), float(sys.argv[5])))    


if __name__ == "__main__":
    main()