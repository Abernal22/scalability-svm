import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import generate_data as gd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    #assumes all data files are available
    endSize = gd.sampSize[len(gd.sampSize)-1]
    for d in range(len(gd.dims)):
        #Read the dimension d data file in and split it.
        XY = pd.read_csv(f"datased_d{gd.dims[d]}_n{endSize}.csv")
        X = XY.iloc[:, :-1]
        Y = XY.iloc[:, -1]
        for n in range(len(gd.sampSize)):
            numSamps = gd.sampSize[n]
            xsamp = X[:numSamps]
            ysamp = Y[:numSamps]
            X_train, X_test, y_train, y_test = train_test_split(xsamp, ysamp, test_size=0.3, random_state=42)
            dual = LinearSVC(dual=True, loss="hinge", random_state=42)
            primal = LinearSVC(dual=False, random_state=42)

            #training models.
            startTime = time.time()
            dual.fit(X_train, y_train)
            dualTime = time.time() - startTime
            startTime = time.time()
            primal.fit(X_train, y_train)
            primalTime = time.time() - startTime

            #testing accuracy
            dualPred = dual.predict(X_test)
            primalPred = primal.predict(X_test)
            accDual = accuracy_score(y_test, dualPred)
            accPrimal = accuracy_score(y_test, primalPred)

            print(f"Dual: {numSamps} samples, {gd.dims[d]} features, {accDual} accuracy, {dualTime} time(s).")
            print(f"Primal: {numSamps} samples, {gd.dims[d]} features, {accPrimal} accuracy, {primalTime} time(s)")
           

