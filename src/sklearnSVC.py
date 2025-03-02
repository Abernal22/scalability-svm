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

    #Demension on vertical axis
    #accuracy data for tables
    primalAccData = []
    dualAccData = []
    #runtime measurements for tables
    primalTimeData = []
    dualTimeData = []

    for d in range(len(gd.dims)):
        #Read the dimension d data file in and split it.
        XY = pd.read_csv(f"datased_d{gd.dims[d]}_n{endSize}.csv")
        X = XY.iloc[:, :-1]
        Y = XY.iloc[:, -1]

        curDim = f'd = {gd.dims[d]}'
        #Set up rows for tables
        primAccRow = [curDim]
        dualAccRow = [curDim]
        primTimeRow = [curDim]
        dualTimeRow = [curDim]
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

            #Add data
            primAccRow.append(round(accPrimal,4))
            dualAccRow.append(round(accDual,4))
            primTimeRow.append(round(primalTime,4))
            dualTimeRow.append(round(dualTime,4))

            print(f"Dual: {numSamps} samples, {gd.dims[d]} features, {accDual} accuracy, {dualTime} time(s).")
            print(f"Primal: {numSamps} samples, {gd.dims[d]} features, {accPrimal} accuracy, {primalTime} time(s)")

        primalAccData.append(primAccRow)
        dualAccData.append(dualAccRow)  
        primalTimeData.append(primTimeRow)
        dualTimeData.append(dualTimeRow)

columns = ['Features(d)']
for i in range (len(gd.sampSize)):
    columns.append(f'n = {gd.sampSize[i]}')

fig, ax = plt.subplots(figsize=(8,4))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText = primalAccData, colLabels=columns, loc='center')
plt.title("Primal Accuracy", fontsize=16)
table.set_fontsize(12)
table.scale(1.2,1.2)

plt.figure(1)
fig, ax = plt.subplots(figsize=(8,4))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText = dualAccData, colLabels=columns, loc='center')
plt.title("Dual Accuracy", fontsize=16)
table.set_fontsize(12)
table.scale(1.2,1.2)

plt.figure(2)
fig, ax = plt.subplots(figsize=(8,4))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText = primalTimeData, colLabels=columns, loc='center')
plt.title("Primal Runtime (seconds)", fontsize=16)
table.set_fontsize(12)
table.scale(1.2,1.2)

plt.figure(3)
fig, ax = plt.subplots(figsize=(8,4))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText = dualTimeData, colLabels=columns, loc='center')
plt.title("Dual Runtime (seconds)", fontsize=16)
table.set_fontsize(12)
table.scale(1.2,1.2)

plt.show()
           

