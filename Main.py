import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from SVM import SVM
from sklearn.metrics import confusion_matrix
from matplotlib.pyplot import scatter


n_samp = 250
n_feat = 2
n_class = 2
C = [1 ** 5, 1 ** -5]

print()
print(
    "################################# Linear SVM on linearly separable data - Hard Margin ##########################################"
)
print()
# Building training and test datasets
print(
    "Generating a linearly separable dataset X_ls of {:d} samples, {:d} feature and {:d} classes.".format(
        n_samp, n_feat, n_class
    )
)
X_ls, y_ls = make_blobs(
    n_samples=n_samp,
    n_features=n_feat,
    centers=n_class,
    random_state=0,
    cluster_std=0.6,
)
y_ls[y_ls == 0] = -1

print("Splitting X_ls in training and test set with ratio 3:1.")
X_ls_train, X_ls_test, y_ls_train, y_ls_test = train_test_split(
    X_ls, y_ls, random_state=0
)
print(
    "The training set X_ls_train consists of {:d} rows and {:d} columns.".format(
        X_ls_train.shape[0], X_ls_train.shape[1]
    )
)

# Training the SVM model
print()
print("Training the SVM model...")
print()
ls_model = SVM()
ls_model.fit(X_ls_train, y_ls_train)

# Using the just trained model to classify test data
print()
print("Using the just trained model to classify test data.")
y_ls_pred = ls_model.predict(X_ls_test)
ls_cm = confusion_matrix(y_ls_test, y_ls_pred)
delta = np.sum(y_ls_pred == y_ls_test).astype(int)
accuracy_test_ls = np.mean(y_ls_pred == y_ls_test) * 100

print()
print("Confusion matrix:")
print(ls_cm)
print()
print(
    "{:d} well classified points out of {:d} points in test set.".format(
        delta, len(y_ls_test)
    )
)
print("The accuracy on test set is {0:.1f}%".format(accuracy_test_ls))

# Plotting dataset and hyperplane with margins
fig, ax = plt.subplots()
fig.suptitle("SVM - Hard Margin")
ax.set_title("{0:d} support vectors".format(len(ls_model.alpha_sv)))
fig.text(0.5, 0.04, "X[0]", ha="center")
fig.text(0.04, 0.5, "X[1]", va="center", rotation="vertical")

scatter = ax.scatter(
    X_ls_train[:, 0], X_ls_train[:, 1], marker="o", c=y_ls_train, cmap=plt.cm.Paired
)
ax.scatter(
    X_ls_test[:, 0], X_ls_test[:, 1], marker="o", c=y_ls_test, cmap=plt.cm.coolwarm
)
p1 = plt.scatter(
    ls_model.x_sv[:, 0],
    ls_model.x_sv[:, 1],
    s=100,
    linewidth=1,
    facecolors="none",
    edgecolors="k",
)
ls_model.plot_hyperplane(ax, X_ls_train)
ls_model.plot_margins(ax, X_ls)

plt.legend(loc="best", fontsize=8)

legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Class")
ax.add_artist(legend1)


legend_elements = ax.legend([p1], ["Support Vector"], scatterpoints=1, numpoints=1)

plt.show()

print()
print(
    "################################# Linear SVM on overlapping data - Soft Margin ##########################################"
)
print()

# Building training and test datasets
print(
    "Generating an overlapping dataset X_od of {:d} samples, {:d} feature and {:d} classes.".format(
        n_samp, n_feat, n_class
    )
)
X_od, y_od = make_blobs(
    n_samples=250, n_features=2, centers=2, random_state=0, cluster_std=1.6
)
y_od[y_od == 0] = -1

print("Splitting X_od in training and test set with ratio 3:1.")
X_od_train, X_od_test, y_od_train, y_od_test = train_test_split(
    X_od, y_od, random_state=0
)

# Preparing plotting elements
fig, axes = plt.subplots(
    1, 2, sharex="col", sharey="row", gridspec_kw={"hspace": 0, "wspace": 0},
)
fig.text(0.5, 0.04, "X[0]", ha="center")
fig.text(0.04, 0.5, "X[1]", va="center", rotation="vertical")
# TODO : controlla come hai calcolato b
# Instantiating useful variables
models = {"SVM_bigC": 2 ** 5, "SVM_smallC": 2 ** -5}
svm = {}

for ax, C in zip(axes.flatten(), models):
    print()
    print("Fitting SVM with C={0:.3f}".format(models[C]))
    print()
    svm["{0}".format(C)] = SVM(C=models[C])
    svm["{0}".format(C)].fit(X_od_train, y_od_train)

    # Using the just trained model to classify test data
    print()
    print("Using the just trained model to classify test data.")
    y_od_pred = svm["{0}".format(C)].predict(X_od_test)
    classified_samples = X_od_test[y_od_test == y_od_pred]
    classified_labels = y_od_test[y_od_test == y_od_pred]

    misclassified_samples = X_od_test[y_od_test != y_od_pred]
    misclassified_labels = y_od_test[y_od_test != y_od_pred]

    misclassified_samples_red = misclassified_samples[
        misclassified_labels == 1
    ]  # For plotting purposes
    misclassified_samples_blue = misclassified_samples[misclassified_labels == -1]

    misclassified_labels_red = misclassified_labels[misclassified_labels == 1]
    misclassified_labels_blue = misclassified_labels[misclassified_labels == -1]

    delta = np.sum(y_od_pred == y_od_test).astype(int)
    od_cm = confusion_matrix(y_od_test, y_od_pred)
    accuracy_test_od = np.mean(y_od_pred == y_od_test) * 100

    print()
    print("Confusion matrix:")
    print(od_cm)
    print()
    print(
        "{:d} well classified points out of {:d} points in test set.".format(
            delta, len(y_od_test)
        )
    )
    print("The accuracy on test set is {0:.1f}%".format(accuracy_test_od))

    ax.set_title("{0:d} support vectors".format(len(svm["{0}".format(C)].alpha_sv)))
    svm["{0}".format(C)].plot_hyperplane(ax, X_od)
    svm["{0}".format(C)].plot_margins(ax, X_od_train)
    ax.scatter(
        X_od_train[:, 0], X_od_train[:, 1], marker="o", c=y_od_train, cmap=plt.cm.Paired
    )
    p1 = ax.scatter(
        classified_samples[:, 0],
        classified_samples[:, 1],
        marker="o",
        c=classified_labels,
        cmap=plt.cm.coolwarm,
        label="Classified",
    )

    p2 = ax.scatter(
        misclassified_samples_red[:, 0],
        misclassified_samples_red[:, 1],
        s=100,
        marker="x",
        c="red",
        cmap=plt.cm.coolwarm,
        label="Misclassified",
    )
    ax.scatter(
        misclassified_samples_blue[:, 0],
        misclassified_samples_blue[:, 1],
        s=100,
        marker="x",
        c="blue",
        cmap=plt.cm.coolwarm,
        label="Misclassified",
    )
    p3 = ax.scatter(
        svm["{0}".format(C)].x_sv[:, 0],
        svm["{0}".format(C)].x_sv[:, 1],
        s=100,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
        label="Support Vector",
    )

fig.suptitle(
    "Comparing Soft-margin SVM C={0:.0f} vs C={1:.3f}".format(
        models["SVM_bigC"], models["SVM_smallC"], fontsize=24,
    )
)


legend1 = ax.legend(*p1.legend_elements(), loc="upper right", title="Class")
ax.add_artist(legend1)


legend_elements = ax.legend(
    [p1, p2, p3],
    ["Corr. Classified", "Misclassified", "Support Vector"],
    scatterpoints=1,
    numpoints=1,
    loc="lower left",
)


plt.show()
