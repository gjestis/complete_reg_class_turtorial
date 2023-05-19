## for data
import numpy as np
import pandas as pd

## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

## for statistical tests
import scipy
import statsmodels.formula.api as smf
import statsmodels.api as sm
import ppscore as pps

## for machine learning
from sklearn import preprocessing, impute, utils, linear_model, feature_selection, model_selection, metrics, \
    decomposition, cluster, ensemble
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    precision_recall_curve, roc_curve
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import KMeans
import imblearn as imb
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors


###############################################################################
#                       DATA ANALYSIS                                         #
###############################################################################
def utils_recognize_type(dtf, col, max_cat=20):
    '''

    :param dtf: dataframe - input data
    :param col: str - name of the column to analyze
    :param max_cat: num - max number of unique values to recognize a column as categorical
    :return: "cat" if the column is categorical, "dt" if datetime, "num" otherwise
    '''
    if (dtf[col].dtype == "O") | (dtf[col].nunique() < max_cat):
        return "cat"
    elif dtf[col].dtype in ['datetime64[ns]', '<M8[ns]']:
        return "dt"
    else:
        return "num"


def dtf_overview(dtf, max_cat=20, figsize=(10, 5)):
    '''

    :param dtf: dataframe - input data
    :param max_cat: num - mininum number of recognize column type
    :param figsize: size of plot
    :return:
    '''
    ## recognize column type
    dic_cols = {col: utils_recognize_type(dtf, col, max_cat=max_cat) for col in dtf.columns}

    ## print info
    len_dtf = len(dtf)
    print("Shape:", dtf.shape)
    print("-----------------")
    for col in dtf.columns:
        info = col + " --> Type:" + dic_cols[col]
        info = info + " | Nas: " + str(dtf[col].isna().sum()) + "(" + str(int(dtf[col].isna().mean() * 100)) + "%)"
        if dic_cols[col] == "cat":
            info = info + " | Categories: " + str(dtf[col].nunique())
        elif dic_cols[col] == "dt":
            info = info + " | Range: " + "({x})-({y})".format(x=str(dtf[col].min()), y=str(dtf[col].max()))
        else:
            info = info + " | Min-Max: " + "({x})-({y})".format(x=str(int(dtf[col].min())), y=str(int(dtf[col].max())))
        if dtf[col].nunique() == len_dtf:
            info = info + " | Possible Primary Key/Unique Identifier"
        print(info)

    ## plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    heatmap = dtf.isnull()
    for k, v in dic_cols.items():
        if v == "num":
            heatmap[k] = heatmap[k].apply(lambda x: 0.5 if x is False else 1)
        else:
            heatmap[k] = heatmap[k].apply(lambda x: 0 if x is False else 1)
    sns.heatmap(heatmap, vmin=0, vmax=1, cbar=False, ax=ax).set_title('Dataset Overview')
    # plt.setp(plt.xticks()[1], rotation=0)
    plt.show()

    ## add legend
    print("\033[1;37;40m Categerocial \033[m", "\033[1;30;41m Numerical/DateTime \033[m", "\033[1;30;47m NaN \033[m")


def freqdist_plot(dtf, x, max_cat=20, top=None, show_perc=True, bins=100, quantile_breaks=(0, 10), box_logscale=False,
                  figsize=(10, 5)):
    try:
        ## cat --> freq
        if utils_recognize_type(dtf, x, max_cat) == "cat":
            ax = dtf[x].value_counts().head(top).sort_values().plot(kind="barh", figsize=figsize)
            totals = []
            for i in ax.patches:
                totals.append(i.get_width())
            if show_perc == False:
                for i in ax.patches:
                    ax.text(i.get_width() + .3, i.get_y() + .20, str(i.get_width()), fontsize=10, color='black')
            else:
                total = sum(totals)
                for i in ax.patches:
                    ax.text(i.get_width() + .3, i.get_y() + .20, str(round((i.get_width() / total) * 100, 2)) + '%',
                            fontsize=10, color='black')
            ax.grid(axis="x")
            plt.suptitle(x, fontsize=20)
            plt.show()

        ## num --> density
        else:
            fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=figsize)
            fig.suptitle(x, fontsize=20)
            ### distribution
            ax[0].title.set_text('distribution')
            variable = dtf[x].fillna(dtf[x].mean())
            print("Q1 quantile of Y : ", np.quantile(variable, .25))
            print("Q2 quantile (median) of Y: ", np.quantile(variable, .50))
            print("Q3 quantile of Y : ", np.quantile(variable, .75))
            breaks = np.quantile(variable, q=np.linspace(0, 1, 11))
            variable = variable[(variable > breaks[quantile_breaks[0]]) & (variable < breaks[quantile_breaks[1]])]
            sns.distplot(variable, hist=True, kde=True, kde_kws={"shade": True}, ax=ax[0])
            des = dtf[x].describe()
            ax[0].axvline(des["25%"], ls='--')
            ax[0].axvline(des["mean"], ls='--')
            ax[0].axvline(des["75%"], ls='--')
            ax[0].grid(True)
            des = round(des, 2).apply(lambda x: str(x))
            box = '\n'.join(("min: " + des["min"], "25%: " + des["25%"], "mean: " + des["mean"], "75%: " + des["75%"],
                             "max: " + des["max"]))
            ax[0].text(0.95, 0.95, box, transform=ax[0].transAxes, fontsize=10, va='top', ha="right",
                       bbox=dict(boxstyle='round', facecolor='white', alpha=1))
            ### boxplot
            if box_logscale == True:
                ax[1].title.set_text('outliers (log scale)')
                tmp_dtf = pd.DataFrame(dtf[x])
                tmp_dtf[x] = np.log(tmp_dtf[x])
                tmp_dtf.boxplot(column=x, ax=ax[1])
            else:
                ax[1].title.set_text('outliers')
                dtf.boxplot(column=x, ax=ax[1])
            plt.show()

    except Exception as e:
        print("--- got error ---")
        print(e)  #


def bivariate_plot(dtf, x, y, max_cat=20, figsize=(10, 5)):
    try:
        ## num vs num --> stacked + scatter with density
        if (utils_recognize_type(dtf, x, max_cat) == "num") & (utils_recognize_type(dtf, y, max_cat) == "num"):
            ### stacked
            dtf_noNan = dtf[dtf[x].notnull()]  # can't have nan
            breaks = np.quantile(dtf_noNan[x], q=np.linspace(0, 1, 11))
            groups = dtf_noNan.groupby([pd.cut(dtf_noNan[x], bins=breaks, duplicates='drop')])[y].agg(
                ['mean', 'median', 'size'])
            fig, ax = plt.subplots(figsize=figsize)
            fig.suptitle(x + "   vs   " + y, fontsize=20)
            groups[["mean", "median"]].plot(kind="line", ax=ax)
            groups["size"].plot(kind="bar", ax=ax, rot=45, secondary_y=True, color="grey", alpha=0.3, grid=True)
            ax.set(ylabel=y)
            ax.right_ax.set_ylabel("Observazions in each bin")
            plt.show()
            ### joint plot
            sns.jointplot(x=x, y=y, data=dtf, dropna=True, kind='reg', height=int((figsize[0] + figsize[1]) / 2))
            plt.show()

        ## cat vs cat --> hist count + hist %
        elif (utils_recognize_type(dtf, x, max_cat) == "cat") & (utils_recognize_type(dtf, y, max_cat) == "cat"):
            fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=figsize)
            fig.suptitle(x + "   vs   " + y, fontsize=20)
            ### count
            ax[0].title.set_text('count')
            order = dtf.groupby(x)[y].count().index.tolist()
            sns.countplot(x=x, hue=y, data=dtf, order=order, ax=ax[0])
            ax[0].grid(True)
            ### percentage
            ax[1].title.set_text('percentage')
            sns.histplot(x=x, hue=y, data=dtf, stat="percent", multiple="dodge", ax=ax[1])
            ax[1].grid(True)
            ### fix figure
            plt.close(2)
            plt.close(3)
            plt.show()

        ## num vs cat --> density + stacked + boxplot
        else:
            if (utils_recognize_type(dtf, x, max_cat) == "cat"):
                cat, num = x, y
            else:
                cat, num = y, x
            fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=figsize)
            fig.suptitle(x + "   vs   " + y, fontsize=20)
            ### distribution
            ax[0].title.set_text('density')
            for i in sorted(dtf[cat].unique()):
                sns.distplot(dtf[dtf[cat] == i][num], hist=False, label=i, ax=ax[0])
            ax[0].grid(True)
            ax[0].legend(loc="upper right", title=y)
            ### stacked
            dtf_noNan = dtf[dtf[num].notnull()]  # can't have nan
            ax[1].title.set_text('bins')
            breaks = np.quantile(dtf_noNan[num], q=np.linspace(0, 1, 11))
            tmp = dtf_noNan.groupby([cat, pd.cut(dtf_noNan[num], breaks, duplicates='drop')]).size().unstack().T
            tmp = tmp[dtf_noNan[cat].unique()]
            tmp["tot"] = tmp.sum(axis=1)
            for col in tmp.drop("tot", axis=1).columns:
                tmp[col] = tmp[col] / tmp["tot"]
            tmp.drop("tot", axis=1)[sorted(dtf[cat].unique())].plot(kind='bar', stacked=True, ax=ax[1], legend=False,
                                                                    grid=True)
            plt.close(2)
            plt.close(3)
            plt.show()
            ### boxplot
            sns.catplot(x=cat, y=num, data=dtf, kind="box", order=sorted(dtf[cat].unique()))
            plt.title("outliers")
            plt.grid(True)
            plt.show()

    except Exception as e:
        print("--- got error ---")
        print(e)


def cross_distributions(dtf, x1, x2, y, max_cat=20, show_only_min_clas=False, min_class_num=1, figsize=(10, 5)):
    ## Y cat
    if utils_recognize_type(dtf, y, max_cat) == "cat":

        ### cat vs cat --> contingency table
        if (utils_recognize_type(dtf, x1, max_cat) == "cat") & (utils_recognize_type(dtf, x2, max_cat) == "cat"):
            cont_table = pd.crosstab(index=dtf[x1], columns=dtf[x2], values=dtf[y], aggfunc="sum")
            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(cont_table, annot=True, fmt='.0f', cmap="YlGnBu", ax=ax, linewidths=.5).set_title(
                x1 + '  vs  ' + x2 + '  (filter: ' + y + ')')

        ### num vs num --> scatter with hue
        elif (utils_recognize_type(dtf, x1, max_cat) == "num") & (utils_recognize_type(dtf, x2, max_cat) == "num"):
            if show_only_min_clas == False:
                sns.lmplot(x=x1, y=x2, data=dtf, hue=y, height=figsize[1])
            else:
                sns.lmplot(x=x1, y=x2, data=dtf[dtf[y] == min_class_num], hue=y, fit_reg=True, height=figsize[1])

        ### num vs cat --> boxplot with hue
        else:
            if (utils_recognize_type(dtf, x1, max_cat) == "cat"):
                cat, num = x1, x2
            else:
                cat, num = x2, x1
            fig, ax = plt.subplots(figsize=figsize)
            sns.boxplot(x=cat, y=num, hue=y, data=dtf, ax=ax).set_title(x1 + '  vs  ' + x2 + '  (filter: ' + y + ')')
            ax.grid(True)

    ## Y num
    else:
        ### all num --> 3D scatter plot
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')
        plot3d = ax.scatter(xs=dtf[x1], ys=dtf[x2], zs=dtf[y], c=dtf[y], cmap='inferno', linewidth=0.5)
        fig.colorbar(plot3d, shrink=0.5, aspect=5, label=y)
        ax.set(xlabel=x1, ylabel=x2, zlabel=y)
        plt.show()


###############################################################################
#                         CORRELATION                                         #
###############################################################################
'''
Computes the correlation matrix.
:parameter
    :param dtf: dataframe - input data
    :param method: str - "pearson" (numeric), "spearman" (categorical), "kendall"
    :param negative: bool - if False it takes the absolute values of correlation
    :param lst_filters: list - filter rows to show
    :param annotation: logic - plot setting
'''


def corr_matrix(dtf, method="pearson", negative=True, annotation=True, figsize=(10, 5)):
    ## factorize
    dtf_corr = dtf.copy()
    for col in dtf_corr.columns:
        if dtf_corr[col].dtype == "O":
            # Mapping each category to a unique number so that corr is possible
            print("--- WARNING: Factorizing", dtf_corr[col].nunique(), "labels of", col, "---")
            dtf_corr[col] = dtf_corr[col].factorize(sort=True)[0]
    ## corr matrix
    dtf_corr = dtf_corr.corr(method=method)
    dtf_corr = dtf_corr if negative is True else dtf_corr.abs()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(dtf_corr, annot=annotation, fmt='.2f', cmap="YlGnBu", ax=ax, cbar=True, linewidths=0.5)
    plt.title(method + " correlation")
    return dtf_corr


'''
Computes the pps matrix.
'''


def pps_matrix(dtf, annotation=True, figsize=(10, 5)):
    dtf_pps = pps.matrix(dtf)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(dtf_pps, vmin=0., vmax=1., annot=annotation, ax=ax, fmt='.2f', cmap="YlGnBu", cbar=True, linewidths=0.5)
    plt.title("predictive power score")
    return dtf_pps


'''
Computes correlation/dependancy and p-value (prob of happening something different than what observed in the sample)
'''


def test_corr(dtf, x, y, max_cat=20):
    ## num vs num --> pearson
    if (utils_recognize_type(dtf, x, max_cat) == "num") & (utils_recognize_type(dtf, y, max_cat) == "num"):
        dtf_noNan = dtf[dtf[x].notnull()]  # can't have nan
        coeff, p = scipy.stats.pearsonr(dtf_noNan[x], dtf_noNan[y])
        coeff, p = round(coeff, 3), round(p, 3)
        conclusion = "Significant" if p < 0.05 else "Non-Significant"
        print("Pearson Correlation:", coeff, conclusion, "(p-value: " + str(p) + ")")

    ## cat vs cat --> cramer (chiquadro)
    elif (utils_recognize_type(dtf, x, max_cat) == "cat") & (utils_recognize_type(dtf, y, max_cat) == "cat"):
        cont_table = pd.crosstab(index=dtf[x], columns=dtf[y])
        chi2_test = scipy.stats.chi2_contingency(cont_table)
        chi2, p = chi2_test[0], chi2_test[1]
        n = cont_table.sum().sum()
        phi2 = chi2 / n
        r, k = cont_table.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        coeff = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
        coeff, p = round(coeff, 3), round(p, 3)
        conclusion = "Significant" if p < 0.05 else "Non-Significant"
        print("Cramer Correlation:", coeff, conclusion, "(p-value: " + str(p) + ")")

    ## num vs cat --> 1way anova (f: the means of the groups are different)
    else:
        if (utils_recognize_type(dtf, x, max_cat) == "cat"):
            cat, num = x, y
        else:
            cat, num = y, x
        model = smf.ols(num + ' ~ ' + cat, data=dtf).fit()
        table = sm.stats.anova_lm(model)
        p = table["PR(>F)"][0]
        coeff, p = None, round(p, 3)
        conclusion = "Correlated" if p < 0.05 else "Non-Correlated"
        print("Anova F: the variables are", conclusion, "(p-value: " + str(p) + ")")

    return coeff,


###############################################################################
#                   MODEL DESIGN & TESTING - REGRESSION                       #
###############################################################################

def evaluate_regr_model(y_test, predicted, figsize=(25, 5)):
    ## Kpi
    print("R2 (explained variance):", metrics.r2_score(y_test, predicted))
    print("Mean Absolute Perc Error (Σ(|y-pred|/y)/n):", np.mean(np.abs((y_test - predicted) / predicted)))
    print("Mean Absolute Error (Σ|y-pred|/n):", metrics.mean_absolute_error(y_test, predicted))
    print("Root Mean Squared Error (sqrt(Σ(y-pred)^2/n)):", np.sqrt(metrics.mean_squared_error(y_test, predicted)))
    print("Mean Squared Error Σ(y-pred)^2/n):", metrics.mean_squared_error(y_test, predicted))

    ## residuals
    residuals = y_test - predicted
    max_error = max(residuals) if abs(max(residuals)) > abs(min(residuals)) else min(residuals)
    max_idx = list(residuals).index(max(residuals)) if abs(max(residuals)) > abs(min(residuals)) else list(
        residuals).index(min(residuals))
    max_true, max_pred = y_test[max_idx], predicted[max_idx]
    print("Max Error:", "{:,.0f}".format(max_error))

    ## Plot predicted vs true
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize)
    from statsmodels.graphics.api import abline_plot
    ax[0].scatter(predicted, y_test, color="black")
    abline_plot(intercept=0, slope=1, color="red", ax=ax[0])
    ax[0].vlines(x=max_pred, ymin=max_true, ymax=max_true - max_error, color='red', linestyle='--', alpha=0.7,
                 label="max error")
    ax[0].grid(True)
    ax[0].set(xlabel="Predicted", ylabel="True", title="Predicted vs True")
    ax[0].legend()

    ## Plot predicted vs residuals
    ax[1].scatter(predicted, residuals, color="red")
    ax[1].vlines(x=max_pred, ymin=0, ymax=max_error, color='black', linestyle='--', alpha=0.7, label="max error")
    ax[1].grid(True)
    ax[1].set(xlabel="Predicted", ylabel="Residuals", title="Predicted vs Residuals")
    ax[1].hlines(y=0, xmin=np.min(predicted), xmax=np.max(predicted))
    ax[1].legend()

    ## Plot residuals distribution
    sns.distplot(residuals, color="red", hist=True, kde=True, kde_kws={"shade": True}, ax=ax[2],
                 label="mean = " + "{:,.0f}".format(np.mean(residuals)))
    ax[2].grid(True)
    ax[2].set(yticks=[], yticklabels=[], title="Residuals distribution")
    plt.show()


###############################################################################
#                   MODEL DESIGN & TESTING - CLASSIFICATION                   #
###############################################################################

def evaluate_classif_model(y_test, models_from_train, name_of_model, X_test, show_thresholds=True, figsize=(15, 10)):
    # Print
    print("Results for :", name_of_model)

    # Get classes
    classes = np.unique(y_test)

    # Define plot
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize)

    # Get pred from model
    pred = models_from_train[name_of_model].predict(X_test)

    # Confusion matrix
    cm = metrics.confusion_matrix(y_test, pred, labels=classes)
    sns.heatmap(cm / np.sum(cm), annot=True, fmt='.2%', cmap='Blues', ax=ax[0])
    ax[0].set(xlabel="Pred", ylabel="True", title="Confusion matrix")
    ax[0].set_yticklabels(labels=classes, rotation=0)

    # Accuracy score
    print('Accuracy:', accuracy_score(y_test, pred))

    # Precision
    precision = cm[1][1] / (cm[0][1] + cm[1][1])
    print('Precision:', precision)

    # Recall
    recall = cm[1][1] / (cm[1][0] + cm[1][1])
    print('Recall:', recall)

    # F1_score
    print('F1:', f1_score(y_test, pred))

    # Obtain prediction probabilities
    pred = models_from_train[name_of_model].predict_proba(X_test)
    pred = [p[1] for p in pred]

    # Plot ROC
    fpr, tpr, thresholds = roc_curve(y_test, pred)
    ax[1].plot(fpr, tpr, color='darkorange', lw=3, label='ROC AUC = %0.3f' % metrics.auc(fpr, tpr))
    ax[1].plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    ax[1].hlines(y=recall, xmin=-0.05, xmax=1 - cm[0, 0] / (cm[0, 0] + cm[0, 1]), color='red', linestyle='--',
                 alpha=0.7, label="chosen threshold")
    ax[1].vlines(x=1 - cm[0, 0] / (cm[0, 0] + cm[0, 1]), ymin=0, ymax=recall, color='red', linestyle='--', alpha=0.7)
    ax[1].set(xlim=[-0.05, 1], ylim=[0.0, 1.05], xlabel='False Positive Rate', ylabel="True Positive Rate (Recall)",
              title="Receiver operating characteristic")
    ax[1].legend(loc="lower right")
    ax[1].grid(True)
    if show_thresholds is True:
        thres_in_plot = []
        for i, t in enumerate(thresholds):
            t = np.round(t, 1)
            if t not in thres_in_plot:
                ax[1].annotate(t, xy=(fpr[i], tpr[i]), xytext=(fpr[i], tpr[i]), textcoords='offset points', ha='left',
                               va='bottom')
                thres_in_plot.append(t)

    # Plot Precision-recall curve
    ## Plot precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, pred)
    ax[2].plot(recalls, precisions, color='darkorange', lw=3, label='PR AUC = %0.3f' % metrics.auc(recalls, precisions))
    ax[2].plot([0, 1], [(cm[1, 0] + cm[1, 0]) / len(y_test), (cm[1, 0] + cm[1, 0]) / len(y_test)], linestyle='--',
               color='navy', lw=3)
    ax[2].hlines(y=precision, xmin=0, xmax=recall, color='red', linestyle='--', alpha=0.7, label="chosen threshold")
    ax[2].vlines(x=recall, ymin=0, ymax=precision, color='red', linestyle='--', alpha=0.7)
    ax[2].set(xlim=[0.0, 1.05], ylim=[0.0, 1.05], xlabel='Recall', ylabel="Precision", title="Precision-Recall curve")
    ax[2].legend(loc="lower left")
    ax[2].grid(True)
    if show_thresholds is True:
        thres_in_plot = []
        for i, t in enumerate(thresholds):
            t = np.round(t, 1)
            if t not in thres_in_plot:
                ax[2].annotate(np.round(t, 1), xy=(recalls[i], precisions[i]), xytext=(recalls[i], precisions[i]),
                               textcoords='offset points', ha='right', va='bottom')
                thres_in_plot.append(t)

    plt.show()


def tune_classif_model(X_train, y_train, model_base=None, param_dic=None, scoring="f1", searchtype="RandomSearch",
                       n_iter=1000, cv=10, figsize=(10, 5)):
    ## params
    model_base = ensemble.GradientBoostingClassifier() if model_base is None else model_base
    param_dic = {'learning_rate': [0.15, 0.1, 0.05, 0.01, 0.005, 0.001],
                 'n_estimators': [100, 250, 500, 750, 1000, 1250, 1500, 1750],
                 'max_depth': [2, 3, 4, 5, 6, 7]} if param_dic is None else param_dic
    dic_scores = {'accuracy': metrics.make_scorer(metrics.accuracy_score),
                  'precision': metrics.make_scorer(metrics.precision_score),
                  'recall': metrics.make_scorer(metrics.recall_score), 'f1': metrics.make_scorer(metrics.f1_score)}

    ## Search
    print("---", searchtype, "---")
    if searchtype == "RandomSearch":
        random_search = model_selection.RandomizedSearchCV(model_base, param_distributions=param_dic, n_iter=n_iter,
                                                           scoring=dic_scores, refit=scoring).fit(X_train, y_train)
        print("Best Model parameters:", random_search.best_params_)
        print("Best Model " + scoring + ":", round(random_search.best_score_, 2))
        model = random_search.best_estimator_

    elif searchtype == "GridSearch":
        grid_search = model_selection.GridSearchCV(model_base, param_dic, scoring=dic_scores, refit=scoring).fit(
            X_train, y_train)
        print("Best Model parameters:", grid_search.best_params_)
        print("Best Model mean " + scoring + ":", round(grid_search.best_score_, 2))
        model = grid_search.best_estimator_

    ## K fold validation
    print("")
    print("--- Kfold Validation ---")
    Kfold_base = model_selection.cross_validate(estimator=model_base, X=X_train, y=y_train, cv=cv, scoring=dic_scores)
    Kfold_model = model_selection.cross_validate(estimator=model, X=X_train, y=y_train, cv=cv, scoring=dic_scores)
    for score in dic_scores.keys():
        print(score, "mean - base model:", round(Kfold_base["test_" + score].mean(), 2), " --> best model:",
              round(Kfold_model["test_" + score].mean()))
    utils_kfold_roc(model, X_train, y_train, cv=cv, figsize=figsize)

    ## Threshold analysis
    print("")
    print("--- Threshold Selection ---")
    utils_threshold_selection(model, X_train, y_train, figsize=figsize)

    return model


###############################################################################
#                  FEATURES SELECTION                                         #
###############################################################################


def features_importance(X, y, X_names, model=None, task="classification", figsize=(10, 10)):
    ## model
    if model is None:
        if task == "classification":
            model = ensemble.GradientBoostingClassifier()
        elif task == "regression":
            model = ensemble.GradientBoostingRegressor()
    model.fit(X, y)
    print("--- model used ---")
    print(model)

    ## importance dtf
    importances = model.feature_importances_
    dtf_importances = pd.DataFrame({"IMPORTANCE": importances, "VARIABLE": X_names}).sort_values("IMPORTANCE",
                                                                                                 ascending=False)
    dtf_importances['cumsum'] = dtf_importances['IMPORTANCE'].cumsum(axis=0)
    dtf_importances = dtf_importances.set_index("VARIABLE")

    ## plot
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=figsize)
    fig.suptitle("Features Importance", fontsize=20)
    ax[0].title.set_text('variables')
    dtf_importances[["IMPORTANCE"]].sort_values(by="IMPORTANCE").plot(kind="barh", legend=False, ax=ax[0]).grid(
        axis="x")
    ax[0].set(ylabel="")
    ax[1].title.set_text('cumulative')
    dtf_importances[["cumsum"]].plot(kind="line", linewidth=4, legend=False, ax=ax[1])
    ax[1].set(xlabel="", xticks=np.arange(len(dtf_importances)), xticklabels=dtf_importances.index)
    plt.xticks(rotation=70)
    plt.grid(axis='both')
    plt.show()
    return dtf_importances.reset_index()


###############################################################################
#                       PREPROCESSING                                         #
###############################################################################

def rebalance(dtf, y, balance=None, method="random", replace=True, size=1):
    '''

    :param dtf: dataframe - feature matrix dtf
    :param y: str - column to use as target
    :param balance: str - "up", "down", if None just prints some stats
    :param method: str - "random" for sklearn or "knn" for imblearn
    :param replace:
    :param size: num - 1 for same size of the other class, 0.5 for half of the other class
    :return: rebalanced dtf
    '''
    ## check
    print("--- situation ---")
    check = dtf[y].value_counts().to_frame()
    check["%"] = (check[y] / check[y].sum() * 100).round(1).astype(str) + '%'
    print(check)
    print("tot:", check[y].sum())

    ## sklearn
    if balance is not None and method == "random":
        ### set the major and minor class
        major = check.index[0]
        minor = check.index[1]
        dtf_major = dtf[dtf[y] == major]
        dtf_minor = dtf[dtf[y] == minor]

        ### up-sampling
        if balance == "up":
            print("--- upsampling ---")
            print("   randomly replicate observations from the minority class (Overfitting risk)")
            dtf_minor = utils.resample(dtf_minor, replace=replace, random_state=123,
                                       n_samples=int(size * len(dtf_major)))
            dtf_balanced = pd.concat([dtf_major, dtf_minor])

        ### down-sampling
        elif balance == "down":
            print("--- downsampling ---")
            print("   randomly remove observations of the majority class (Underfitting risk)")
            dtf_minor = utils.resample(dtf_minor, replace=replace, random_state=123,
                                       n_samples=int(size * len(dtf_major)))
            dtf_balanced = pd.concat([dtf_major, dtf_minor])

    ## imblearn
    if balance is not None and method == "knn":
        ### up-sampling
        if balance == "up":
            print("--- upsampling ---")
            print("   create synthetic observations from the minority class (Distortion risk)")
            smote = imb.over_sampling.SMOTE(random_state=123)
            dtf_balanced, y_values = smote.fit_resample(dtf.drop(y, axis=1), y=dtf[y])
            dtf_balanced[y] = y_values

        ### down-sampling
        elif balance == "down":
            print("--- downsampling ---")
            print("   select observations that don't affect performance (Underfitting risk)")
            nn = imb.under_sampling.CondensedNearestNeighbour(random_state=123)
            dtf_balanced, y_values = nn.fit_resample(dtf.drop(y, axis=1), y=dtf[y])
            dtf_balanced[y] = y_values

    ## check rebalance
    if balance is not None:
        print("--- new situation ---")
        check = dtf_balanced[y].value_counts().to_frame()
        check["%"] = (check[y] / check[y].sum() * 100).round(1).astype(str) + '%'
        print(check)
        print("tot:", check[y].sum())
        return dtf_balanced


###############################################################################
#                       CLUSTERING (UNSUPERVISED)                             #
###############################################################################
def find_best_k(X, max_k=10, plot=True, elbow=True):
    # ELBOW METHOD
    if elbow is True:
        ## iterations
        distortions = []
        for i in range(1, max_k + 1):
            if len(X) >= i:
                model = cluster.KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
                model.fit(X)
                distortions.append(model.inertia_)

        ## best k: the lowest second derivative
        k = [i * 100 for i in np.diff(distortions, 2)].index(min([i * 100 for i in np.diff(distortions, 2)]))
        ## plot
        if plot is True:
            fig, ax = plt.subplots()
            ax.plot(range(1, len(distortions) + 1), distortions)
            ax.axvline(k, ls='--', color="red", label="k = " + str(k))
            ax.set(title='The Elbow Method', xlabel='Number of clusters', ylabel="Distortion")
            ax.legend()
            ax.grid(True)
            plt.show()
        return print("Best number of cluster :", k)

    # CILHOUETTE METHODE
    if elbow is False:
        ## iterations (canidate values for nr of cluster)
        parameters = [*range(2, max_k + 1, 1)]
        ## instantiating ParameterGrid, pass number of clusters as input
        parameter_grid = ParameterGrid({'n_clusters': parameters})
        best_score = -1
        kmeans_model = cluster.KMeans(init='k-means++', max_iter=300, n_init=10, random_state=0)
        distortions = []
        silhouette_scores = []
        ## evaluation based on silhouette_score
        for p in parameter_grid:
            kmeans_model.set_params(**p)  # set current hyper parameter
            kmeans_model.fit(X)  # fit model on dataset, this will find clusters based on parameter p
            ss = metrics.silhouette_score(X, kmeans_model.labels_)  # calculate silhouette_score
            silhouette_scores += [ss]
            print('Parameter:', p, 'Score', ss)
            print("------------")
            # check p which has the best score
            if ss > best_score:
                best_score = ss
                best_grid = p
        # plotting silhouette score
        plt.bar(range(len(silhouette_scores)), list(silhouette_scores), align='center', color='#722f59', width=0.5)
        plt.xticks(range(len(silhouette_scores)), list(parameters))
        plt.title('Silhouette Score', fontweight='bold')
        plt.xlabel('Number of Clusters')
        plt.show()
        return print("Best number of clusters :", best_grid['n_clusters'])


def utils_plot_cluster(dtf, x1, x2, th_centroids=None, figsize=(10, 5)):
    ## plot points and real centroids
    fig, ax = plt.subplots(figsize=figsize)
    k = dtf["cluster"].nunique()
    sns.scatterplot(x=x1, y=x2, data=dtf, palette=sns.color_palette("bright", k),
                    hue='cluster', size="centroids", size_order=[1, 0],
                    legend="brief", ax=ax).set_title('Clustering (k=' + str(k) + ')')

    ## plot theoretical centroids
    if th_centroids is not None:
        ax.scatter(th_centroids[:, dtf.columns.tolist().index(x1)],
                   th_centroids[:, dtf.columns.tolist().index(x2)],
                   s=50, c='black', marker="x")

    ## plot links from points to real centroids
    # if plot_links is True:
    #     centroids_idx = dtf[dtf["centroids"]==1].index
    #     colors = itertools.cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    #     for k, col in zip(range(k), colors):
    #         class_members = dtf["cluster"].values == k
    #         cluster_center = dtf[[x1,x2]].values[centroids_idx[k]]
    #         plt.plot(dtf[[x1,x2]].values[class_members, 0], dtf[[x1,x2]].values[class_members, 1], col + '.')
    #         plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
    #         for x in dtf[[x1,x2]].values[class_members]:
    #             plt.plot([cluster_center[0], x[0]],
    #                      [cluster_center[1], x[1]],
    #                      col)

    ax.grid(True)
    plt.show()


def fit_ml_cluster(X, model=None, k=None, lst_2Dplot=None, figsize=(10, 5)):
    ## model
    if (model is None) and (k is None):
        model = cluster.AffinityPropagation()
        print("--- k not defined: using Affinity Propagation ---")
    elif (model is None) and (k is not None):
        model = cluster.KMeans(n_clusters=k, init='k-means++')
        print("---", "k=" + str(k) + ": using k-means ---")

    ## clustering
    dtf_X = X.copy()
    dtf_X["cluster"] = model.fit_predict(X)
    col_clusters = pd.DataFrame(dtf_X["cluster"])
    k = dtf_X["cluster"].nunique()
    print("--- found", k, "clusters ---")
    print(dtf_X.groupby("cluster")["cluster"].count().sort_values(ascending=False))

    ## find real centroids
    closest, distances = scipy.cluster.vq.vq(model.cluster_centers_, dtf_X.drop("cluster", axis=1).values)
    dtf_X["centroids"] = 0
    for i in closest:
        dtf_X["centroids"].iloc[i] = 1

    ## plot
    if (lst_2Dplot is not None) or (X.shape[1] == 2):
        lst_2Dplot = X.columns.tolist() if lst_2Dplot is None else lst_2Dplot
        th_centroids = model.cluster_centers_ if "KMeans" in str(model) else None
        utils_plot_cluster(dtf_X, x1=lst_2Dplot[0], x2=lst_2Dplot[1], th_centroids=th_centroids, figsize=figsize)

    return model, dtf_X, col_clusters


def inter_clusters(df, col_with_clus):
    # Create hist for each cluster
    for c in df:
        grid = sns.FacetGrid(df, col=col_with_clus)
        grid.map(plt.hist, c)
        plt.show

    # Get statistics for each cluster
    stat = df.groupby(col_with_clus).describe()
    return stat


def visual_clusters_pca(df_scaled, best_model_labels, colors, names):
    # Pca
    dist = 1 - cosine_similarity(df_scaled)
    pca = PCA(2)
    pca.fit(dist)
    X_PCA = pca.transform(dist)
    print("Shape after PCA red : ", X_PCA.shape)
    x, y = X_PCA[:, 0], X_PCA[:, 1]
    # Creating df
    df_pc = pd.DataFrame({'x': x, 'y': y, 'label': best_model_labels})
    groups = df_pc.groupby('label')
    # Plotting
    fig, ax = plt.subplots(figsize=(20, 13))
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=5,
                color=colors[name], label=names[name], mec='none')
        ax.set_aspect('auto')
        ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        ax.tick_params(axis='y', which='both', left='off', top='off', labelleft='off')

    ax.legend()
    ax.set_title("Customers Segmentation based on their Credit Card usage behaviour.")
    plt.show()


def hierar_clust_dendo(data_scaled, link):
    plt.figure(figsize=(10, 7))
    plt.title("Dendrograms")
    plt.ylabel('Euclidean distances')
    dend = shc.dendrogram(shc.linkage(data_scaled, method=link))


def dbscan_with_parameter_tun(df):
    # Define minPts
    minPts = df.shape[1] * 2
    # Fit knn model
    nbrs = NearestNeighbors(n_neighbors=minPts).fit(df)
    # Find the k-neighbors of a point
    neigh_dist, neigh_ind = nbrs.kneighbors(df)
    # Sort the neighbor distances (lengths to points) in ascending order
    # axis = 0 represents sort along first axis i.e. sort along row
    sort_neigh_dist = np.sort(neigh_dist, axis=0)
    # Plot the distances
    k_dist = sort_neigh_dist[:, minPts - 1]
    plt.plot(k_dist)
    plt.ylabel("k-NN distance")
    plt.xlabel("Sorted observations " + str(minPts) + "th NN")
    plt.show()
    return minPts


###############################################################################
#                         TS ANALYSIS                                         #
###############################################################################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#from lightgbm import LGBMRegressor
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import t
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_absolute_error

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error



def decompose_ts(df, share_type, samples=250, period=24):
    if samples == 'all':
        #decomposing all time series timestamps
        res = seasonal_decompose(df[share_type].values, period=period)
    else:
        #decomposing a sample of the time series
        res = seasonal_decompose(df[share_type].values[-samples:], period=period)

    observed = res.observed
    trend = res.trend
    seasonal = res.seasonal
    residual = res.resid

    #plot the complete time series
    fig, axs = plt.subplots(4, figsize=(16,10))
    axs[0].set_title('OBSERVED', fontsize=16)
    axs[0].plot(observed)
    axs[0].grid()

    #plot the trend of the time series
    axs[1].set_title('TREND', fontsize=16)
    axs[1].plot(trend)
    axs[1].grid()

    #plot the seasonality of the time series. Period=24 daily seasonality | Period=24*7 weekly seasonality.
    axs[2].set_title('SEASONALITY', fontsize=16)
    axs[2].plot(seasonal)
    axs[2].grid()

    #plot the noise of the time series
    axs[3].set_title('NOISE', fontsize=16)
    axs[3].plot(residual)
    axs[3].scatter(y=residual, x=range(len(residual)), alpha=0.5)
    axs[3].grid()

    plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error

def train_time_series_with_avg(df,target_var, horizon=10):
    # Isolate the target variable
    y = df[target_var]

    # Create train and test
    y_train, y_test = y.iloc[:-horizon], y.iloc[-horizon:]
    print("Start date for training:", y_train.index[0])
    print("End date for training:", y_train.index[-1])
    print("Start date for test:", y_test.index[0])
    print("End date for test:", y_test.index[-1])

    # Calculate historical average
    historical_average = y_train.mean()
    print("The historical average is:", historical_average)

    # Create an array of historical average values for the length of the validation set
    predictions = np.full_like(y_test, fill_value=historical_average)

    # Calculate MAE
    mae = np.round(mean_absolute_error(y_test, predictions), 3)

    # Plot reality vs prediction for the last week of the dataset
    fig = plt.figure(figsize=(16, 8))
    plt.title(f'Real vs Prediction - MAE {mae}', fontsize=20)
    plt.plot(y_test, color='red')
    plt.plot(pd.Series(predictions, index=y_test.index), color='green')
    plt.xlabel('Month', fontsize=16)
    plt.ylabel('Gas Demand', fontsize=16)
    plt.legend(labels=['Real', 'Prediction'], fontsize=16)
    plt.grid()
    plt.show()


import pandas as pd
import matplotlib.pyplot as plt

def average_forecast(df, start_date, end_date, target):
    # Create a new DataFrame for the forecasted values
    forecast_df = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq='M'))
    # Fill the forecast DataFrame with the average values
    forecast_df['average'] = df[target].mean()
    print("Forecasting: " + str(df[target].mean()) )
    # Plot the original time series and the forecasted values
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[target], label='Original')
    plt.plot(forecast_df.index, forecast_df['average'], label='Forecast')
    plt.title('Original Time Series and Forecast')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    return forecast_df


def create_lagged_matrix(df, column, lag):
    """
    Creates a matrix with lagged values from a given column of a DataFrame
    and divides it into X and y arrays suitable for machine learning training.

    Args:
        df (pandas.DataFrame): DataFrame containing the time series data.
        column (str): Name of the column containing the time series.
        lag (int): Number of lagged values to include.

    Returns:
        numpy.array, numpy.array: X array with lagged values, y array with target values.
    """
    lagged_df = pd.DataFrame()
    for i in range(lag):
        lagged_df[f'Lag{i+1}'] = df[column].shift(i+1)
    lagged_df.dropna(inplace=True)
    X = lagged_df.iloc[:, 1:].values
    y = lagged_df.iloc[:, 0].values
    return X, y, lagged_df


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def train_time_series_with_linreg(X,y ,horizon=24*7):
    # Convert to pd
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)

    # Create train and test
    X_train, X_test = X.iloc[:-horizon,:], X.iloc[-horizon:,:]
    y_train, y_test = y.iloc[:-horizon], y.iloc[-horizon:]

    # Reset index for X_test and y_test
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)


    # Create, train, and do inference of the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    # If predictons are negative map them to min value
    for i in range(len(predictions)):
        if predictions[i] <= 0:
            predictions[i] = 440
    # Calculate MAE
    mae = np.round(mean_absolute_error(y_test, predictions), 3)
    # Plot reality vs prediction for the last week of the dataset
    fig = plt.figure(figsize=(16,8))
    plt.title(f'Real vs Prediction - MAE {mae}', fontsize=20)
    plt.plot(y_test, color='red', marker='o')
    plt.plot(predictions, color='green', marker='o', linestyle='dashed')
    plt.xlabel('Month', fontsize=16)
    plt.ylabel('Gas Demand', fontsize=16)
    plt.legend(labels=['Real', 'Prediction'], fontsize=16)
    plt.grid()
    plt.show()

    # Create a dataframe with the variable importances of the model (not applicable for linear regression)
    df_importances = None

    return model, mae, df_importances


import numpy as np

import numpy as np

def iterative_forecasting(X, model, num_forecasts, column_names):
    """
    Perform iterative forecasting using a machine learning model.

    Args:
        X (numpy.ndarray): Lagged features used for training the model.
        model: Trained machine learning model.
        num_forecasts (int): Number of forecasts to generate.
        column_names (list): Column names of the lagged features.

    Returns:
        pandas.DataFrame: Forecasted values for the specified periods.
    """
    # Create a copy of the original lagged features
    X_copy = pd.DataFrame(X.copy(), columns=column_names)

    # Create an empty DataFrame to store the forecasted values
    forecast_df = pd.DataFrame(index=range(1, num_forecasts + 1), columns=['Forecast'])

    for i in range(num_forecasts):
        print("Forecast nr:", i)
        # Get the last row in the lagged features
        last_row = X_copy.iloc[-1, :].values

        # Make a prediction using the lagged features
        prediction = model.predict([last_row])[0]
        print("Predicted:", prediction[0])

        # Shift the values in the row to the right
        new_row = np.roll(last_row, 1)

        # Update the first value in the row with the prediction
        new_row[0] = prediction[0]

        # Convert the new row back to a DataFrame and set the index
        new_row = pd.DataFrame([new_row], columns=column_names)
        print(new_row)

        # Store the prediction in the forecast DataFrame
        forecast_df.loc[i + 1, 'forecast'] = prediction[0]

        # Append the new row to the lagged features
        X_copy = X_copy.append(new_row, ignore_index=True)
        print(X_copy)

    return forecast_df


def plot_forecast(df, forecast_df, target):
    """
    Plot forecasted values on the original time series plot.

    Args:
        df (pandas.DataFrame): Historical data.
        forecast_df (pandas.DataFrame): Forecasted values.
        target (str): Target variable column name.

    Returns:
        None
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[target], label='Original')
    plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast')
    plt.title('Original Time Series and Forecast')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()


def create_lagged_matrix_2(df, target_column, feature_column, lag):
    """
    Creates a lagged matrix from a pandas DataFrame with a target variable and a feature variable.

    Args:
        df (pandas.DataFrame): DataFrame containing the time series data.
        target_column (str): Name of the column containing the target variable.
        feature_column (str): Name of the column containing the feature variable.
        lag (int): Number of lagged values to include.

    Returns:
        numpy.array, numpy.array: X array with lagged feature values, y array with lagged target values.
    """
    lagged_df = pd.DataFrame()
    for i in range(lag):
        lagged_df[f'Lag{i+1}_{feature_column}'] = df[feature_column].shift(i+1)
    lagged_df[f'Lag1_{target_column}'] = df[target_column].shift(1)
    lagged_df.dropna(inplace=True)
    X = lagged_df.iloc[:, :lag].values
    y = lagged_df.iloc[:, lag].values
    return X, y


def plot(df):
    plt.plot(df)
    plt.show()




def create_lagged_matrix_3(df, column, lag):
    """
    Creates a matrix with lagged values from a given column of a DataFrame
    and divides it into X and y arrays suitable for machine learning training.

    Args:
        df (pandas.DataFrame): DataFrame containing the time series data.
        column (str): Name of the column containing the time series.
        lag (int): Number of lagged values to include.

    Returns:
        numpy.array, numpy.array: X array with lagged values, y array with target values.
    """
    lagged_df = pd.DataFrame()
    for i in range(lag):
        lagged_df[f'Lag{i+1}'] = df[column].shift(i+1)
    lagged_df.dropna(inplace=True)
    #X = lagged_df.iloc[:, 1:].values
    #y = lagged_df.iloc[:, 0].values
    return lagged_df





