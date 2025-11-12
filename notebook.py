import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Twitter Hate Speech Detection
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    import seaborn as sns
    import unicodedata
    import re
    import string
    from nltk.corpus import stopwords
    import html
    from collections import Counter
    import random

    import time

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.metrics import f1_score, classification_report, confusion_matrix
    from scipy.stats import ttest_rel

    import psutil, os, joblib
    return (
        Counter,
        GridSearchCV,
        LinearSVC,
        LogisticRegression,
        StratifiedKFold,
        TfidfVectorizer,
        WordCloud,
        classification_report,
        clone,
        confusion_matrix,
        f1_score,
        html,
        joblib,
        mo,
        np,
        os,
        pd,
        plt,
        psutil,
        random,
        re,
        sns,
        stopwords,
        string,
        time,
        train_test_split,
        ttest_rel,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Dataset Exploration
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Datset shape & contents
    """)
    return


@app.cell
def _(pd):
    train = pd.read_csv('data/train_tweets.csv')
    return (train,)


@app.cell
def _(train):
    print("train set shape:", train.shape)
    print("train set columns:", train.columns)
    return


@app.cell
def _(train):
    print('train set (no null count)')
    print(train.isnull().sum())
    return


@app.cell
def _(train):
    train.head(10)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Example of hate and non-hate speech tweets
    """)
    return


@app.cell
def _(train):
    train[train["label"] == 0].iloc[0]
    return


@app.cell
def _(train):
    train[train["label"] == 1].iloc[0]
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Class distribution
    """)
    return


@app.cell
def _(plt, train):
    class_counts = train["label"].value_counts() # samples per class
    print(class_counts)

    plt.figure(figsize=(3,3))

    colors = ["skyblue", "indianred"]  # one per bar
    class_counts.plot(kind="bar", color=colors, edgecolor="black")

    plt.title("Class distribution")
    plt.xlabel("Label")
    plt.ylabel("Number of samples")
    plt.xticks(rotation=0)
    plt.show()
    return


@app.cell
def _(WordCloud, plt):
    def create_wordcloud_from(df, column_name, title="title"):
        text_data = " ".join(df[column_name].astype(str))

        wc = WordCloud(width=800, 
                       height=500, 
                       background_color="white").generate(text_data)

        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(title)
        plt.show()
    return (create_wordcloud_from,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Wordclouds per class before data cleaning
    """)
    return


@app.cell
def _(create_wordcloud_from, train):
    # class 0 wordcloud
    create_wordcloud_from(train[train["label"] == 0], "tweet", title="Wordcloud for class 0 - Non-hate speech (before preprocessing)")
    return


@app.cell
def _(create_wordcloud_from, train):
    # class 1 wordcloud
    create_wordcloud_from(train[train["label"] == 1], "tweet", title="Wordcloud for class 1 - Hate speech (before preprocessing)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We are working with a hate speech dataset. In total, there are `31962` examples in the dataset, of which `29720` are labelled as non-hate speech (0) and the rest `2242` are labelled as hate speech (1).

    Based on the class distribution, it is clear that the dataset is imbalanced.

    We also notice noise in our data from the word clouds and the samples from the dataframes. Noise includes mentions (@user), hashtags (#lyft) and some weird characters like: `ð`.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Cleaning & Preparing data
    """)
    return


@app.cell
def _(html, re, stopwords, string):
    def text_cleaning(text):
        # ensuring text is string type
        text = str(text)
        # lowercase 
        text = text.lower() 
        # remove URLs, mentions and hashtags
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)  
        text = re.sub(r"@\w+|#", "", text) 
        # converts &amp; -> &, &lt; -> <, etc. 
        text = html.unescape(text)
        # remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation)) 
        # remove numbers
        text = re.sub(r"\d+", "", text) 
        # remove extra whitespace
        text = text.strip() 
        text = re.sub(r"\s+", " ", text)
        # remove non-ASCII chars
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        # remove stopwords 
        stop_words = set(stopwords.words("english"))
        text = " ".join([word for word in text.split() if word not in stop_words])

        return text
    return (text_cleaning,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Side by side comparison of a tweet before and after cleaning
    """)
    return


@app.cell
def _(text_cleaning, train):
    train["clean_tweet"] = train["tweet"].astype(str).apply(text_cleaning)
    train.head(3)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Wordclouds per class after data cleaning
    """)
    return


@app.cell
def _(create_wordcloud_from, train):
    # class 0 wordcloud - after 
    create_wordcloud_from(train[train["label"] == 0], "clean_tweet", title="Wordcloud for class 0 - Non-hate speech (after preprocessing)")
    return


@app.cell
def _(create_wordcloud_from, train):
    # class 1 wordcloud
    create_wordcloud_from(train[train["label"] == 1], "clean_tweet", title="Wordcloud for class 1 - Hate speech (after preprocessing)")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## How rich is the dataset's vocabulary (per class/in general)?

    1. Vocabulary richness: ratio of unique tokens to total tokens
    2. Coverage: how much of the dataset is covered by the top 50 words

    A small vocabulary richness means repetition, while a larger one means more variety or noise. If those few words cover most of the tokens, the text is not very diverse, thus a smaller coverage.
    """)
    return


@app.cell
def _(Counter, random):
    def class_text_analysis(df, text_col="clean_tweet", label_col="label", top_n=20):
        results = {}

        for label in df[label_col].unique():
            subset = df[df[label_col] == label]
            all_tokens = " ".join(subset[text_col]).split()
            total_tokens = len(all_tokens)
            vocab = set(all_tokens)

            # compute metrics
            vocab_richness = len(vocab) / total_tokens if total_tokens > 0 else 0
            counts = Counter(all_tokens)
            most_common = counts.most_common(top_n)
            coverage = sum(freq for _, freq in most_common) / total_tokens if total_tokens > 0 else 0

            results[label] = {
                "total_tokens": total_tokens,
                "unique_tokens": len(vocab),
                "vocab_richness": vocab_richness,
                "coverage_top_20": coverage,
                "most_common": most_common,
                "sample_texts": random.sample(list(subset[text_col]), k=min(3, len(subset)))
            }

        for label, info in results.items():
            print(f"\n--- Class {label} ---")
            print(f"total tokens: {info['total_tokens']}")
            print(f"unique tokens: {info['unique_tokens']}")
            print(f"vocab richness: {info['vocab_richness']*100:.2f}%")
            print(f"coverage (top {top_n} words): {info['coverage_top_20']*100:.2f}%\n")

            print("\nrandom samples:")
            for t in info["sample_texts"]:
                print("-", t)
            print("\n" + "-" * 50)

        return results
    return (class_text_analysis,)


@app.cell
def _(class_text_analysis, train):
    results = class_text_analysis(train)
    return


@app.cell
def _(train):
    all_tokens = " ".join(train["clean_tweet"]).split()
    vocab_size = len(set(all_tokens))
    vocab_richness = vocab_size / len(all_tokens)
    print(f"vocabulary richness: {vocab_richness*100:.2f}%")
    return (all_tokens,)


@app.cell
def _(Counter, all_tokens):
    counts = Counter(all_tokens)
    most_common = counts.most_common(50)
    coverage = sum(freq for _, freq in most_common) / len(all_tokens)
    print(f"vocabulary coverage: {coverage*100:.2f}%")
    return


@app.cell
def _(mo):
    mo.md(r"""
    The non-hate speech class shows a broad and diverse vocabulary (richness ≈ 16 %), while the hate-speech class, though smaller, showed a denser and more distinctive language (richness ≈ 32 %), indicating that hateful content relies on a narrower set of highly specific terms and phrases. Thus, it is decided to cap the vocabulary at around 10k features and include bigrams for better context capture.
    """)
    return


@app.cell
def _(TfidfVectorizer):
    vectorizer = TfidfVectorizer(
        max_features=10000,  # keep top 10k most informative words
        min_df=3,            # drop words appearing in <3 docs
        max_df=0.95,         # drop words in >95% of docs
        ngram_range=(1,2)
    )
    return (vectorizer,)


@app.cell
def _(train, train_test_split, vectorizer):
    # text and label
    X_text = train["clean_tweet"]
    y = train["label"]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, stratify=y, random_state=42
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    return X_test, X_test_text, X_train, y_test, y_train


@app.cell
def _(mo):
    mo.md(r"""
    # Model Training
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Grid Search
    """)
    return


@app.cell
def _(GridSearchCV, LinearSVC, LogisticRegression, X_train, time, y_train):
    def tune_best_clf(clf, params, X, y, cv=3, scoring="f1_macro"):
        print(f"Starting tuning for {clf.__class__.__name__}...")
        t0 = time.perf_counter()

        grid_search = GridSearchCV(
            estimator=clf,
            param_grid=params,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )
        grid_search.fit(X, y)
        t1 = time.perf_counter()
        print(f"Tuning finished in {(t1-t0):.2f}s.")
        print(f"Best score: {grid_search.best_score_:.4f}")
        print(f"Best params: {grid_search.best_params_}")

        return grid_search.best_estimator_

    # base models and tuning parameters
    base_logreg = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
    base_linearsvc = LinearSVC(class_weight="balanced", dual="auto") # dual="auto" handles sparse data efficiently

    # parameter grid 
    param_grid_logreg = {'C': [0.1, 1.0, 10.0]}
    param_grid_linearsvc = {'C': [0.1, 1.0, 10.0]}

    best_logreg = tune_best_clf(base_logreg, param_grid_logreg, X_train, y_train)
    best_linearsvc = tune_best_clf(base_linearsvc, param_grid_linearsvc, X_train, y_train)

    models = {
        "logreg": best_logreg,
        "linearsvc": best_linearsvc,
    }
    return (models,)


@app.cell
def _(mo):
    options = {
        "Run sample Test (N=5) - Quick check": 5,
        "Run Full Test (N=100, ~10 mins) - Final report data": 100,
    }

    test_selector = mo.ui.radio(
        options=list(options.keys()),
        value=list(options.keys())[0], 
        label="### Select test length"
    )
    return options, test_selector


@app.cell
def _(options, test_selector):
    N_REPEATS_final = options[test_selector.value]
    return (N_REPEATS_final,)


@app.cell
def _(mo):
    mo.md(r"""
    ## HERE YOU CAN CHANGE EXPERIMENT PARAMETERS

    The 5 fold cross validation will run 5 times by default, the results discussed in the report were from running the experiments 100 times. To do so change the test length to 100 by clicking the button below.
    """)
    return


@app.cell
def _(N_REPEATS_final, mo, test_selector):
    mo.md(f"""
    {test_selector}

    **Current Repetitions (N):** `{N_REPEATS_final}`. The evaluation will run {5 * N_REPEATS_final} training/testing iterations per model.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 Fold Cross-Validation
    """)
    return


@app.cell
def _(
    N_REPEATS_final,
    StratifiedKFold,
    X_train,
    clone,
    f1_score,
    joblib,
    models,
    np,
    os,
    psutil,
    time,
    y_train,
):
    save_path = f"clf_results_{N_REPEATS_final}.pkl"
    eff_path = f"clf_efficiency_{N_REPEATS_final}.pkl"

    if os.path.exists(save_path) and os.path.exists(eff_path):
        print(f"Found existing results for {N_REPEATS_final} repetitions. Loading from disk...")
        clf_results_f1 = joblib.load(save_path)
        clf_results_efficiency = joblib.load(eff_path)
    else:
        print(f"No cached results found for {N_REPEATS_final} repetitions. Running experiment...")

        # f1 scores - for statistical test
        clf_results_f1 = {name: {"f1_macro": []} for name in models.keys()}
        # all time/space metrics across all repeats
        clf_results_efficiency = {
            name: {
                "train_time": [],
                "test_time": [],
                "mem_usage_mb": [],
                "model_size_mb": []
            }
            for name in models.keys()
        }

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        X_cv = X_train
        y_cv = y_train.reset_index(drop=True)

        for repeat_idx in range(N_REPEATS_final):

            for fold_idx, (idx_tr, idx_val) in enumerate(skf.split(X_cv, y_cv), start=1):
                X_tr, X_val = X_cv[idx_tr], X_cv[idx_val]
                y_tr, y_val = y_cv.iloc[idx_tr], y_cv.iloc[idx_val]

                for name, clf in models.items():
                    model = clone(clf)

                    # measure memory before training
                    process = psutil.Process(os.getpid())
                    mem_before = process.memory_info().rss / (1024 ** 2)

                    # measure training time
                    t0 = time.perf_counter()
                    model.fit(X_tr, y_tr)
                    t1 = time.perf_counter()

                    mem_after = process.memory_info().rss / (1024 ** 2)
                    mem_used = max(mem_after - mem_before, 0)

                    # temporarily save model to check file size
                    temp_file = f"temp_{name}_{repeat_idx}_{fold_idx}.pkl" # unique name
                    joblib.dump(model, temp_file)
                    model_size = os.path.getsize(temp_file) / (1024 ** 2)
                    os.remove(temp_file)

                    # measure test time
                    t2 = time.perf_counter()
                    y_pred = model.predict(X_val)
                    t3 = time.perf_counter()

                    # record all efficiency metrics
                    clf_results_efficiency[name]["train_time"].append(t1 - t0)
                    clf_results_efficiency[name]["test_time"].append(t3 - t2)
                    clf_results_efficiency[name]["mem_usage_mb"].append(mem_used)
                    clf_results_efficiency[name]["model_size_mb"].append(model_size)

                    if repeat_idx == 0:
                        f1 = f1_score(y_val, y_pred, average="macro")
                        clf_results_f1[name]["f1_macro"].append(f1)

        # save results to disk
        joblib.dump(clf_results_f1, save_path)
        joblib.dump(clf_results_efficiency, eff_path)
        print(f"Results saved to {save_path} and {eff_path}")

    # final output and averaging
    print(f"Averaged results over {N_REPEATS_final} repetitions ---")
    for name in models.keys():
        f1_stats = clf_results_f1[name]["f1_macro"]
        eff_stats = clf_results_efficiency[name]

        print(f"\n=== {name} ===")
        # effectiveness (F1 based on 5 folds from the first run)
        print(f"f1_macro:  mean={np.mean(f1_stats):.4f} (5 folds)")
        # efficiency (time/space based on 5 folds * N_REPEATS_final)
        print(f"train_time (s): mean={np.mean(eff_stats['train_time']):.4f} ({5*N_REPEATS_final} runs)")
        print(f"test_time  (s): mean={np.mean(eff_stats['test_time']):.4f} ({5*N_REPEATS_final} runs)")
        print(f"memory usage (MB): mean={np.mean(eff_stats['mem_usage_mb']):.2f} ({5*N_REPEATS_final} runs)")
        print(f"model size (MB): mean={np.mean(eff_stats['model_size_mb']):.2f} ({5*N_REPEATS_final} measurements)")

    clf_results = clf_results_f1 
    return clf_results, clf_results_efficiency, clf_results_f1


@app.cell
def _(clf_results, ttest_rel):
    logreg_f1 = clf_results["logreg"]["f1_macro"]
    svm_f1 = clf_results["linearsvc"]["f1_macro"]

    print("logreg f1:", logreg_f1)
    print("svm f1:   ", svm_f1)

    t_stat, p_val = ttest_rel(logreg_f1, svm_f1)
    print(f"\npaired t-test on macro f1 (5 folds)")
    print(f"t = {t_stat:.4f}, p = {p_val:.4f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Best model - Results & Error Analysis
    """)
    return


@app.cell
def _(X_test, X_train, clone, models, y_train):
    best_model = clone(models['logreg']) #LinearSVC(class_weight="balanced")
    best_model.fit(X_train, y_train)
    y_pred_test = best_model.predict(X_test)
    return (y_pred_test,)


@app.cell
def _(classification_report, confusion_matrix, y_pred_test, y_test):
    print(classification_report(y_test, y_pred_test, digits=4))

    cm = confusion_matrix(y_test, y_pred_test)
    print("confusion matrix:\n", cm)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Error analysis
    """)
    return


@app.cell
def _(X_test_text, np, y_pred_test, y_test):
    # reset index for easier alignment with X_test_text
    X_test_text_reset = X_test_text.reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True)

    fn_idx = np.where((y_test_reset == 1) & (y_pred_test == 0))[0]  # hate flagged as non-hate
    fp_idx = np.where((y_test_reset == 0) & (y_pred_test == 1))[0]  # non-hate flagged as hate

    print("=== sample false negatives (real hate predicted as non-hate) ===")
    for i in fn_idx[:10]:
        print("-", X_test_text_reset.iloc[i])

    print("\n=== sample false positives (non-hate predicted as hate) ===")
    for i in fp_idx[:10]:
        print("-", X_test_text_reset.iloc[i])
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Summary - Results & Plots
    """)
    return


@app.cell
def _(N_REPEATS_final):
    # total runs for titles and filenames
    NUM_RUNS = 5 * N_REPEATS_final
    return (NUM_RUNS,)


@app.cell
def _(NUM_RUNS, clf_results_efficiency, clf_results_f1, np, pd):
    # console summary
    summary_data = []
    for clf_name_s in clf_results_f1.keys():
        f1_stats_summary = clf_results_f1[clf_name_s]["f1_macro"]
        eff_stats_summary = clf_results_efficiency[clf_name_s]

        summary_data.append({
            "model": clf_name_s,
            "f1_macro": np.mean(f1_stats_summary),
            "train_time_s": np.mean(eff_stats_summary['train_time']),
            "test_time_s": np.mean(eff_stats_summary['test_time']),
            "model_size_mb": np.mean(eff_stats_summary['model_size_mb']),
            "mem_usage_mb": np.mean(eff_stats_summary['mem_usage_mb'])
        })

    df_summary = pd.DataFrame(summary_data).set_index("model")
    print(f"Summary DataFrame (averages of {NUM_RUNS} runs):\n", df_summary.round(4))

    dist_data = []
    for clf_name_s, stats in clf_results_efficiency.items():
        # training Time
        for val in stats['train_time']:
            dist_data.append({"model": clf_name_s, "time_type": "Train Time", "time_value": val})
        # test Time
        for val in stats['test_time']:
            dist_data.append({"model": clf_name_s, "time_type": "Inference Time", "time_value": val})
    df_dist = pd.DataFrame(dist_data)

    f1_plot_data = []
    for model_name, stats in clf_results_f1.items():
        for f1_macro_score in stats['f1_macro']:
            f1_plot_data.append({"model": model_name, "f1_macro": f1_macro_score})
    df_f1 = pd.DataFrame(f1_plot_data)
    return df_dist, df_f1


@app.cell
def _(mo):
    mo.md(r"""
    ## Time and memory usage
    """)
    return


@app.cell
def _(NUM_RUNS, N_REPEATS_final, df_dist, plt, sns):
    # box plots for TIME distribution ---
    plt.figure(figsize=(12, 5))

    # training time distribution
    plt.subplot(1, 2, 1)
    sns.boxplot(
        x="model", 
        y="time_value", 
        data=df_dist[df_dist['time_type'] == 'Train Time'], 
        hue="model", 
        palette="Set2",
        legend=False
    )
    plt.title(f"Training Time Distribution ({NUM_RUNS} Runs)")
    plt.ylabel("Time (seconds)")
    plt.xlabel("Classifier Model")
    plt.grid(axis='y', linestyle='--')

    # inference time distribution
    plt.subplot(1, 2, 2)
    sns.boxplot(
        x="model", 
        y="time_value", 
        data=df_dist[df_dist['time_type'] == 'Inference Time'], 
        hue="model", 
        palette="Set2",
        legend=False
    )
    plt.title(f"Inference Time Distribution ({NUM_RUNS} Runs)")
    plt.ylabel("Time (seconds)")
    plt.xlabel("Classifier Model")
    plt.grid(axis='y', linestyle='--')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) 

    plt.tight_layout()
    plt.savefig(f"time_distribution_comparison_N{N_REPEATS_final}.png")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Performance
    """)
    return


@app.cell
def _(N_REPEATS_final, df_f1, plt, sns):
    # box plot for f1 distribution 
    plt.figure(figsize=(6, 4))
    sns.boxplot(
        x="model",
        y="f1_macro",
        data=df_f1,
        hue="model",
        palette="Set1",
        legend=False
    )
    plt.title(f"Effectiveness Distribution (Macro F1-score Across 5 Folds)")
    plt.ylabel("Macro F1-score")
    plt.xlabel("Classifier Model")
    plt.ylim(0.75, 0.85) 
    plt.grid(axis='y', linestyle='--')
    plt.savefig(f"f1_distribution_N{N_REPEATS_final}.png")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
