# Autism Spectrum Disorder Prediction Project

The objective of this project is to build a machine learning model that predicts whether a child exhibits Autism Spectrum Disorder (ASD) traits based on demographic and behavioral screening data using a Random Forest classifier implemented from scratch.

## Problem

Autism Spectrum Disorder (ASD) affects how individuals communicate, interact socially, and process the world around them. Early identification is critical. Research consistently shows that children who receive early intervention have significantly better developmental outcomes than those diagnosed later. However, formal clinical diagnosis is often slow, expensive, and inaccessible, especially in regions with limited access to specialists, leading to long waitlists and delayed support for families who need it most.

The AQ-10 (Autism Spectrum Quotient – 10 items) is a short, validated screening questionnaire designed to flag individuals who may benefit from a full diagnostic evaluation. It is not a diagnostic tool itself, it's a low-cost, accessible first step that helps prioritize who should seek professional assessment. The children's version used in this project is tailored for ages 4–11, capturing behavioral and attention patterns relevant to that age group.

This project investigates whether a screening outcome can be predicted using a combination of behavioral indicators and demographic / background information, with a particular focus on whether models can remain useful even when the most direct, leakage-prone source of signal (the AQ-10 total score) is excluded. The broader goal is to explore the potential and the limitations of using lightweight, data-driven screening tools to complement, not replace, clinical evaluation.

## Introduction to Dataset

We begin by loading the dataset and understanding its characteristics:

```
# All Libraries
import pandas as pd
import numpy as np
import ucimlrepo
import matplotlib.pyplot as plt
import seaborn as sns
import random

#Load the dataset from UCI ML Repo
from ucimlrepo import fetch_ucirepo

dataset = fetch_ucirepo(id=419)

X = dataset.data.features
Y = dataset.data.targets

# Concatenate our full dataset
data = pd.concat([X, Y], axis=1)

# Display value counts for each column
for col in data.columns:
    counts = data[col].value_counts()
    print(f'data[{col}]')
    print(counts)
    print('\n')
```

From the result above, we can summarize the dataset description:

- `Age`: age in years
- `Gender`: Male or Female
- `Ethnicity`: The applicant's ethnicity
- `Jaundice`: Whether the case was born with jaundice (Y/N)
- `Autism`: Whether any immidiate family member has a PDD (Y/N)
- `Relation`: Who is completing the test (Parent, self, caregiver, medical staff, clinician ,etc.)
- `Country_Of_Res`: Country of residency
- `Used_App_Before`: Whether the user has used a screening app (Y/N)
- `Age_Desc`: Age category, which is 4-11 years old
- `A1_Score` to `A10_Score`: The answer code of the question based on the screening method used (1/0)
- `Result`: Total score of answer

## Standardizing Data

### Standardizing Column Names

Column names were standardized by converting them to title case to improve readability and maintain consistency throughout the analysis.

```
data.columns = data.columns.str.title()
```

### Standardizing Text Data

Text-based categorical variables were cleaned by removing leading and trailing spaces, removing quotation marks, and converting all values to lowercase. This ensured consistent category labels and prevented duplicate categories caused by formatting differences.

```
cat_cols = data.select_dtypes(include='object').columns
for col in cat_cols:
    data[col] = (data[col]
               .str.strip()
               .str.strip("'\"")   # remove leading/trailing quote characters
               .str.strip()        # remove whitespace that was hiding inside the quotes
               .str.lower())
```

## Handling Missing Values

Percentage of missing values in each column:

`data.isnull().sum() * 100 / len(data)`

Missing values were identified in Age, Ethnicity, and Relation. Records with missing Age values were removed due to their small proportion (less than 5%). Missing values in Ethnicity and Relation were assigned an "unknown" category to preserve observations while avoiding demographic assumptions.

```
data.dropna(subset=['Age'], inplace=True)

data['Ethnicity'] = data['Ethnicity'].fillna('Unknown')
data['Relation'] = data['Relation'].fillna('Unknown')
```

## Exploratory Data Analysis

We examined the dataset through visual analysis using seaborn to plot distributions, correlations, and demographic breakdowns that inform our preprocessing and modeling decisions.

### Class Histogram

```
sns.countplot(x='Class', data=data)

plt.title('Class Distribution')
plt.show()
```

The class distribution is relatively balanced across the two screening outcomes. Out of 292 children, 148 were screened negative for ASD (No), while 139 screened positive (Yes) (a near-even 52:48 split). This is a favorable characteristic for a classification task, as it means the model is unlikely to be biased toward predicting the majority class, and standard accuracy remains a reasonably meaningful metric alongside recall and F1-score.

### Age Histogram

```
sns.histplot(data['Age'])

plt.title('Age Distribution')
plt.show()
```

The age distribution across the dataset spans from 4 to 11 years, consistent with the children's screening scope. The distribution is notably right-skewed, with younger children making up the bulk of the data, where age 4 accounts for the largest group at 92 instances, followed by ages 5 and 6 with 45 and 39 instances respectively. Representation drops off from age 7 onward, with each remaining age group sitting between 18 and 26 instances. This imbalance is worth noting: the model will have seen considerably more examples from younger age groups during training, which may affect how reliably it generalizes to screening outcomes for older children in the 9–11 range.

### Countplot of Gender

```
sns.countplot(x='Gender', data=data)

plt.title('Countplot of Gender')
plt.show()
```

The gender distribution shows a notable skew toward male children, with 204 instances compared to 83 female (a roughly 71:29 split). This imbalance is not incidental; it actually reflects a well-established epidemiological pattern in ASD research, where males are consistently diagnosed at significantly higher rates than females, often cited at a ratio of around 4:1. While this makes the gender imbalance in the dataset expected and clinically grounded, it is still worth flagging as a modeling consideration (the model will have seen far more male cases during training, which may affect its sensitivity when screening female children). This is especially relevant given ongoing research suggesting that ASD in females is frequently underdiagnosed due to differing behavioral presentations, meaning the dataset's label itself may carry this real-world bias forward.

### Countplot of Ethnicity

```
sns.countplot(x='Ethnicity', data=data)

plt.title('Countplot of Ethnicity')
plt.show()
```

The ethnicity distribution highlights a significant imbalance across groups in the dataset. White-European is the dominant group by a wide margin, with 108 instances, followed by Asian at 46 and Hispanic at 40. The remaining groups (Middle Eastern (26), South Asian (21), Latino (9), Turkish (7), Pasifika (3), Black (visible but small), and Other (15)) are considerably underrepresented. This degree of imbalance is a meaningful limitation: with some ethnicities represented by fewer than 10 instances, the model has very little basis to learn reliable patterns for those groups. This also raises a fairness concern worth acknowledging in the limitations section where a model trained on this distribution may not generalize equally well across all ethnic backgrounds, which is a particularly sensitive consideration in a healthcare screening context.

### Countplot of Jaundice

```
sns.countplot(x='Jaundice', data=data)

plt.title('Jaundice Distribution')
plt.show()
```

The jaundice distribution reveals that the majority of children in the dataset were not born with jaundice, accounting for 209 instances compared to 78 who were. This roughly 73:27 split indicates that jaundice history is a minority characteristic in this sample. It is worth noting that neonatal jaundice has been explored in existing literature as a potential early biological risk factor associated with ASD, making this a feature of genuine clinical interest despite its imbalance, and one to pay attention to when examining feature importances later in the modeling section.

### Countplot of Family Member with PDD

```
sns.countplot(x='Autism', data=data)

plt.title('Countplot of Family Member with PDD')
plt.show()
```

The distribution of family history of Pervasive Developmental Disorder (PDD) shows that the majority of children (240 instances) have no family member with PDD, while 50 do. This 83:17 split makes family history a relatively rare characteristic in the dataset. Despite its low frequency, it remains a clinically meaningful feature: ASD is known to have a strong hereditary component, with siblings and first-degree relatives of diagnosed individuals carrying a significantly elevated likelihood of also being on the spectrum. As such, even with limited representation, this feature may carry disproportionate predictive signal relative to its count, making it one to watch closely when interpreting feature importances in the modeling section.

### Countplot of Whether The User Has Used App

```
sns.countplot(x='Used_App_Before', data=data)

plt.title('Countplot of Whether The User Has Used App')
plt.show()
```

The vast majority of respondents (approximately 279 instances) had not previously used an ASD screening app, with only around 10 having done so. This extreme 97:3 imbalance makes used_app_before an effectively near-constant feature in this dataset. As flagged during the preprocessing stage, a feature with this little variation carries almost no discriminative power for the model regardless of any real-world relationship it might have with screening outcomes. Therefore, this column will be dropped before modeling.

## Label Encoding

### Coverting to Numerical Values

```
# Using get_dummies for 2 categorical values
gender = pd.get_dummies(data['Gender'], prefix = 'Gender', dtype=int, drop_first=True)
jaundice = pd.get_dummies(data['Jaundice'], prefix = 'Jaundice', dtype=int, drop_first=True)
autism = pd.get_dummies(data['Autism'], prefix = 'Autism', dtype=int, drop_first=True)
class_lbl = pd.get_dummies(data['Class'], prefix = 'Class', dtype=int, drop_first=True)

# Using mapping for more than two categorical values
for col in ['Ethnicity', 'Country_Of_Res', 'Relation']:
    categories = data[col].unique()
    mapping = {cat: i for i, cat in enumerate(categories)}
    data[col] = data[col].map(mapping)
    print(f"{col}: {mapping}")  # we keep this to know what number maps to what

# Concatenate our dummie variables
data = pd.concat([data, gender, jaundice, autism, class_lbl], axis=1)

# Drop the initial columns and Age_Desc column
data.drop(['Gender', 'Jaundice', 'Autism', 'Class', 'Age_Desc', 'Used_App_Before'], axis=1, inplace=True)
```

### Heatmap

```
# Get all feature columns without the target column
x = data.drop(columns=['Class_yes'])

# Make a hetmap to show correlation
plt.figure(figsize=(14, 12))
sns.heatmap(x.corr(),
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            linewidths=0.5,
            annot_kws={'size': 8})

plt.title('Correlation Heatmap of All Features', fontsize=13)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.show()
```

Based on the correlation heatmap above, `Result` have high correlation with `A1_Score` to `A10_Score`. This is expected since result is the total score of answer. Therefore, we dropped this variable to avoid multicollinearity. 

`data.drop(['Result'], axis=1, inplace=True)`

## Building a Machine Learning Model

### Why We Use a Random Forest Model?

We chose Random Forest because it naturally handles both numerical and categorical features, and its ensemble structure (averaging predictions across many decorrelated decision trees) substantially reduces the overfitting that a single decision tree is prone to. Implementing it from scratch also made it possible to directly control and inspect the bootstrap sampling and feature randomization steps that drive this variance reduction, rather than treating them as a black box.

### Data Splitting

We split the data with roughly 70:30 split.

```
training_data = data_list[0:231]
testing_data = data_list[231:289]
```

### Random Forest Implementation

Rather than using a pre-built Random Forest implementation, a custom Random Forest classifier was developed using Python, NumPy, and Pandas.

The implementation consists of:

1. Bootstrap sampling for generating training subsets.
2. Decision tree construction using recursive node splitting.
3. Gini impurity calculation for split evaluation.
4. Random feature selection at each split.
5. Majority voting across trees for final classification.

### Helper Functions
```
# Helper function to store set of values in each column
def unique_vals(rows, col): # it will iterate over all the rows using for loop, and in each iteration, it would be one row, and it will return its column as a list and this list will be converted into set
    return set([row[col] for row in rows]) # set stores unique values

# Helper function to get dict with keys 0 or 1 for survived/not and values is the total
def class_counts(rows): # it will return the numb of classes in these rows
    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

# Helper function to check if the value is numeric or not
def is_num(value):
    return isinstance(value, int) or isinstance(value, float)

# Helper function how to ask question in the root
class Question:
    def __init__(self, col, val):
        self.column = col
        self.value = val
    def match(self, example):
        val = example[self.column]
        if is_num(val):
            return val >= self.value
        else:
            return val == self.value

# Helper function to split the data between true and false
def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows
```

### Gini Impurity Function

```
def gini(rows):
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl ** 2
    return impurity
```

### Information Gain Function

```
def information_gain(left, right, current_impurity): # 'right' is the right row and 'left' is the left row
    prob = float(len(left)) / (len(left) + len(right))
    return current_impurity - prob * gini(left) - (1 - prob) * gini(right)
```

### Best Split Selection

```
def best_split(rows):
    best_gain = 0
    best_question = None
    current_impurity = gini(rows)
    n_features = len(rows[0])-1

    for col in range(n_features):
        values = set([row[col] for row in rows])
        for val in values:
            question = Question(col, val)
            true_rows, false_rows = partition(rows, question)

            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            gain = information_gain(true_rows, false_rows, current_impurity)

            if gain >= best_gain:
                best_gain, best_question = gain, question
    return best_gain, best_question
```

### Tree Node Creation

```
# Function for Leaf node
class Leaf:
    def __init__(self, rows):
        self.predictions = class_counts(rows)

# Function for Decision node
class DecisionNode:
    def __init__(self, question, true_branch, false_branch):
        self.question = question 
        self.true_branch = true_branch
        self.false_branch = false_branch
```

### Tree Building

```
def build_tree(rows):
    gain, question = best_split(rows)

    if gain == 0:
        return Leaf(rows)

    true_rows, false_rows = partition(rows, question)
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)

    return DecisionNode(question, true_branch, false_branch)
```

### Classifier Function

```
# Function to classify if it is a leaf or decision tree
def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions

    if node.question.match(row):
        return classify(row, node.true_branch)

    else:
        return classify(row, node.false_branch)
```

### Bootstrap Sampling

```
def bootstrap_sample(rows):
    sample = []
    for _ in range(len(rows)):
        # randomly pick a row WITH replacement
        random_row = rows[random.randint(0, len(rows) - 1)]
        sample.append(random_row)
    return sample
```

### Random Forest Construction

```
def build_forest(rows, n_trees=10):
    forest = []
    for i in range(n_trees):
        # get random sample of rows
        sample = bootstrap_sample(rows)
        
        # build a tree on that sample
        tree = build_tree(sample)
        
        # add tree to forest
        forest.append(tree)
        print(f"Tree {i+1} built ✅")
    
    return forest
```

### Prediction via Majority Voting

```
def forest_classify(row, forest):
    # collect prediction from each tree
    votes = {}
    for tree in forest:
        prediction = list(classify(row, tree).keys())[0]
        
        # count votes for each class
        if prediction not in votes:
            votes[prediction] = 0
        votes[prediction] += 1
    
    # return the class with most votes
    return max(votes, key=votes.get)
```

## Build Our Random Forest

We built our random forest model for our final dataset:

```
random.seed(100000)
forest = build_forest(training_data, n_trees=43)
```

## Model Evaluation

### Testing Loop Function

```
TP = 0
TN = 0
FP = 0
FN = 0

for row in testing_data:
    prediction = forest_classify(row[0:-1], forest)
    actual = row[-1]

    if prediction == '1' and actual == '1':
        TP += 1

    elif prediction == '0' and actual == '0':
        TN += 1

    elif prediction == '1' and actual == '0':
        FP += 1

    elif prediction == '0' and actual == '1':
        FN += 1
```

### Calculate Our Model Evaluation

```
accuracy = (TP + TN) / (TP + TN + FP + FN)

precision = TP / (TP + FP)

recall = TP / (TP + FN)

f1 = 2 * precision * recall / (precision + recall)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
```

**Interpretation:**

The model correctly classified approximately 84.5% of children in the test set, well above the ~51% baseline a majority-class guess would achieve given the dataset's near-even class split.

**An important caveat on these results:** while the `Result` column (the precomputed AQ-10 total) was excluded to avoid the most direct form of label leakage, the ten individual A1–A10 item scores were retained as features. Since `Class/ASD` is defined as a threshold on the *sum* of these exact ten items, the model still has indirect access to nearly all the information used to construct the label itself — it must learn the additive relationship across binary splits rather than reading it off a single precomputed column, but the underlying signal is the same. This likely explains why performance lands at 84.5% rather than the ~100% achievable with `Result` included directly: the tree-based splits recover most, but not all, of that additive pattern.

As a result, these metrics should be read as evidence that the model *can substantially reconstruct the AQ-10 screening rule from raw item-level responses* — a meaningful technical result in its own right — rather than as evidence that ASD screening status can be predicted from independent behavioral or demographic signal alone. A more rigorous test of the latter question would require excluding A1–A10 entirely and relying only on demographic and background features, which we explore as a follow-up experiment below.

Looking at the individual metrics: precision (0.8846) exceeds recall (0.7931), meaning the model is more reliable when it predicts a positive screen than when it's tasked with catching every true positive case — it misses roughly 1 in 5 children who should screen positive. In a real screening context, this kind of false negative is generally costlier than a false positive, since it represents a missed opportunity for early referral. This gap would be the priority area to address before considering any real-world application of a model like this.

### Confusion Matrix

```
print("\nConfusion Matrix")
print(f"TN = {TN}    FP = {FP}")
print(f"FN = {FN}    TP = {TP}")

cm = np.array([
    [TN, FP],
    [FN, TP]
])

print(cm)
```

**Interpretation:**

Out of 58 children in the test set:

- **26 true negatives**: correctly identified as not screening positive for ASD.
- **23 true positives**: correctly flagged as screening positive.
- **3 false positives**: children incorrectly flagged as a positive screen when they were not. In a screening context, this is the cheaper error: it leads to an unnecessary follow-up evaluation, but no missed opportunity for support.
- **6 false negatives**: children who should have screened positive but were missed by the model. This is the more consequential error type in a screening application, since it represents a child who may not get referred for early intervention they could benefit from.

The model produces roughly twice as many false negatives as false positives (6 vs. 3), which aligns with the precision-recall gap observed earlier (precision 0.8846 > recall 0.7931). This consistently points to the same conclusion: the model is conservative about predicting a positive screen, erring on the side of "no" when uncertain — the opposite of what's ideal for a screening tool, where catching borderline cases matters more than avoiding false alarms.

This reinforces that, were this model considered for any real-world screening use, the priority improvement would be reducing false negatives, even if that means accepting a higher false positive rate, rather than optimizing for overall accuracy.

## Limitations

While the model achieves promising results, several limitations are important to acknowledge, both as honest scientific practice and because they meaningfully shape how (or whether) a model like this could be used in practice.

**1. Indirect data leakage via A1–A10 item scores.**
Although the precomputed `Result` column was excluded to avoid the most obvious leakage, the ten individual AQ-10 item scores (A1–A10) were retained as features. Since `Class` is deterministically derived from the sum of these exact items, the model has near-complete indirect access to the rule that generated the label. As discussed in the Model Evaluation section, this means the reported accuracy (84.5%) largely reflects the model's ability to *reconstruct the AQ-10 threshold rule* from raw responses, rather than evidence that ASD screening status can be predicted from independent behavioral or demographic signal alone. A genuinely leakage-free evaluation would require excluding A1–A10 entirely.

**2. Small sample size.**
With only 292 instances, the dataset is small for training a robust classifier, particularly once split into train/test or cross-validation folds, where a single test set may contain as few as ~58 examples. Small misclassification counts can shift reported metrics by several percentage points, making point estimates less stable than they would be with a larger dataset.

**3. Demographic imbalance and fairness concerns.**
Several features show substantial skew: gender (71% male), ethnicity (White-European alone accounts for over a third of the dataset, while several groups have fewer than 10 instances), and age (heavily weighted toward 4-year-olds). The model has limited basis to learn reliable patterns for underrepresented subgroups, raising fairness concerns about how consistently it would perform across different populations, a particularly important consideration for a tool intended for healthcare screening contexts.

**4. The label itself is not an independent clinical diagnosis.**
`Class` reflects an AQ-10 threshold score, not a formal diagnostic assessment by a clinician. The dataset therefore measures whether a model can predict a *screening* outcome, not whether it can predict ASD itself, these are related but distinct targets, and the distinction matters when interpreting what the model's performance actually demonstrates.

**5. Near-constant and low-variance features.**
Features such as `Used_App_Before` (97% "no") carry very little discriminative information at this sample size, and country/ethnicity categories with very few instances each contribute minimal reliable  signal individually, even after encoding.

**6. No independent test set from a different population.**
All evaluation was performed on data drawn from the same source distribution as the training data. Performance on this dataset does not guarantee similar performance on children from different countries, cultural contexts, or age ranges outside 4–11.

**7. Limited hyperparameter exploration.**
As the Random Forest was implemented from scratch rather than using a library such as scikit-learn, extensive hyperparameter tuning (e.g. grid search over tree depth, minimum samples per split, or number of features considered per split) was not performed to the same depth that optimized library implementations typically allow. Performance may improve further with more systematic tuning.

These limitations don't undermine the project's value as a technical exercise, but they do mean its results should be read as a demonstration of method and process, not as a validated or deployment-ready screening tool.
