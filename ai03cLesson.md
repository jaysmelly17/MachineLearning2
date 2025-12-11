---
marp: true
theme: default
paginate: true
---

# Machine Learning with Decision Trees
## Lesson 3: Data Preparation

**AI/ML Course**
Medina County Career Center

---

# Recap: Where We Are

✅ **Lesson 1**: Loaded and inspected data
✅ **Lesson 2**: Cleaned data (selected features, handled missing values)

**Today**: Prepare data for machine learning
- Convert categorical variables to numbers
- Separate features (X) from target (y)
- Understand why these steps matter

---

# The Problem: Categorical Data

Our cleaned dataset has these columns:
```
pclass (numeric)    survived (numeric)
sex (TEXT!)         age (numeric)
sibsp (numeric)     parch (numeric)
fare (numeric)      embarked (TEXT!)
```

**Problem**: Machine learning models only understand **numbers**, not text!

We need to convert "male"/"female" and "C"/"Q"/"S" to numbers.

---

# Why Can't Models Use Text?

**Machine learning is math!**

Models calculate things like:
- Distance between data points
- Weights and probabilities
- Patterns and correlations

**You can't do math with words:**
- What's "male" + "female"?
- Is "male" > "female"? (Doesn't make sense!)

**Solution**: Convert text categories to numbers

---

# Two Types of Data

**Numerical (Quantitative)**
- Numbers with mathematical meaning
- Examples: age (25), fare ($30.50), pclass (2)
- Can calculate average, compare sizes

**Categorical (Qualitative)**
- Labels or categories
- Examples: sex ("male"/"female"), embarked ("C"/"Q"/"S")
- Can't do math with them

**Our job**: Convert categorical → numerical

---

# Method: One-Hot Encoding (Dummy Variables)

**One-hot encoding** = Create separate columns for each category

**Example with 'sex':**

Before:
```
sex
----
male
female
male
```

After:
```
sex_male
--------
   1
   0
   1
```

**1 = yes (is male), 0 = no (not male/is female)**

---

# Why It's Called "One-Hot"

**"One-hot"** = only ONE column is "hot" (1) at a time

For each row:
- If male: `sex_male = 1`
- If female: `sex_male = 0`

**Think of it like a light switch:**
- Switch ON (1) = this category applies
- Switch OFF (0) = this category doesn't apply

---

# Step: Convert 'sex' to Dummy Variables

```python
# Convert 'sex' to dummy variables
df = pd.get_dummies(df, columns=['sex'], drop_first=True)

print("Columns after converting sex:")
print(df.columns.tolist())
```

**Result**: `sex` column disappears, `sex_male` column appears!

**`drop_first=True`** = Keep only one column (we'll explain why next)

---

# Understanding `pd.get_dummies()`

```python
pd.get_dummies(df, columns=['sex'], drop_first=True)
```

**Breaking it down:**
- `pd.get_dummies()` → pandas function for one-hot encoding
- `df` → our DataFrame
- `columns=['sex']` → which columns to convert
- `drop_first=True` → drop one dummy variable

**Before**: 1 column (sex)
**After**: 1 column (sex_male)

---

# **Why `drop_first=True`?**

When we convert a categorical column (like `"sex"`) into dummy variables, pandas creates one column per category.

### **Without `drop_first`:**

```
sex_male   sex_female
   1          0        ← male
   0          1        ← female
```

This is **redundant** — if `sex_male = 0`, we automatically know `sex_female = 1`.

---

## **Multicollinearity (Simple Explanation)**

This redundancy causes **multicollinearity**, which means two features contain the **same information**.
Some models (like linear or logistic regression) struggle with this because they can’t tell which feature actually matters.

This specific situation is called the **dummy variable trap**.

---

## **How `drop_first=True` Fixes It**

By dropping one of the dummy columns, we remove the duplication:

### **With `drop_first=True`:**

```
sex_male
   1     ← male
   0     ← female
```

This:

* Removes redundant information
* Avoids the dummy variable trap (multicollinearity)
* Makes the data cleaner and more efficient
* Still keeps all the important information (the dropped category becomes the **baseline**)

---

# Step: Convert 'embarked' to Dummy Variables

'embarked' has 3 categories: C, Q, S

```python
# Convert 'embarked' to dummy variables
df = pd.get_dummies(df, columns=['embarked'], drop_first=True)

print("Columns after converting embarked:")
print(df.columns.tolist())
```

**Result**: 
- `embarked` column disappears
- `embarked_Q` and `embarked_S` columns appear
- (C is the baseline - when both Q and S are 0)

---

# Understanding the 3-Category Case

**Original**: embarked = C, Q, or S

**After encoding** (with `drop_first=True`):
```
embarked_Q  embarked_S  | Meaning
    0           0       | Embarked at C (baseline)
    1           0       | Embarked at Q
    0           1       | Embarked at S
```

**We only need 2 columns to represent 3 categories!**

The third category (C) is implied when both are 0.

---

# Our Data After Encoding

**Before encoding:**
```
pclass, survived, sex, age, sibsp, parch, fare, embarked
```

**After encoding:**
```
pclass, survived, age, sibsp, parch, fare, sex_male, embarked_Q, embarked_S
```

**Changes:**
- `sex` → `sex_male` (1 column)
- `embarked` → `embarked_Q` and `embarked_S` (2 columns)

---

# Quick Check: Reading Dummy Variables

```
sex_male  embarked_Q  embarked_S
   1          0           1
```

**What does this mean?**
- `sex_male = 1` → Male passenger
- `embarked_Q = 0` → Did NOT embark at Q
- `embarked_S = 1` → DID embark at S

**Answer**: Male passenger who boarded at Southampton (S)

---

# Step: Separate Features (X) from Target (y)

Now we need to split our data:
- **Features (X)**: What we use to make predictions
- **Target (y)**: What we're trying to predict

**Features**: pclass, age, sibsp, parch, fare, sex_male, embarked_Q, embarked_S
**Target**: survived

Think of it like a test:
- **X** = the questions/information given
- **y** = the answer we're trying to find

---

# Coding It: Create X and y

```python
# Creates a new DataFrame with every column except 'survived' (these are our input features)
X = df.drop('survived', axis=1)          

# Selects just the 'survived' column (this is what we want the model to predict)
y = df['survived']                       

# Check the shapes; X has rows AND columns, but y only shows rows 
# because y is a Series (1 dimensional) when you select a single column
print(f"X shape: {X.shape}")             
print(f"y shape: {y.shape}")


# See the feature names
print(f"\nFeature names: {X.columns.tolist()}")   # Prints a list of all the column names being used as features in X

```

---

# Understanding `.drop()`

```python
X = df.drop('survived', axis=1)
```

**Breaking it down:**
- `.drop('survived')` → remove the 'survived' column
- `axis=1` → drop a COLUMN (axis=0 would drop a ROW)
- Returns a new DataFrame without 'survived'

**Result**: X has all columns except the one we're predicting

---

# Understanding X and y Shapes

```python
print(f"X shape: {X.shape}")  # (1307, 8)
print(f"y shape: {y.shape}")  # (1307,)
```

**X shape (1307, 8):**
- 1307 passengers (rows)
- 8 features (columns)

**y shape (1307,):**
- 1307 passengers (rows)
- 1 value each (survived or not)

**Each row in X corresponds to the same row in y!**

---

# Visualizing X and y

```
X (Features)                              y (Target)
=================================         ===========
pclass  age  fare  sex_male  ...         survived
  3     22   7.25     1      ...            0
  1     38  71.28     0      ...            1
  3     26   7.93     1      ...            0
```

**For each passenger:**
- X contains their characteristics
- y contains whether they survived

**The model learns**: Given X features, predict y outcome

---

# Why Separate X and y? Because this is Supervised Learning

**Machine learning workflow:**
1. Model looks at X (features)
2. Model predicts y (target)
3. Compare prediction to actual y
4. Learn from mistakes and improve

**It's like studying for a test:**
- X = study materials (what you're given)
- y = answers on the test (what you need to predict)

---

# What Is Supervised Learning? 

Decision trees (and many ML models) use **supervised learning**, which means the model learns using examples that already have the correct answers.

* **X** = the information we give the model (features)
* **y** = the correct answer for each row (target)

The model tries to learn the relationship between X and y so it can predict new y values in the future.

**In other words:**
We “supervise” the model by showing it the right answers during training.

---

# Complete Preparation Code

```python
# Step: Convert sex to dummy variables
df = pd.get_dummies(df, columns=['sex'], drop_first=True)

# Step: Convert embarked to dummy variables
df = pd.get_dummies(df, columns=['embarked'], drop_first=True)

# Step: Separate features (X) and target (y)
X = df.drop('survived', axis=1)
y = df['survived']

# Verify
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Features: {X.columns.tolist()}")
```

---

# Before and After Summary

**Original data:**
- Text categories: "male", "female", "C", "Q", "S"
- All mixed together

**Prepared data:**
- All numerical values (0s and 1s)
- Features (X) separated from target (y)
- **Ready for machine learning!**

---

# Common Questions

**Q: Why not just use 1=male, 2=female?**
**A:** That implies an order (male < female), which doesn't make sense. Dummy variables treat categories equally.

**Q: Can I encode text manually?**
**A:** You could, but `pd.get_dummies()` is easier and safer!

**Q: What if I have 10 categories?**
**A:** You'll get 9 dummy columns (one is always dropped as baseline)

---

# What's Next?

**Lesson 4**: Training the Model
- Split data into training and testing sets
- Create a decision tree classifier
- Train the model on training data
- Understand what "training" means

**We're finally ready to build our model!**

---

# Practice Task

Complete **ai03cTasks.ipynb**:

1. Convert 'sex' to dummy variables
2. Convert 'embarked' to dummy variables
3. Verify the new column names
4. Separate X (features) and y (target)
5. Check shapes of X and y
6. Answer reflection questions

---

# Key Terms to Remember

- **Categorical data**: Text labels/categories
- **Numerical data**: Numbers with mathematical meaning
- **One-hot encoding**: Converting categories to binary (0/1) columns
- **Dummy variables**: Binary columns representing categories
- **`pd.get_dummies()`**: Pandas function for one-hot encoding
- **`drop_first=True`**: Drops one dummy to avoid redundancy
- **Features (X)**: Input data for predictions
- **Target (y)**: What we're trying to predict
- **Dummy variable trap**: Having redundant dummy variables

---

# Questions?

Next lesson: We'll split our data and train our first decision tree model!
