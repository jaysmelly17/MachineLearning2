---
marp: true
theme: default
paginate: true
---

# Machine Learning with Decision Trees
## Lesson 2: Data Cleaning

**AI/ML Course**
Medina County Career Center

---

# Recap: What We Did Last Time

✅ Installed and imported libraries
✅ Loaded the Titanic dataset
✅ Inspected the data with `.head()`, `.info()`, `.describe()`
✅ Found missing values

**Today**: We'll clean up the data so it's ready for machine learning!

---

# Why Data Cleaning Matters

**"Garbage in, garbage out"**

If you feed a model messy, incomplete, or irrelevant data, you'll get poor predictions.

**Data cleaning is 60-80% of a data scientist's job!**

Maybe not the best analogy... but think of data pre-processing/cleaning like prepping ingredients before cooking:
- Wash vegetables (clean data)
- Remove bad parts (handle missing values)
- Cut to right size (select features)

---

# The Titanic Dataset - Full Version

The original Titanic dataset has **15 columns**:

```
pclass, survived, name, sex, age, sibsp, parch, 
ticket, fare, cabin, embarked, boat, body, home.dest
```

**Do we need all of these for our model?**

Let's think about which ones actually help predict survival...

---

# Step: Select Useful Features

**Features we'll KEEP:**
- `pclass` - Passenger class (1st, 2nd, 3rd) → wealth indicator
- `survived` - Our target variable/feature (what we're predicting)
- `sex` - Male or female → "women and children first"
- `age` - Age in years → children prioritized
- `sibsp` - Siblings/spouses aboard → family size
- `parch` - Parents/children aboard → family size
- `fare` - Ticket price → another wealth indicator
- `embarked` - Port of boarding (C, Q, S) → might matter

---

# Features We'll DROP

**Why drop these?**

- `name` - Each person has unique name (no pattern to learn)
- `ticket` - Ticket number is random
- `cabin` - Too many missing values + complex
- `boat` - Lifeboat number (only if survived!)
- `body` - Body ID number (only if died!)
- `home.dest` - Too many unique values 

**Machine learning needs patterns, not unique identifiers!**

---

# Coding It: Select Columns

```python
# Select only these columns/features: the inside brackets create a list of column names, 
# and the outside brackets use that list to keep (choose) those columns from the DataFrame
df = df[['pclass', 'survived', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]

print("Columns after selection:")
print(df.columns.tolist())     # Prints all column names in the DataFrame


# There are a couple of ways to show the number of rows and columns in the updated dataframe ("df"):

# Option 1: Print rows and columns separately
print(f"Rows: {df.shape[0]}")     # Number of rows in the DataFrame
print(f"Columns: {df.shape[1]}")  # Number of columns in the DataFrame

# Option 2: Print them together as a tuple
print(f"\nShape: {df.shape}")     # Shows (rows, columns) together
```

**Result**: We now have 8 columns instead of 15!

---

# Step: Handle Missing Values

**Missing values** = cells with no data (shown as `NaN` or `None`)

**Why are values missing?**
- Data wasn't collected
- Person didn't provide it
- Lost during data entry

**We must handle these before machine learning!**

---

# Check for Missing Values

```python
# Count missing values in each column
print("Missing values per column:")
print(df.isnull().sum())
```

**Expected output:**
```
pclass        0
survived      0
sex           0
age         263    ← Missing!
sibsp         0
parch         0
fare          1    ← Missing!
embarked      2    ← Missing!
```

---

# Understanding `.isnull()`

```python
df.isnull()       # Returns True/False for each cell
df.isnull().sum() # Checks every column for missing values and prints how many blanks each column contains
```

**Breaking it down:**
1. `.isnull()` → checks every cell, creates True/False table
2. `.sum()` → adds up the `True` values (True = 1, False = 0)

**Result**: Count of missing values per column

---

# Strategies for Missing Values

**Strategy 1: Fill with a value**
- Use **median** (middle value) for numeric data with outliers
- Use **mean** (average) for normally distributed numeric data
- Use **mode** (most common) for categorical data

**Strategy 2: Drop the rows/columns**
- If very few missing (< 5%), can drop rows
- If most values missing (> 50%), drop entire column

**Choose based on how much data is missing!**

---

# Handling Missing Age Values

**Age** has 263 missing values (out of ~1300)

That's about 20% missing - too many to drop!

**Solution: Fill with median age**

```python
# Calculate median age the median function is built into Pandas
median_age = df['age'].median()
print(f"Median age: {median_age}")

# Fill missing ages with median
df['age'].fillna(median_age, inplace=True)

# Verify it worked
print(f"Missing ages after filling: {df['age'].isnull().sum()}")
```

---

# Why Median for Age?

**Median** = middle value when data is sorted

**Why not mean (average)?**
- Age has outliers (babies = 0, elderly = 80)
- Mean gets pulled by extreme values
- Median is more "typical"

**Example:**
- Ages: [5, 10, 15, 20, 85]
- Mean = 27 (not representative!)
- Median = 15 (better!)

---

# Understanding `.fillna()`

```python
df['age'].fillna(median_age, inplace=True). # Replace any missing ages with the median age, updating the column directly
```

**Breaking it down:**
- `df['age']` → selects the age column
- `.fillna(median_age)` → replaces NaN with median
- `inplace=True` → modifies the original DataFrame

**Without `inplace=True`**: Creates a new copy (original unchanged)
**With `inplace=True`**: Modifies df directly

---

# Handling Missing Fare Values

**Fare** has only 1 missing value

```python
# Fill missing fare with median fare
median_fare = df['fare'].median()
# Print median fare
print(f"Median fare: {median_fare}")

df['fare'].fillna(median_fare, inplace=True)

# Verify
print(f"Missing fares: {df['fare'].isnull().sum()}")
```

**Result**: 0 missing fares!

---

# Handling Missing Embarked Values

**Embarked** has only 2 missing values (very few!)

**Strategy: We will just drop those records/rows**

```python
# Drop rows where embarked is missing
df.dropna(subset=['embarked'], inplace=True)

# Verify
print(f"Missing embarked: {df['embarked'].isnull().sum()}")
print(f"Rows remaining: {len(df)}")
```

We lose 2 rows out of 1300+ → minimal impact!

---

# Understanding `.dropna()`

```python
df.dropna(subset=['embarked'], inplace=True)
```

**Breaking it down:**
- `.dropna()` → removes rows with missing values
- `subset=['embarked']` → only look at 'embarked' column
- `inplace=True` → modifies df directly

**Without subset**: Drops ANY row with ANY missing value
**With subset**: Only drops rows with missing values in specified columns

---

# Verify All Missing Values Are Gone

```python
# Final check for missing values
print("Missing values after cleaning:")
print(df.isnull().sum())
```

**Expected output:**
```
pclass      0
survived    0
sex         0
age         0  ← Fixed!
sibsp       0
parch       0
fare        0  ← Fixed!
embarked    0  ← Fixed!
```

**All zeros = clean data! ✓**

---
# Export the Cleaned Dataset as a CSV

```python
# Save cleaned data
# Writes the DataFrame to a new CSV file without adding extra row numbers (index)
df.to_csv("Titanic_Cleaned.csv", index=False)     

# Prints a confirmation message so the user knows the save was successful
print("✓ Cleaned data saved to 'Titanic_Cleaned.csv'")   
```
---
# Before and After Comparison

**Before cleaning:**
- 15 columns (many irrelevant)
- 263 missing ages
- 1 missing fare
- 2 missing embarked values

**After cleaning:**
- 8 relevant columns
- 0 missing values
- Lost only 2 rows (minimal!)

**Data is now ready for the next step!**

---

# Quick Check: When to Fill vs Drop?

| Situation | Strategy | Why? |
|-----------|----------|------|
| Many values missing (>20%) | Fill with median/mean/mode | Don't want to lose too much data |
| Few values missing (<5%) | Drop the rows | Minimal data loss |
| Column mostly empty (>50%) | Drop entire column | Not enough info to be useful |

---

# Common Mistakes to Avoid

❌ **Filling categorical data with median**
- Median doesn't make sense for categories!
- Use mode (most common value) instead

❌ **Dropping too many rows**
- If you drop >10%, may lose important patterns

❌ **Not verifying the fix**
- Always check `.isnull().sum()` after cleaning!

---

# Visual: The Cleaning Process

```
Raw Data (1309 rows, 15 columns)
         ↓
Select Features (1309 rows, 8 columns)
         ↓
Check Missing Values (age=263, fare=1, embarked=2)
         ↓
Fill age & fare with median
         ↓
Drop rows with missing embarked
         ↓
Clean Data (1307 rows, 8 columns, 0 missing!)
```

---

# Code Summary

```python
# Step: Select useful columns
df = df[['pclass', 'survived', 'sex', 'age', 
         'sibsp', 'parch', 'fare', 'embarked']]

# Step: Check for missing values
print(df.isnull().sum())

# Step: Fill missing numeric values
df['age'].fillna(df['age'].median(), inplace=True)
df['fare'].fillna(df['fare'].median(), inplace=True)

# Step: Drop rows with missing categorical values
df.dropna(subset=['embarked'], inplace=True)

# Step: Verify no missing values remain
print(df.isnull().sum())
```

---

# What's Next?

**Lesson 3**: Data Preparation
- Convert categorical data to numbers
- Why computers need numerical data
- Using `pd.get_dummies()` for one-hot encoding
- Separating features (X) from target (y)

**Remember**: Clean data → Better models!

---

# Practice Task

Complete **ai03bTasks.ipynb**:

1. Select the 8 useful columns
2. Check for missing values
3. Fill missing ages with median
4. Fill missing fares with median
5. Drop rows with missing embarked values
6. Verify all missing values are gone

**Bonus**: Try filling age with mean instead of median - compare the values!

---

# Key Terms to Remember

- **Feature Selection**: Choosing which columns to use
- **Missing Values**: Empty cells (NaN, None)
- **Imputation**: Filling missing values
- **Median**: Middle value (better for data with outliers)
- **Mean**: Average value
- **Mode**: Most common value
- **`.fillna()`**: Fill missing values
- **`.dropna()`**: Remove rows/columns with missing values
- **`inplace=True`**: Modify original DataFrame


