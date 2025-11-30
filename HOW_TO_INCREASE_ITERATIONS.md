# How to Increase the Number of Iterations

There are several ways to increase training iterations. Choose the method that works best for you!

---

## **Method 1: Use the Simple Script** (EASIEST!)

I've created `train_with_custom_iterations.py` for easy experimentation.

### Step 1: Open the file
```bash
nano train_with_custom_iterations.py
# or use any text editor
```

### Step 2: Change these two lines (around line 59-62):
```python
# CHANGE THIS NUMBER for more iterations
n_iterations = 500  # 👈 Change from 500 to any number you want!

# CHANGE THIS NUMBER for different learning rate
learning_rate = 1.0  # 👈 Recommended: 0.3 to 2.0
```

### Step 3: Run it
```bash
python3 train_with_custom_iterations.py
```

**Examples:**
- For **100 iterations**: Change to `n_iterations = 100`
- For **500 iterations**: Change to `n_iterations = 500`
- For **1000 iterations**: Change to `n_iterations = 1000`

---

## **Method 2: Modify Existing Scripts**

### For `step2_sigmoid_training.py`:

**Find this line (around line 46):**
```python
n_iterations = 20
```

**Change it to:**
```python
n_iterations = 500  # or any number you want
```

### For `step3_full_analysis.py`:

**Find this line (around line 45):**
```python
n_iterations = 20
```

**Change it to:**
```python
n_iterations = 500  # or any number you want
```

Then run:
```bash
python3 step3_full_analysis.py
```

---

## **Method 3: Modify the .env File**

You can add iteration settings to your `.env` file for easy configuration.

### Step 1: Edit `.env`
```bash
nano .env
```

### Step 2: Add this line:
```
NUM_ITERATIONS=500
```

### Step 3: Modify your scripts to read from .env
In your Python script, add:
```python
import os
from dotenv import load_dotenv

load_dotenv()
n_iterations = int(os.getenv('NUM_ITERATIONS', 20))  # Default 20 if not set
```

---

## **Method 4: Use Python Code Directly**

You can create your own script:

```python
from dataset_generator import DatasetGenerator
from sigmoid_generator import SigmoidGenerator

# Generate data
generator = DatasetGenerator(n_samples=6000)
data, labels = generator.generate(mean0=(0.3, 0.3), mean1=(0.7, 0.7), std=0.1)

# Train with YOUR choice of iterations
n_iterations = 1000  # 👈 CHANGE THIS!
learning_rate = 1.0  # 👈 CHANGE THIS!

model = SigmoidGenerator(beta0=[0, 0.3, -0.5], eta=learning_rate)
model.train(data, labels, n_iterations=n_iterations)

# Check accuracy
predictions = model.predict(data)
accuracy = (predictions == labels).mean() * 100
print(f"Accuracy with {n_iterations} iterations: {accuracy:.2f}%")
```

---

## **Method 5: Command Line Arguments**

Create a script that accepts iterations as command-line argument:

```python
# train_model.py
import sys
from dataset_generator import DatasetGenerator
from sigmoid_generator import SigmoidGenerator

if len(sys.argv) > 1:
    n_iterations = int(sys.argv[1])
else:
    n_iterations = 100  # default

# ... rest of the code ...
```

Run with:
```bash
python3 train_model.py 500    # Train with 500 iterations
python3 train_model.py 1000   # Train with 1000 iterations
```

---

## **Recommended Settings for Different Goals**

| Goal | Iterations | Learning Rate | Expected Accuracy | Time |
|------|-----------|---------------|-------------------|------|
| Quick test | 10-20 | 0.3 | ~55-60% | Fast |
| Good results | 100 | 0.3-1.0 | ~98-99% | Medium |
| Best accuracy | 500 | 1.0 | ~99.7-99.8% | Slower |
| Maximum | 1000+ | 1.0-2.0 | ~99.8%+ | Slowest |

---

## **Quick Start Examples**

### Example 1: Train with 100 iterations
```bash
python3 train_with_custom_iterations.py
# (Default is set to 500, edit line 59 to change to 100)
```

### Example 2: Best performance (500 iterations)
Edit line 59 to:
```python
n_iterations = 500
```
Edit line 62 to:
```python
learning_rate = 1.0
```

### Example 3: Maximum accuracy (1000 iterations)
Edit line 59 to:
```python
n_iterations = 1000
```
Edit line 62 to:
```python
learning_rate = 2.0
```

---

## **Important Notes**

⚠️ **Training Time:** More iterations = longer training time
- 20 iterations: ~1 second
- 100 iterations: ~2-3 seconds
- 500 iterations: ~10-15 seconds
- 1000 iterations: ~20-30 seconds

⚠️ **Diminishing Returns:** After ~200-500 iterations, accuracy improvements become very small

✅ **Recommended:** Start with 100-200 iterations for good balance of speed and accuracy

---

## **Testing Different Values**

Want to test multiple iteration counts? Try this:

```python
for n_iter in [50, 100, 200, 500]:
    print(f"\nTesting {n_iter} iterations...")
    model = SigmoidGenerator(beta0=[0, 0.3, -0.5], eta=1.0)
    model.train(data, labels, n_iterations=n_iter)
    accuracy = (model.predict(data) == labels).mean() * 100
    print(f"Accuracy: {accuracy:.2f}%")
```

---

**Need help?** Run the simple script first:
```bash
python3 train_with_custom_iterations.py
```

Then modify the numbers on lines 59-62 to experiment!
