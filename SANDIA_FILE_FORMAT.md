# SANDIA FLAME D FILE FORMAT GUIDE

## What You're Seeing

You have **statistics files** (not raw measurement files):

```
D01.Yave  ← Time-averaged scalar data
D01.Ycnd  ← Conditional averaged data
D01.Yfav  ← Favré-averaged data

D02.Yave
D02.Ycnd
D02.Yfav

... and so on for D03, D15, D30, D45, etc.
```

---

## FILE TYPES EXPLAINED

### .Yave (Favre-time-averaged data) ← **USE THIS ONE**
- Most commonly used for model validation
- Contains: Mean values + standard deviations
- Data at different radial positions (y) at fixed axial location (x)
- **This is what you should use for your validation!**

### .Ycnd (Conditional averaged data)
- Specialized conditional statistics
- Requires special interpretation
- Optional for this project

### .Yfav (Favre-averaged data)
- Density-weighted averages
- Also good for validation
- Alternative to .Yave

---

## WHICH FILES TO USE FOR VALIDATION

**Best choice: .Yave files**
- D01.Yave ← Flame D at x/d = 1
- D02.Yave ← Flame D at x/d = 2
- D03.Yave ← Flame D at x/d = 3
- D15.Yave ← Flame D at x/d = 15
- D30.Yave ← Flame D at x/d = 30
- D45.Yave ← Flame D at x/d = 45

**Recommendation**: Use D15, D30, D45 (downstream locations with well-developed flame)

---

## HOW TO PARSE THESE FILES

These are text files. You can:

### Option 1: Open in text editor
```
1. Right-click D01.Yave
2. Open with → Notepad
3. Look at first few lines to understand format
```

### Option 2: Use Python to read
```python
import pandas as pd

# Try reading the file
df = pd.read_csv("pmD.stat/D01.Yave", 
                 delim_whitespace=True, 
                 comment='#', 
                 engine='python')

print(df.head())
print(df.columns)
```

---

## NEXT STEP: INSPECT A FILE

Let me create a script to **read and display** one of these files so you can see the structure:

**Create file: `inspect_sandia_file.py`**

```python
#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

# Try to find and read a Sandia file
sandia_folder = Path("pmCDEF/pmD.stat")

if sandia_folder.exists():
    # Find first .Yave file
    yave_files = sorted(sandia_folder.glob("*.Yave"))
    
    if yave_files:
        test_file = yave_files[0]
        print(f"Reading: {test_file.name}")
        
        try:
            df = pd.read_csv(test_file, 
                           delim_whitespace=True,
                           comment='#',
                           engine='python')
            
            print(f"\nShape: {df.shape}")
            print(f"\nColumns: {list(df.columns)}")
            print(f"\nFirst 10 rows:")
            print(df.head(10))
            
            # Save for inspection
            df.to_csv("sandia_sample_data.csv", index=False)
            print("\nSaved sample to: sandia_sample_data.csv")
            
        except Exception as e:
            print(f"Error: {e}")
            # Try reading as text
            print("\nReading as raw text:")
            with open(test_file) as f:
                for i, line in enumerate(f):
                    if i < 20:
                        print(line.rstrip())
                    else:
                        break
else:
    print(f"Folder not found: {sandia_folder}")
```

---

## WHAT TO DO NOW

### Step 1: Find the pmD.stat folder
Look in your downloaded pmCDEF folder for:
```
pmCDEF/
  pmD.stat/        ← This is what you have
    D01.Yave
    D01.Ycnd
    D01.Yfav
    D02.Yave
    ... etc
```

### Step 2: Examine one file
Open `D15.Yave` in Notepad to see what's inside:
- First line: column headers
- Following lines: data

### Step 3: Tell me what columns you see
Share a screenshot or tell me what columns are in D15.Yave
- Is there: x, y, T, CH4, O2, velocity, etc.?
- What are the exact column names?

### Step 4: I'll update the parser
Once I know the exact column names, I'll update:
- `sandia_yall_parser.py` → to read .Yave files
- `sandia_data_mapper.py` → to map correctly

---

## SUMMARY

```
You have: Statistics files (.Yave, .Ycnd, .Yfav)
Use: .Yave files (time-averaged, most standard)
For validation: D15.Yave, D30.Yave, D45.Yave (downstream locations)
Next: Open one file to see column structure
```

This is perfect for validation! Just need to see the actual column names.

**Send me:** A screenshot of D15.Yave opened in Notepad (first 20 lines) or tell me what columns you see!
