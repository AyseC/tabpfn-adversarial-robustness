"""Add stratify=y to all train_test_split calls"""
import re
import glob

files = glob.glob('run_*_experiment.py') + glob.glob('run_*_nes_experiment.py')

for filepath in files:
    # Skip synthetic files - they use stratify already or don't need it
    if 'synthetic' in filepath:
        continue
        
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check if stratify is missing in train_test_split
    if 'train_test_split(' in content and 'stratify=y' not in content:
        # Add stratify=y before the closing parenthesis
        content = re.sub(
            r'(train_test_split\([^)]+random_state=42)\)',
            r'\1, stratify=y)',
            content
        )
        
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"✓ Fixed: {filepath}")
    else:
        print(f"  OK: {filepath}")

print("\nDone!")
