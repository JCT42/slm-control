# This is a small script to fix the error_history attribute issue in pattern_gen_2.0.py
import re

# Read the original file
with open('pattern_gen_2.0.py', 'r') as f:
    content = f.read()

# Add error_history initialization to PatternGenerator.__init__
pattern = r'(self\.mixing_parameter = mixing_parameter\n\s*)'
replacement = r'\1        self.error_history = []\n    '
content = re.sub(pattern, replacement, content)

# Fix the optimize method to use self.error_history
pattern = r'field = initial_field\.copy\(\)\n\s*error_history = \[\]'
replacement = 'field = initial_field.copy()\n        self.error_history = []'
content = re.sub(pattern, replacement, content)

pattern = r'(current_error = self\.calculate_error\(field, algorithm\)\n\s*)error_history\.append\(current_error\)'
replacement = r'\1self.error_history.append(current_error)'
content = re.sub(pattern, replacement, content)

pattern = r'return field, error_history, stop_reason'
replacement = 'return field, self.error_history, stop_reason'
content = re.sub(pattern, replacement, content)

# Add try-except block for error_history access
pattern = r'(self\.reconstruction = self\.reconstruction / np\.max\(self\.reconstruction\)\n\s*)\n\s*self\.error_history = self\.pattern_generator\.error_history'
replacement = r'\1\n            try:\n                self.error_history = self.pattern_generator.error_history\n            except AttributeError:\n                print("Warning: pattern_generator has no error_history attribute")\n                self.error_history = []'
content = re.sub(pattern, replacement, content)

# Write the modified content back to the file
with open('pattern_gen_2.0.py', 'w') as f:
    f.write(content)

print("Fixed error_history attribute issue in pattern_gen_2.0.py")
