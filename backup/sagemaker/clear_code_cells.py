"""script to duplicate and empty the jupyter notebook cells"""
import nbformat

# Load the duplicated notebook
path = "/Users/vamsi_mbmax/Library/CloudStorage/OneDrive-Personal/01_vam_PROJECTS/LEARNING/proj_ML/subproj_mls_aws/backup/sagemaker/practise_cohort.ipynb"
with open(path, 'r', encoding='utf-8') as f:
    notebook = nbformat.read(f, as_version=4)

# Clear code cells
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        cell['source'] = ''  # Clear the code
        cell['outputs'] = [] # Clear the outputs

# Save the cleared notebook
with open(path, 'w', encoding='utf-8') as f:
    nbformat.write(notebook, f)
