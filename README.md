# BU, ExtremeXP: Meta-learning Search Engine
Application repository for the proof-of-concept Meta-learning Search Engine developed as a small part of the wider Bournemouth University contribution to the ExtremeXP project.
(https://www.bournemouth.ac.uk/research/projects/extremexp).

<br>

## Implementation Overview
This tool aims to provide a pipeline where users compare algorithm performance on different datasets, prioritised according to statistical measures of similarity, thereby analysing which algorithms are more likely to be effective for an individual use case.
<br>
![image](https://github.com/user-attachments/assets/bc999c48-470f-46ac-9fd1-b4ed9227feb2)

<br>

Users upload & compare different datasets. Datasets are split into:
- 1. **‘Reference Dataset’ (i.e. ‘independent’ variable)** - A singular Dataset which requires an algorithm 'reccomendation'
- 2. **‘Comparison Datasets’ (i.e. ‘dependent’ variables)** - Multiple Datasets for the ‘Reference Dataset’ to be compared against (calculate similairty between them)
- 3. **‘Performance Metrics’ (i.e. algorithm statistics)** - For each ‘Comparison Dataset’ containing the performance information of the algorithms that were run on the Dataset (used for algorithm 'reccomendation')

<br>

There are two data upload mechanisms:
- **Interface upload** - All Datasets uploaded to be used in the comparison are stored temporarily in session memory.
- **Database upload** - Comparison datasets can be uploaded to a local MySQL Database for use in subsequent comparison operation with uloaded Reference Datasets

<br>

A ‘task type’ can be set to filter uploaded algorithm performance according to chosen type (Classification/ Regression)

<br>
<br>

## Setting up and running the application

### 1. Create a Python Virtual Environment (venv)
- In the Command Prompt, run:
  - Navigate to an appropriate directory for a venv folder to be created.
  - Run: `python -m venv venv_name` (replacing venv_name).

<br>

### 2. Activate venv
- In the Command Prompt:
  - Make sure you are still in the same directory as in Step 1.
  - Run: `venv_name\Scripts\activate`

<br>

### 3. Install dependencies into venv
- In the Command Prompt:
  - Navigate to the top-level of the repository (.../metalearning)
  - Run: `pip install -r requirements.txt` to install required dependencies.

<br>

### 4. Deploy/run a local version of the application
- Make sure your venv is currently/still activated before progressing.

- In the Command Prompt:
  - Navigate to the top-level of the repository (.../metalearning).
  - Run: `streamlit run main.py` to launch the application.

<br>
