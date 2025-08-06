**XAI-Enhanced Adversarial Robustness for Phishing Detection in Zero-Trust Enterprises**

This project focuses on developing a robust and transparent phishing detection system for zero-trust enterprise environments. We leverage machine learning models, specifically Random Forest and XGBoost, to identify phishing attempts from both URLs and emails. A key aspect of this research is enhancing the models' resilience against adversarial attacks by incorporating adversarial training. Furthermore, we integrate Explainable AI (XAI) techniques, such as SHAP, to provide transparency into the model's decision-making process, which is crucial for establishing trust in a zero-trust framework.

Methodology
2.1 Data Acquisition
PhishTank Dataset
Nazario/Mendeley Dataset
Legitimate URLs
Synthetic Non-Phishing Emails
2.2 Data Preprocessing and Feature Engineering
Cleaning and Loading Datasets
Feature Extraction
Combining Datasets and Splitting
2.3 Model Development
Random Forest Classifier
XGBoost Classifier
2.4 Adversarial Attack Simulation
Generating Adversarial URLs
Evaluating Models on Adversarial Examples
2.5 Model Retraining
Retraining with Adversarial Data
Performance Assessment
2.6 Evaluation


Installation
To set up the project environment, you need to install the required Python libraries. You can do this by either using a requirements.txt file (recommended for reproducibility) or by installing the packages directly via pip.

Option 1: Using requirements.txt (Recommended)

If a requirements.txt file is provided in the repository, you can install all dependencies with a single command:

pip install -r requirements.txt
Option 2: Installing Directly

Alternatively, you can install the necessary libraries individually using the following command:

pip install scikit-learn pandas numpy textattack shap lime treeinterpreter xgboost beautifulsoup4
Ensure you have pip installed and accessible in your environment. It is recommended to use a virtual environment to avoid conflicts with other Python projects.


Dataset
The dataset used in this project is a comprehensive collection of phishing and legitimate data, comprising both URLs and emails. It was constructed by combining data from several sources and supplementing with synthetically generated data to ensure a balanced representation of classes.

Data Acquisition
PhishTank Dataset: Phishing URLs were obtained from the PhishTank database, a community-based archive of suspected phishing URLs. The raw data was initially in CSV format (phishtank_urls.csv). For this project, we filtered for 'verified' URLs and sampled approximately 3,500 entries to ensure a manageable and relevant subset.

Nazario/Mendeley Dataset: Phishing emails were sourced from the Nazario/Mendeley Phishing Corpus (private-phishing4.mbox). This dataset, originally in MBOX format, contains a collection of reported phishing emails. We processed this MBOX file to extract relevant email components (subject and body) and saved them into a CSV format (evil_email.csv). This yielded 3,523 phishing email samples.

Legitimate URLs: Legitimate URLs were extracted from the Mendeley Phishing Websites Dataset's index.sql file. This file contains a large collection of website data, including URLs labeled as legitimate (result=0). We parsed this SQL file to extract approximately 3,500 legitimate URLs, ensuring a balanced representation with the phishing URL subset.

Synthetic Non-Phishing Emails: To balance the Nazario phishing emails, we generated 3,523 synthetic non-phishing emails. These emails were created with simple, non-malicious content (subjects like "Meeting tomorrow", "Project update" and generic body text) to represent typical legitimate email communication.


**Data Preprocessing and Feature Engineering**
Following data acquisition, each dataset underwent preprocessing and feature engineering steps:

Cleaning and Loading: The acquired data files (phishtank_urls.csv, evil_email.csv, index.sql, and the synthetically generated non-phishing data) were loaded into pandas DataFrames. Initial cleaning involved handling missing values, particularly in the subject and message fields of the email data, by replacing them with empty strings.

Feature Extraction: For both URLs and emails, we extracted simple, yet effective, handcrafted features:

length: The total number of characters in the URL or email text.
special_chars: The count of special characters (e.g., `!@#$%^&*()_+-=[]{}|;:,.<>?/~``) present in the text.
has_login: A binary indicator (1 or 0) denoting the presence of the word "login" (case-insensitive) in the text.
has_https: A binary indicator (1 or 0) denoting the presence of "https" (case-insensitive) in the text, relevant primarily for URLs but included for consistency.
These features were chosen for their simplicity and potential to capture common characteristics of phishing attempts without requiring complex natural language processing or deep URL parsing.

Combining Datasets and Splitting: The processed DataFrames (phishing URLs with extracted features, phishing emails with extracted features, legitimate URLs with extracted features, and synthetic non-phishing emails with extracted features) were concatenated into a single combined dataset. This dataset includes the extracted features and a label column (1 for phishing, 0 for legitimate/non-phishing). A type column was added to differentiate between URLs and emails. This combined dataset was saved as combined_data.csv.

The combined dataset was then split into training and testing sets using an 80/20 ratio (X_train.csv, X_test.csv, y_train.csv, y_test.csv) to prepare the data for model development and evaluatio
Standard Metrics
Adversarial Robustness and Evasion Rate
Zero-Trust Framing and Interpretability (SHAP Analysis)


Usage
This project can be run as a Jupyter Notebook in environments like Google Colab, Jupyter Notebook, or VS Code with the Python extension.

Running as a Jupyter Notebook
Access the Notebook: Open the .ipynb notebook file in your preferred environment (Google Colab, Jupyter Notebook, or VS Code).
Install Dependencies: Ensure all necessary libraries are installed by running the cell containing the installation commands (or by installing from requirements.txt if provided).
Execute Cells Sequentially: Run each code cell in the notebook in the order they appear. The notebook is structured to follow the methodology, so executing cells sequentially will perform data loading, preprocessing, model training, adversarial simulation, retraining, and evaluation.
Review Outputs: Review the output of each cell, including printed messages, DataFrame previews, and plots, to follow the progress and results of the analysis.
Running as a Python Script (Optional)
If you prefer to run the core logic as a Python script (assuming the notebook code is adapted into a .py file), you would typically execute it from your terminal.

Save as Python Script: Convert or adapt the relevant code cells from the notebook into a single Python file (e.g., run_phishing_detection.py).

Install Dependencies: Ensure all dependencies are installed in your Python environment (see Installation section).

Run the Script: Execute the script from your terminal using the Python interpreter:

python run_phishing_detection.py
Check Outputs: The script will print progress and results to the console. Output files (e.g., combined_data.csv, model files) will be saved in the directory where the script is executed.

(Note: The provided notebook is designed for interactive use. Running as a script might require modifications to handle file paths, user inputs, or command-line arguments depending on the desired workflow.)

Methodology
This project implemented a structured methodology to develop and evaluate a robust phishing detection system, incorporating adversarial training and XAI for enhanced transparency in a zero-trust context. The process involved several key stages, from data acquisition and preprocessing to model development, adversarial simulation, retraining, and comprehensive evaluation.

2.1 Data Acquisition
The dataset was compiled from multiple sources to create a diverse and balanced representation of both phishing and legitimate data across URLs and emails.

PhishTank Dataset: Phishing URLs were sourced from the publicly available PhishTank database (phishtank.com). The phishtank_urls.csv file, containing over 43,000 entries, was filtered to include only 'verified' phishing URLs. A subset of 3,500 verified URLs was then sampled to serve as the phishing URL component of the dataset.
Nazario/Mendeley Dataset: Phishing emails were obtained from the Nazario/Mendeley Phishing Corpus, specifically the private-phishing4.mbox file. This MBOX file, containing a collection of phishing emails, was parsed using Python's email library to extract the subject, sender, and message body of each email. This process yielded 3,523 phishing email samples, which were saved to evil_email.csv.
Legitimate URLs: Legitimate URLs were extracted from the index.sql file of the Mendeley Phishing Websites Dataset. This SQL file contains entries for numerous websites with a 'result' field indicating their legitimacy (0 for legitimate, 1 for phishing). The file was parsed to identify and extract legitimate URLs (where 'result' was 0), and approximately 3,500 legitimate URLs were selected.
Synthetic Non-Phishing Emails: To create a balanced dataset, 3,523 synthetic non-phishing emails were generated. These emails were constructed using simple, benign subjects and body texts to simulate typical non-malicious email communication.
2.2 Data Preprocessing and Feature Engineering
Once the data was acquired, it underwent cleaning, feature extraction, and combination into a single dataset.

Cleaning and Loading: The raw data from PhishTank URLs (phishtank_urls.csv), Nazario emails (evil_email.csv), legitimate URLs (parsed from index.sql), and synthetic non-phishing emails were loaded into pandas DataFrames. Initial cleaning focused on handling missing values, particularly in the email subject and message fields, by filling them with empty strings to prevent errors during feature extraction. The processed phishing URLs and Nazario emails were saved as phishtank_clean.csv and nazario_clean.csv respectively.
Feature Extraction: For each data point (URL or email text), a set of simple, handcrafted features was extracted. These features included the total length of the text, the count of special_chars (common punctuation and symbols), a binary flag has_login indicating the presence of the substring "login" (case-insensitive), and a binary flag has_https indicating the presence of "https" (case-insensitive), primarily relevant for URLs. These features were chosen for their computational efficiency and relevance to identifying suspicious patterns in both URLs and emails. The features were extracted and added to the respective dataframes, saved as phishtank_features.csv and nazario_features.csv.
Combining Datasets and Splitting: The processed datasets (phishing URLs, phishing emails, legitimate URLs, and synthetic non-phishing emails), each with the extracted features and a label column (1 for phishing, 0 for legitimate), were concatenated into a single DataFrame. A type column ('url' or 'email') was added to distinguish the data source. This combined dataset, containing a balanced representation of approximately 14,046 entries (7,023 phishing and 7,023 legitimate), was saved as combined_data.csv. The dataset was then split into training (80%) and testing (20%) sets (X_train.csv, X_test.csv, y_train.csv, y_test.csv) to prepare for model training and evaluation.
2.3 Model Development
Two machine learning models, Random Forest and XGBoost, were selected for phishing detection based on their strong performance in classification tasks.

Random Forest Classifier: A Random Forest classifier was trained on the extracted features of the training data (X_train, y_train). The model was configured with 100 estimators and a random state for reproducibility.
XGBoost Classifier: An XGBoost classifier was also trained on the same training data. XGBoost, an optimized distributed gradient boosting library, is known for its speed and performance. The model was configured with use_label_encoder=False and eval_metric='logloss' for compatibility and standard evaluation.
The initial performance of both models was evaluated on the original test set (X_test, y_test) using standard metrics including accuracy, precision, recall, and F1 score.

2.4 Adversarial Attack Simulation
To assess the robustness of the trained models, an adversarial attack simulation was conducted by generating perturbed versions of the phishing URLs.

Generating Adversarial URLs: A simple adversarial perturbation technique was applied to the phishing URLs from the PhishTank dataset. This involved adding legitimate-looking subdomains (e.g., secure., login., www.) or paths (e.g., /home, /secure) to the original phishing URLs. This method aims to create URLs that might visually appear less suspicious or trick feature-based detection methods. The features (length, special characters, has_login, has_https) were re-extracted for these adversarial URLs, and the resulting dataset was saved as adversarial_urls.csv.
Evaluating Models on Adversarial Examples: The trained Random Forest and XGBoost models were then evaluated on the dataset of adversarial URLs (X_adversarial) using their original labels (1, phishing). This step measured how well the models could detect phishing attempts when presented with slightly modified, potentially deceptive inputs. The performance metrics (accuracy, precision, recall, F1 score) on this adversarial test set highlighted the models' vulnerability to this type of attack.
2.5 Model Retraining
To enhance the robustness of the phishing detection system, the Random Forest model was retrained to include adversarial examples in its training data.

Retraining with Adversarial Data: The original training data (X_train, y_train) was augmented with the adversarial URL examples (X_test_adv, y_test where y_test corresponds to the original phishing labels for the perturbed URLs). The Random Forest model (rf_retrained) was then retrained on this combined dataset (X_train_adv, y_train_adv). This process is a form of adversarial training, aiming to make the model more resilient to adversarial perturbations by exposing it to such examples during the training phase. The retrained model was saved using joblib as rf_retrained_model.pkl.
Performance Assessment: The performance of the retrained Random Forest model was evaluated on both the original test set (X_test, y_test) and the adversarial test set (X_test_adv, y_test). This evaluation allowed for a direct comparison of the retrained model's accuracy, precision, recall, and F1 score against the original model, particularly its improved ability to correctly classify adversarial examples. The adversarial evasion rate, calculated as 1 minus the accuracy on the adversarial test set, quantified the percentage of adversarial examples that successfully evaded detection by the retrained model.
2.6 Evaluation
The final evaluation phase assessed the overall effectiveness and interpretability of the retrained Random Forest model, particularly within the framework of a zero-trust enterprise.

Standard Metrics: The performance of the retrained model was reported using standard classification metrics (accuracy, precision, recall, and F1 score) on both the original and adversarial test sets to provide a comprehensive view of its detection capabilities.
Adversarial Robustness and Evasion Rate: A key evaluation point was the model's robustness to adversarial attacks, measured by its accuracy on the adversarial test set and the corresponding evasion rate. Improved performance on adversarial data after retraining demonstrated the effectiveness of the adversarial training approach.
Zero-Trust Framing and Interpretability (SHAP Analysis): To align with zero-trust principles of continuous verification and transparency, SHAP (SHapley Additive exPlanations) analysis was performed on the retrained Random Forest model using the original test set (X_test). SHAP values were computed to explain the contribution of each feature (length, special_chars, has_login, has_https) to the model's prediction for individual instances. The SHAP summary plot provided a global understanding of feature importance and how each feature impacts the model's output across the dataset. This interpretability is vital in a zero-trust environment, allowing security analysts to understand why a particular input is flagged as phishing and enabling auditing and compliance with regulations like GDPR and CCPA.


File Structure
This repository contains several files essential for the phishing detection project, including raw datasets, processed data, model files, and the main notebook.

phishtank_urls.csv: Original raw dataset from PhishTank containing phishing URLs.
private-phishing4.mbox: Original raw dataset from the Nazario/Mendeley Phishing Corpus containing phishing emails in MBOX format.
index.sql: Original raw dataset from the Mendeley Phishing Websites Dataset containing legitimate URLs and other website data in SQL format.
evil_email.csv: Processed CSV file containing phishing emails extracted from private-phishing4.mbox, with 'subject', 'sender', and 'message' fields.
phishtank_clean.csv: Cleaned and sampled phishing URLs from phishtank_urls.csv, containing 'url' and 'label' (1 for phishing).
nazario_clean.csv: Cleaned phishing emails from evil_email.csv, containing 'message' and 'label' (1 for phishing), with missing values handled.
legit_urls_features.csv: Processed legitimate URLs extracted from index.sql with extracted features (length, special_chars, has_login, has_https) and label (0 for legitimate).
non_phishing_features.csv: Synthetically generated non-phishing emails with extracted features and label (0 for non-phishing).
phishtank_features.csv: Phishing URLs from phishtank_clean.csv with extracted features.
nazario_features.csv: Phishing emails from nazario_clean.csv with extracted features.
combined_data.csv: The unified dataset combining processed phishing URLs, phishing emails, legitimate URLs, and synthetic non-phishing emails, including extracted features and labels. Used for training and testing.
X_train.csv: Features for the training set, split from combined_data.csv.
X_test.csv: Features for the original test set, split from combined_data.csv.
y_train.csv: Labels for the training set.
y_test.csv: Labels for the original test set.
adversarial_urls.csv: Adversarially perturbed phishing URLs generated from phishtank_features.csv with extracted features and label (1 for phishing).
X_test_adv.csv: Features for the adversarial test set, used to evaluate the retrained model's robustness.
rf_retrained_model.pkl: The retrained Random Forest model saved using joblib.
phishing_detection_notebook.ipynb: The main Jupyter Notebook file containing all the code for data acquisition, preprocessing, model development, adversarial simulation, retraining, and evaluation.



Future Work
This project lays the groundwork for developing robust and transparent phishing detection systems. Several avenues for future research and development can further enhance this work:

Exploring More Sophisticated Feature Engineering: Investigate advanced feature extraction techniques, including Natural Language Processing (NLP) for detailed analysis of email content and more in-depth domain name analysis for URLs (e.g., using domain reputation scores, analyzing subdomains and TLDs).
Implementing and Evaluating Other Machine Learning Models: Explore the performance and robustness of other machine learning algorithms suitable for text and URL classification, such as Support Vector Machines (SVMs), Neural Networks (e.g., LSTMs, Transformers for text), or ensemble methods beyond Random Forest and XGBoost.
Investigating More Advanced Adversarial Attack Methods: Implement and evaluate the system's robustness against more sophisticated adversarial attack techniques. This could involve using libraries like TextAttack with compatible configurations for text-based attacks on emails or exploring other adversarial libraries for generating more nuanced URL perturbations. Overcoming compatibility issues with TextAttack and newer scikit-learn versions would be a key step here.
Developing and Evaluating More Advanced Adversarial Training Techniques: Implement and compare different adversarial training strategies beyond simply augmenting the training data with adversarial examples. Techniques like Projected Gradient Descent (PGD) or other defense mechanisms could be explored to further enhance model resilience.
Expanding the Dataset with More Diverse Data: Incorporate larger and more diverse datasets, including a wider range of phishing types (e.g., SMS phishing, voice phishing) and a broader collection of legitimate URLs and emails from various sources to improve generalization.
Deploying the Model as a Real-time Detection Service: Develop a practical application or API to deploy the trained model as a real-time phishing detection service that can be integrated into enterprise security systems. This would involve considerations for efficiency, scalability, and continuous monitoring.
Conducting a More In-depth Analysis of SHAP Values: Perform a deeper dive into the SHAP analysis to identify specific patterns in feature contributions for different types of phishing attacks or to understand the model's behavior on misclassified examples. This could involve analyzing individual instance explanations and clustering similar explanations.
