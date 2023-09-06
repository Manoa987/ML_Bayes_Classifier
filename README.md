# Data Classification and Analysis Project

This project addresses the implementation and application of naive Bayes classifier algorithms in different contexts. The project is divided into three main parts:

## Part 1: Cultural Preferences Classification (Simple_Bayes_Classifier.py)

In this part, we implement a naive Bayes classifier to classify the cultural preferences of individuals based on a vector of binary attributes. The vector includes features such as scones, beer, whiskey, oats, and soccer. The goal is to determine whether a person is English or Scottish based on their preferences.

### Usage Instructions:

- The file "Britanic_Tastes.xlsx" contains preferences of a group of English and Scottish individuals.
- We implement the classifier and apply it to preference examples, such as x1 = (1, 0, 1, 1, 0) and x2 = (0, 1, 1, 0, 1), to determine if they correspond to an English or Scottish person.

## Part 2: Text Classification (Articles_Bayes_Classifier.py)

In this part, we develop a text classifier using the naive Bayes classifier. We use a dataset called "Argentina_News.xlsx" to classify news articles into different categories. The goal is to evaluate the classifier's performance in terms of accuracy and other evaluation metrics.

### Usage Instructions:

- We use at least four categories to classify the news articles.
- We split the text dataset into training and testing sets.
- We construct a confusion matrix and calculate evaluation metrics such as Accuracy, Precision, True Positive Rate, False Positive Rate, and F1-score.
- We calculate the ROC curve to assess the classifier's performance.

## Part 3: University Admission Analysis (Bayes_Network_Classifier.py)

In this part, we analyze a dataset "University_Admissions.csv" containing information about student admissions to a university. The variables include the admission decision, GRE scores, GPA, and the rank of the high school attended by students.

### Usage Instructions:

- We discretize the GRE and GPA variables and explore the relationships between them and admission.
- We calculate the probability that a student from a rank 1 school was not admitted.
- We calculate the probability that a student from a rank 2 school with GRE = 450 and GPA = 3.5 was admitted.
- We reflect on the learning process in this exercise.

This project provides an opportunity to understand and apply the naive Bayes classifier with networks in different classification and data analysis contexts.
