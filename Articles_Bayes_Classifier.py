import pandas as pd
import unidecode as ud
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Read the Excel file
file_path = './data/Argentina_News.xlsx'
df = pd.read_excel(file_path)
laplace_correction_needed = False

# Define symbols and stopwords to remove
symbols = [".", ",","¡", "!", "¿", "?", "\"", "\'", "%", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0" "(", ")", ":", ";", "-",]

# Open stopwords text file and get words
file_path = './data/Spanish_Stopwords.txt'
with open(file_path, 'r') as file:
    stopwords = []
    for line in file:
        stopwords.append(line.strip())

stopwords = symbols + stopwords
titulares = df["titular"].tolist()
title_words = []

# Remove stopwords from the titles
for i, title in tqdm(enumerate(titulares), total = len(titulares), desc="Removing stopwords"):
    title_words = title.split(" ")
    final_words = []
    for word in title_words:
        word = (ud.unidecode(word.lower()))  # For unicode characters
        if word not in stopwords:
            final_word = "".join(w for w in word if w not in symbols)
            final_words.append(final_word)

    titulares[i] = " ".join(final_words)
    title_words = titulares[i].split(" ")
df["titular"] = titulares

# Remove impartial categories
df = df.loc[(df["categoria"] != "Noticias destacadas") & (df["categoria"] != "Destacadas")]
df = df.dropna(subset=['categoria'])

# Get category list
categories = sorted(df["categoria"].astype(str).unique())
categories_amount = len(categories)
# Partition dataset
percentage = 20 # Modify as needed
# Shuffle the DataFrame
shuffled_df = df.sample(frac=1)
# Calculate the number of rows for the first partition
partition_size = int(len(shuffled_df) * percentage / 100)

# Split the shuffled DataFrame into test and train partitions
test_p = shuffled_df.iloc[:partition_size]
train_p = shuffled_df.iloc[partition_size:]
test_p.reset_index(drop=True, inplace=True)
train_p.reset_index(drop=True, inplace=True)
print("Training entries: ", len(train_p))
print("Testing entries: ",len(test_p))

# Get Category frequency table
freq_table = {}
for cat in categories:
    freq_table[cat] = {}
for row, title in enumerate(train_p["titular"]):
    words = title.split(" ")
    title_cat = train_p["categoria"][row]
    for word in words:
        if not word in freq_table[title_cat]:
            freq_table[title_cat][word] = 1
        else:
            freq_table[title_cat][word] += 1

# Get Class probabilities
class_prob = {}
class_total = 0
for value in train_p["categoria"]:
    if value not in class_prob:
        class_prob[value] = 0
    class_prob[value] += 1
    class_total += 1
for key in class_prob:
    class_prob[key] /= class_total

# Initialize data for confusion matrix and ROC
predicted_cats = []
true_cats = []
# roc_thresholds = [i / 100 for i in range(10, 81)]
roc_thresholds = [i / 10 for i in range(0, 10)]
filtered_roc=[]
# Initialize lists to store probabilities and true labels
class_probs = []
class_true_labels = []

# Get Conditional probabilities and make predictions
for i, row in test_p.iterrows():
    probs = {}
    for cat in categories:
      probs[cat] = 0
    for class_name, class_p in class_prob.items():
      freq = sum(freq_table[class_name].values())
      curr_p = class_p
      words = row["titular"].split(" ")
      for w in words:
        occurs = freq_table[class_name][w] if w in freq_table[class_name] else 0
        curr_p *= (occurs + 1)/(freq + categories_amount)  # Laplace correction

      probs[class_name] = curr_p

    total = sum(probs.values())
    for k in probs:
      probs[k] /= total
    cat_prediction = max(probs, key=probs.get)
    predicted_cats.append(cat_prediction)
    cat_correct = row["categoria"]
    true_cats.append(cat_correct)





# Calculate confusion matrix
confusion = confusion_matrix(true_cats, predicted_cats, labels=categories)

# Display the confusion matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=categories, yticklabels=categories)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Calculate TP, FP, TN, FN
TP = confusion.diagonal()
FP = confusion.sum(axis=0) - TP
FN = confusion.sum(axis=1) - TP
TN = confusion.sum() - (TP + FP + FN)

# Calculate accuracy, precision, recall, false positive rate, and F1-score
# accuracy = (TP.sum() + TN.sum()) / confusion.sum()
accuracy = (TP + TN) / confusion.sum()
precision = TP / (TP + FP)
recall = TP / (TP + FN)
false_positive_rate = FP / (FP + TN)
f1_score = (2 * (precision * recall)) / (precision + recall)

# Print the metrics
for i, category in enumerate(categories):
    print(f"Category: {category}")
    print(f"Accuracy: {accuracy[i]:.4f}")
    print(f"Precision: {precision[i]:.4f}")
    print(f"Recall (True Positive Rate): {recall[i]:.4f}")
    print(f"False Positive Rate: {false_positive_rate[i]:.4f}")
    print(f"F1-Score: {f1_score[i]:.4f}")
    print("-" * 40)

#ROC
# Get Conditional probabilities and make predictions
print("Starting ROC")
full_roc_data=[]
for idx, roc_cat in enumerate(categories):
  cat_roc_data=[]
  for thresh in roc_thresholds:
    predicted_cats=[]
    true_cats=[]
    for i, row in test_p.iterrows():
        probs = {}
        for cat in categories:
          probs[cat] = 0
        for class_name, class_p in class_prob.items():
          freq = sum(freq_table[class_name].values())
          curr_p = class_p
          words = row["titular"].split(" ")
          for w in words:
            occurs = freq_table[class_name][w] if w in freq_table[class_name] else 0
            curr_p *= (occurs + 1)/(freq + categories_amount)  # Laplace correction

          probs[class_name] = curr_p

        total = sum(probs.values())
        for k in probs:
          probs[k] /= total
        cat_prediction = max(probs, key=probs.get)
        # probs=probs.get(cat_prediction)
        cat_prediction_prob=probs.get(roc_cat)

        if(cat_prediction_prob>thresh):
          predicted_cats.append('PASS')
        else:
          predicted_cats.append('FAIL')

        # if(row["categoria"]==cat_prediction and roc_cat==cat_prediction):
        if(row["categoria"]==roc_cat):
          cat_correct = 'PASS'
        else:
          cat_correct = 'FAIL'
        true_cats.append(cat_correct)

    confusion_ROC = confusion_matrix(true_cats, predicted_cats, labels=["PASS","FAIL"])
    TVP = confusion_ROC.item(0,0)/(confusion_ROC.item(0,0)+confusion_ROC.item(0,1))
    TFP = confusion_ROC.item(1,0)/(confusion_ROC.item(1,0)+confusion_ROC.item(1,1))
    cat_roc_data.append((TVP,TFP))
  full_roc_data.append(cat_roc_data)
  print("Finished ROC curve number "+str(idx+1)+"/"+str(len(categories)))

finalized_roc_data=[]
for cat in full_roc_data:
    roc_data_x=[]
    roc_data_y=[]
    for data in cat:
        roc_data_x.append(data[1])
        roc_data_y.append(data[0])
    roc_data_x.append(0)
    roc_data_y.append(0)
    finalized_roc_data.append((roc_data_x,roc_data_y))

plt.figure(figsize=(10, 8))
for idx,x in enumerate(finalized_roc_data):
  rounded_auc=round(auc(x[0], x[1]),5)
  plt.plot(x[0], x[1],label=categories[idx]+" AOC="+str(rounded_auc))
plt.title("ROC Curve")
plt.xlabel("TFP")
plt.ylabel("TVP")
plt.legend(loc ="lower right")
plt.show()