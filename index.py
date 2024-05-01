import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import spacy
from collections import defaultdict

# Step 1: Data Acquisition
try:
    intents_train = pd.read_csv("C:/Users/jaina/Documents/archive/intents_train.csv")
except FileNotFoundError:
    print("Error: Training dataset file not found.")
    exit()

# Step 2: Prepare the data for training
X = intents_train.iloc[:, 1]  # Assuming the second column contains user utterances
y = intents_train.iloc[:, 0]  # Assuming the first column contains intents

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Intent Recognition
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)

classifier = SVC(kernel='linear')
classifier.fit(X_train_tfidf, y_train)

# Step 5: Entity Extraction
nlp = spacy.load("en_core_web_sm")

# Step 6: Slot Filling and Context Handling
def fill_slots(intent, entities, conversation_context):
    slots = defaultdict(str)
    if intent == 'atis_flight':
        for entity, label in entities:
            if label == 'GPE':
                if 'departure_city' not in slots:
                    slots['departure_city'] = entity
                elif 'arrival_city' not in slots:
                    slots['arrival_city'] = entity
            elif label == 'ORG':
                slots['airline'] = entity
            elif label == 'TIME':
                slots['departure_time'] = entity
    elif intent == 'atis_airfare':
        for entity, label in entities:
            if label == 'GPE':
                if 'departure_city' not in slots:
                    slots['departure_city'] = entity
                elif 'arrival_city' not in slots:
                    slots['arrival_city'] = entity
            elif label == 'ORG':
                slots['airline'] = entity
    conversation_context.update(slots)
    return conversation_context

# Step 7: Dialog Management
def manage_dialog(user_utterance, classifier, vectorizer, conversation_context):
    user_utterance_tfidf = vectorizer.transform([user_utterance])
    predicted_intent = classifier.predict(user_utterance_tfidf)[0]
    
    doc = nlp(user_utterance)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    conversation_context = fill_slots(predicted_intent, entities, conversation_context)
    
    response = generate_response(predicted_intent, conversation_context)
    
    print("User utterance:", user_utterance)
    print("Predicted Intent:", predicted_intent)
    print("Extracted Entities:", entities)
    print("Response:", response)
    print()

def generate_response(intent, slots):
    if intent == 'atis_flight':
        if 'departure_city' in slots and 'arrival_city' in slots:
            response = f"You want to fly from {slots['departure_city']} to {slots['arrival_city']}."
            if 'airline' in slots:
                response += f" Airline: {slots['airline']}."
            if 'departure_time' in slots:
                response += f" Departure time: {slots['departure_time']}."
            return response
        else:
            return "Please provide both departure and arrival cities for your flight."
    elif intent == 'atis_airfare':
        if 'departure_city' in slots and 'arrival_city' in slots and 'airline' in slots:
            response = f"The airfare for flights from {slots['departure_city']} to {slots['arrival_city']} on {slots['airline']} is $XXX."
            return response
        else:
            return "Please provide both departure and arrival cities, and the airline for the airfare inquiry."
    return "Sorry, I couldn't understand your request."

# Step 8: Model Evaluation
X_test_tfidf = vectorizer.transform(X_test)
y_pred = classifier.predict(X_test_tfidf)

intent_accuracy = accuracy_score(y_test, y_pred)
intent_precision = precision_score(y_test, y_pred, average='weighted')
intent_recall = recall_score(y_test, y_pred, average='weighted')
intent_f1 = f1_score(y_test, y_pred, average='weighted')

print("Intent Recognition Metrics:")
print("Accuracy:", intent_accuracy)
print("Precision:", intent_precision)
print("Recall:", intent_recall)
print("F1 Score:", intent_f1)

# Step 9: Simulated Dialog Testing
print("\nSimulated Dialog Testing:")
conversation_context = defaultdict(str)
for user_utterance, intent in zip(X_test, y_test):
    manage_dialog(user_utterance, classifier, vectorizer, conversation_context)
