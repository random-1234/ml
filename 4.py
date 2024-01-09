import csv
def find_s(training_data):
    hypothesis=[]
    hypothesis = training_data[0][:-1]
    for example in training_data:
        features = example[:-1]
        label = example[-1]
        if label == 'Yes':
            for i in range(len(hypothesis)):
                if hypothesis[i] != features[i]:
                    hypothesis[i] = '?'
                print(hypothesis)
    return hypothesis

training_data = []
with open('enjoysport.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        training_data.append(row)
    print(training_data)
    training_data.pop(0)
    print(training_data)

h = find_s(training_data)
print("Most specific hypothesis:", h)