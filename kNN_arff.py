import sys
import math
import heapq
import scipy.io.arff as arff


train_file = sys.argv[1]
test_file = sys.argv[2]
k = int(sys.argv[3])

train_data = []
train_labels = []
flag = 0
feature_vec_length = 0

#Parse the ARFF file to get the training data, features
data, meta = arff.loadarff(train_file)
for itemd in data:
	l = []
	for i in range(len(itemd)):
		l.append(str(itemd[i]))
	train_data.append(l)

mstr = str(meta)
mstr = mstr.split("\n\t")
for im in mstr:
	if "type is" in im:
		feature_vec_length += 1
	if im.startswith("class"):
		flag = 0
		train_labels = im.split("(")[1].split(")")[0].replace(" ","").replace("'","").split(",")
		feature_vec_length -= 1
	elif im.startswith("response"):
		flag = 1
		feature_vec_length -= 1
		
#print "train labels ", train_labels
#print "train_data", train_data
#print "data" , data
#print "meta" , meta
#print "lengths ", len(data), len(meta)

#-------------------------------------------------------Read the test data---------------------------------------------------------------
test_data = []
data, meta = arff.loadarff(test_file)
for itemd in data:
	l = []
	for i in range(len(itemd)):
		l.append(str(itemd[i]))
	test_data.append(l)
#print test_data
#print feature_vec_length

print"k value :", k

#------------------------------------------------------------Execute learning algo on test data -------------------------------------------
if flag == 1: #Regression task
	error_sum = 0.0
	for item in test_data:
		distances = []
		for i in range(len(train_data)):
			sum = 0.0
			for j in range(len(item)-1):
				sum += math.pow((float(item[j]) -  float(train_data[i][j])),2)
			distances.append([math.sqrt(sum), i])
		k_smallest = heapq.nsmallest(k, distances)

		y_sum = 0.0
		for itemq in k_smallest:
			y_sum += float(train_data[itemq[1]][feature_vec_length])
		y_avg = y_sum/k
		predicted_value = y_avg
		actual_value = float(item[feature_vec_length])
		error_sum += math.fabs(predicted_value - actual_value)
		print "Predicted value : "+str('%.6f' % round(predicted_value,6))+"\tActual value : "+str('%.6f' % round(actual_value,6))
	print "Mean absolute error : "+ '%.16f' % round(float(error_sum/len(test_data)),16)
	print "Total number of instances : "+str(len(test_data))

else : #Classification task
	correct = 0
	cd = {}
	
	for itemtest in test_data:
		distances = []
		for i in range(len(train_data)):
			sum = 0.0
			for j in range(len(itemtest)-1):
				sum += math.pow((float(itemtest[j]) -  float(train_data[i][j])),2)
			distances.append([math.sqrt(sum), i])
		k_smallest = heapq.nsmallest(k, distances)

		class_count = []
		for itemq in k_smallest:
			c = train_data[itemq[1]][feature_vec_length]
			classes = [item[0] for item in class_count]
			if c not in classes:
				class_count.append([c,1])
			else :
				p = classes.index(c)
				class_count[p][1] += 1
		
		counts = [item[1] for item in class_count]
		counts_max = [item for item in counts if item == max(counts)]
		if len(counts_max) == 1 :
			p = counts.index(max(counts))
			predicted_class = class_count[p][0]
			print "Predicted class : "+predicted_class+"\tActual class : "+itemtest[feature_vec_length]
		elif len(counts_max) > 1: 
			classes = [item[0] for item in class_count if item[1] == counts_max[0]]
			class_indices = [train_labels.index(item) for item in classes]
			class_indices.sort()
			predicted_class = train_labels[class_indices[0]]
			print "Predicted class : "+predicted_class+"\tActual class : "+itemtest[feature_vec_length]
		
		if predicted_class == itemtest[feature_vec_length]:
			correct += 1

	print "Number of correctly classified instances : "+str(correct)
	print "Total number of instances : "+str(len(test_data))
	print "Accuracy : "+ '%.16f' % round(float(correct)/len(test_data),16)


			
		
				
			
	
			
		
		
				

	
