tennis:
	python3 best_feature.py -r /Users/julietteortiz/Desktop/Data_Science/HW6/data/tennis_train.arff > test.txt

d:
	python3 best_feature.py -r /Users/julietteortiz/Desktop/Data_Science/HW6/data/tennis_train.arff > data.txt

movie:
	python3 best_feature.py -r data/movies_train.arff > movie.txt

tests:
	python3 best_feature.py -r data/tennis_train.arff 
	python3 best_feature.py -r data/movies_train.arff
	python3 best_feature.py -r data/heart_train.arff
	python3 best_feature.py -r data/diabetes_train.arff