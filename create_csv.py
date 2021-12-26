import pandas as pd
import os
from sklearn.utils import shuffle
from utils import encode_labels

def main():
	dataset_path = os.path.join('dataset', 'raw-img')
	labels = encode_labels(dataset_path)

	train_df = pd.DataFrame(columns = ['image_path', 'label'])
	test_df = pd.DataFrame(columns = ['image_path', 'label'])
	val_df = pd.DataFrame(columns = ['image_path', 'label'])
	

	for dirname in os.listdir(dataset_path):
	    full_dirname = os.path.join(dataset_path, dirname)
	    sample_number = len(os.listdir(full_dirname))    
	    percent80 = int(0.8 * sample_number)
	    percent10 = int(0.1 * sample_number)	
	    end_train = percent80
	    end_val = percent80 + percent10
	    
	    for idx,file in enumerate(os.listdir(full_dirname)):
	        if idx < end_train:
	            train_df = train_df.append(
	                {
	                    "image_path": os.path.join(full_dirname, file),
	                    "label": labels[dirname] 
	                },
	            ignore_index = True
	            )
	        elif idx < end_val:
	            val_df = val_df.append(
	                {
	                    "image_path": os.path.join(full_dirname, file),
	                    "label": labels[dirname] 
	                },
	            ignore_index = True
	            )
	        else:
	            test_df = test_df.append(
	                {
	                    "image_path": os.path.join(full_dirname, file),
	                    "label": labels[dirname] 
	                },
	            ignore_index = True
	            )

	train_df = shuffle(train_df)
	test_df = shuffle(test_df)
	val_df = shuffle(val_df)

	train_df.reset_index(inplace=True, drop=True)
	test_df.reset_index(inplace=True, drop=True)
	val_df.reset_index(inplace=True, drop=True)

	train_df.to_csv(os.path.join('dataset',"train.csv"))
	test_df.to_csv(os.path.join('dataset',"test.csv"))
	val_df.to_csv(os.path.join('dataset',"val.csv"))

if __name__ == "__main__":
	main()