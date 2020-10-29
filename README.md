



To create data : 
python -m deep_miml.create_dataset —num_bags_train 2000 —num_bags_test 100 —num_bags_val 100


1. ResNext 50 AVG
python -m deep_miml.train --model_name resnext50 --model_type avg --use_pretrained True --data_file_path data/miml_data.pt --batch_size 32

Results:
Using cuda training time 90 seconds:  {'precision_at': {1: 0.889, 2: 0.738, 3: 0.618, 4: 0.518, 5: 0.441, 6: 0.388}, 'recall_at': {1: 0.448, 2: 0.673, 3: 0.801, 4: 0.87, 5: 0.913, 6: 0.949}}

Using cpu training time 284.3655 seconds:  {'precision_at': {1: 0.89, 2: 0.74, 3: 0.612, 4: 0.516, 5: 0.441, 6: 0.385}, 'recall_at': {1: 0.445, 2: 0.672, 3: 0.792, 4: 0.865, 5: 0.911, 6: 0.942}}


2. Resnext50 Attention
python -m deep_miml.train --model_name resnext50 --model_type attention --use_pretrained True --data_file_path data/miml_data.pt --batch_size 32

Results:
Using cpu training time 402 seconds:  {'precision_at': {1: 0.86, 2: 0.717, 3: 0.598, 4: 0.506, 5: 0.435, 6: 0.376}, 'recall_at': {1: 0.422, 2: 0.641, 3: 0.768, 4: 0.846, 5: 0.898, 6: 0.924}}


3. ResNet18 AVG
python -m deep_miml.train --model_name resnet18 --model_type avg --use_pretrained True --data_file_path data/miml_data.pt --batch_size 32

Results:
Using cuda training time 37 seconds:  {'precision_at': {1: 0.876, 2: 0.735, 3: 0.615, 4: 0.516, 5: 0.442, 6: 0.383}, 'recall_at': {1: 0.434, 2: 0.662, 3: 0.791, 4: 0.865, 5: 0.913, 6: 0.941}}

Using cpu training time 591 seconds: {'precision_at': {1: 0.891, 2: 0.74, 3: 0.618, 4: 0.524, 5: 0.445, 6: 0.386}, 'recall_at': {1: 0.444, 2: 0.67, 3: 0.793, 4: 0.874, 5: 0.917, 6: 0.943}}


4. ResNet18 Attention
python -m deep_miml.train --model_name resnet18 --model_type attention --use_pretrained True --data_file_path data/miml_data.pt --batch_size 32

Results:
Using cpu training time 427 seconds:{'precision_at': {1: 0.873, 2: 0.73, 3: 0.617, 4: 0.513, 5: 0.44, 6: 0.382}, 'recall_at': {1: 0.432, 2: 0.656, 3: 0.792, 4: 0.859, 5: 0.907, 6: 0.935}}