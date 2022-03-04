# nn_uproot_training

Demo memory leak when using uproot - I might just be doing something daft but I can't see whats wrong with the code

Small dataset (2.34 GB) to test on available here https://drive.google.com/drive/folders/15hqdIR4rxr7Ahb3MQhBRDmgmwy0UoVdV?usp=sharing

Place folder (called test_data) inside repo directory - contains 9 files (one for each file type)

To run:
python3 leak_demo.py

This creates a class which handles 9 uproot.iterate instaces (one for each file type). Code just loops through each iterator and fetches next array, no additional processing is applied. After cycling through all events iterators are remade and the process is repeated


Should see memory consumption gradually increase (see mem_usage.png)
