import pandas as pd
import numpy as np


df = pd.read_json("D:/ARC/arc-data/arc-agi_training_challenges.json")

for task in df:
    # This variable holds only the train row of the dataset
    train_ex = df[task][1]

    # This line prints the amount of examples for each task
    print(len(train_ex))
    for example in train_ex:
        # This prints all the examples of the dataset
        input, output = example["input"], example["output"]
        print(input)
        print(output)