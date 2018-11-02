#!/bin/bash

counter=1
while [ $counter -le 3 ]
do
python train_deepscores.py
((counter++))
done
