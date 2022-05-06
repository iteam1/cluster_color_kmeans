# Description

Load image, cluster the color then save the result in `./imgs`

![result1](./imgs/parrot_clustered.jpg)

![result2](./imgs/flower_clustered.jpg)

# How to run?

- Show help: `python main.py --help`

![issue1](./imgs/Capture.PNG)

- Run: `python main.py -i ./imgs/parrot.jpg`

![issue2](./imgs/Capture2.PNG)

# Terminal command

- Create virtual enviroment: `virtualenv env`
- Activate virtual enviroment on **Window**: `cd env/Scripts` + `activate`
- Activate  virtual enviroment on **Linux**: `source env/bin/activate`
- Dectivate virtual enviroment: `deactivate`
- Export required libraries: `pip freeze > requirments.txt`
- Install all required libraries: `pip install -r requirements.txt`

# References

https://www.machinelearningplus.com/predictive-modeling/k-means-clustering/

https://www.askpython.com/python/examples/k-means-clustering-from-scratch

https://medium.com/@rishit.dagli/build-k-means-from-scratch-in-python-e46bf68aa875

https://analyticsarora.com/k-means-for-beginners-how-to-build-from-scratch-in-python/