### cluster_color_kmeans

### Description

- Load image, cluster the color then save the result in `./imgs`
- *(New)* find the optimal color number of image

![result1](./imgs/parrot_clustered.jpg)

![result2](./imgs/flower_clustered.jpg)

![result3](./imgs/parrot_optimal_plot.jpg)

### How to run?

- Show help: `python main.py --help`

![run1](./imgs/Capture.PNG)

- Run clustering: `python main.py -i ./imgs/parrot.jpg`

![run2](./imgs/Capture2.PNG)

- Run finding optimal: `python main.py -i ./imgs/parrot.jpg -o`

![run3](./imgs/Capture3.PNG)

### Terminal command
- Generate SSH: `ssh-geygen` 
- Cat pulish SSH: `cat <your_ssh_dir>/id_rsa.pub`
- Setting your commit email: `$ git config --global user.email "email@example.com"`
- Setting your commitname: `$ git config --global user.name "yourname"`
- Install pip on Ubutu: `sudo apt-get install python3-pip`
- Create virtual enviroment: `virtualenv env`
- Activate virtual enviroment on **Window**: `cd env/Scripts` + `activate`
- Activate  virtual enviroment on **Linux**: `source env/bin/activate`
- Dectivate virtual enviroment: `deactivate`
- Export required libraries: `pip freeze > requirments.txt`
- Install all required libraries: `pip install -r requirements.txt`

### References

[k-means-clustering](https://www.machinelearningplus.com/predictive-modeling/k-means-clustering/)

[k-means-clustering-from-scratch](https://www.askpython.com/python/examples/k-means-clustering-from-scratch)

[build-k-means-from-scratch-in-python](https://medium.com/@rishit.dagli/build-k-means-from-scratch-in-python-e46bf68aa875)

[k-means-for-beginners-how-to-build-from-scratch-in-python](https://analyticsarora.com/k-means-for-beginners-how-to-build-from-scratch-in-python/)
