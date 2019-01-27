<h1>Building a Neural Network from Scratch</h1>
<p>
  In this project I have built a neural network from scratch so that the people who are new to Neural Networks and Deep learning could
  understand how they work. Most of the times, no one will build Neural Networks from scratch. Often we use libraries like 
  <a href="https://pytorch.org/">Pytorch</a> and <a href="https://www.tensorflow.org/">TensorFlow</a> (ofcourse there are many others too)
  to build the neural networks. But building them from scratch will help in understanding the working behind these libraries.
</p>
<p>
  <img href="https://github.com/SurajChinna/Building-a-Neural-Network-from-Scratch-1/blob/master/assets/img1.png" />
  simple neural network with one layer
</p>
<p>
  We will build a simple neural network with just one layer. The dataset can be downloaded from 
  <a href=" http://www.ats.ucla.edu/stat/data/binary.csv">here</a>. It consists of three input features: GRE score, GPA, 
  and the rank(numbered 1 through 4). The goal here is to predict if a student will be admitted to a graduate program based 
  on his rank, gre and gpa scores. The first step of building is data cleaning.
</p>
<h2>Data Cleaning</h2>
<p>
  Firstly, the categorical data must be removed. In this dataset <b>rank</b> is the categorical variable. Learn more about categorical
  variables <a href="https://en.wikipedia.org/wiki/Categorical_variable">here</a>. Next, the data must be scaled such that
  they have mean as zero and standard deviation as one. Finally the data should be split into training and testing set so that
  we train on training on training data and test the model performance on testing data.
  The exact code with more decription can be found in the <i>neural net from scratch.ipynb</i> notebook
</p>
<h2>Model building</h2>
<p>
  The next step after cleaning the data is model building. First we initialise the weights with size same as number of features. Next,
  we take each row from the training data and get the output after multiplying with weights and passing into activation function.
  The activation function we are using here is <a href="https://en.wikipedia.org/wiki/Sigmoid_function">Sigmoid function</a>. Next we
  calculate the error which is actual output minus obtained output <b><i>error = actual_output - obtained_output</i></b>. The formula for error
  term is <b><i>error_term = error*sigmoid(obtained_output)</i></b>. Next we add the <b><i>error_term*input</i></b> to a variable <b><i>delta_of_w</i></b>. 
  The <b><i>delta_of_w</i></b> is summed for all the input values and finally we update the acutal weights. The formula to update the 
  actual weights is <b><i>original_weights += learn_rate*delta_of_w/n_records</i></b>, where learn_rate is hyper-parameter that controls how 
  much we are adjusting the weights of our network. We repeat this process for many iterations to reach to the optimal value of
  weights that predict the output accurately. See the notebook for more conceptual clarity and step by step descriptions.
</p>
<p>
  After the model has been trained for 4000 iterations(also called epochs),Finally the model is tested is testing data and the 
  accuracy obtained as 75%. That is really is great as we have used just one layer. The accuracy can be improved by adding more layers
  and tuning the hyperparameters.
</p>
