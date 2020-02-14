# Portfolio

##### Table of Contents  
1. [ Airbnb data project. ](#Airbnb)

   1.1. [ Libraries needed. ](#Airbnb_lib)
   
   1.2. [ Files needed. ](#Airbnb_files)
   
   1.3. [ Acknowledgments. ](#Airbnb_ack)
   
2. [ MNIST data project. ](#mnist)

   2.1. [ Libraries needed. ](#mnist_lib)
   
   2.2. [ Files needed. ](#mnist_files)
   
   2.3. [ Acknowledgments. ](#mnist_ack)

3. [ Audiobook business case project. ](#audio)

   3.1. [ Libraries needed. ](#audio_lib)
   
   3.2. [ Files needed. ](#audio_files)
   
   3.3. [ Acknowledgments. ](#audio_ack)
   
4. [ Web application. ](#webapp)

   4.1. [ Libraries needed. ](#webapp_lib)
   
   4.2. [ Files needed. ](#webapp_files)
   
   4.3. [ Acknowledgments. ](#webapp_ack)

Right now there are four projects in this portfolio and they are described below.

<a name="Airbnb"></a>
## Airbnb data project:

The aim of this project is to analyse the data of Seattle's Airbnb listings obtained from the Udacity Nanodegree in Data Science. 

<a name="Airbnb_lib"></a>
### Libraries needed

The following libraries are used in this project:
1) numpy
2) pandas
3) matplotlib
4) sklearn
5) statsmodels
6) seaborn
7) re
8) os
9) imageio
10) tensorflow

<a name="Airbnb_files"></a>
### Files needed

There are several files in this project, some generated by the notebook (Airbnb_full_project.ipynb) and some used as an input for the notebook. The files used as an input for the notebook are:

1) Seattle_map2.png, to generate some images relating the map to the airbnb listings
2) Seattle_calendar.csv: data of the last ~2 millions event of bookings.
3) Seattle_listings.csv: data with the general features of all the listings in seattle up to 2017. There is a little bit less than 4000 listings.

<a name="Airbnb_ack"></a>
### Acknowledgements

I would like to express my gratitude to Simone Centellegher who provided on his github a beautiful way to plot bar graphs, which I usually find rather dull. link accessable on Jan/27/2020: https://scentellegher.github.io/visualization/2018/10/10/beautiful-bar-plots-matplotlib.html. Moreover, the file Seattle_map2.png is a printscreen of the Seattle city taken on google maps, which I have altered the quality and colors to be able to decrease file sizes.

<a name="mnist"></a>
## MNIST data project:

This as small and concise project on the tensorflow dataset of handwritten digits. In this project I aim to concisely model the dataset in order to predict handwritten digits. The dataset has basically no need for preprocessing, which is perfect for the goal of this project. I used a part of a course I've done in Udemy (https://www.udemy.com/course/the-data-science-course-complete-data-science-bootcamp/) as the basis for this project. By the end of the project I show that I manage to have an accuracy of about 98% on predicting new handwritten digits.

<a name="mnist_lib"></a>
### Libraries needed

The following libraries are used in this project:
1) numpy
2) tensorflow
3) tensorflow_datasets

<a name="mnist_files"></a>
### Files needed

There is only one file in this project and it is Complete_MNIST_dataset_done_by_me.ipynb.

<a name="mnist_ack"></a>
### Acknowledgements

I would like to express my gratitude to 365 Careers Team for the very good course.

<a name="audio"></a>
## Audiobook business case project:

This is another concise project on using Tensorflow in order to do predictions. In this case, the aim of the project is to predict the return of clients to buy audiobooks, in order to perform marketing recommendation. I used a part of a course I've done in Udemy (https://www.udemy.com/course/the-data-science-course-complete-data-science-bootcamp/) as the basis for this project. By the end of the project I show that I manage to have an accuracy of about 91% on predicting the return of the customers.

<a name="audio_lib"></a>
### Libraries needed

The following libraries are used in this project:
1) numpy
2) tensorflow
3) sklearn

<a name="audio_lib"></a>
### Files needed

There are several files in this project, some generated by the notebook Preprocessing_the_data.ipynb and some used as an input for the notebook. The files used as an input for the notebook are:

1) Audiobooks_data.csv presents the full dataset used for the whole analysis.
2) Audiobooks_data_test.npz, Audiobooks_data_train.npz and Audiobooks_data_validation are generated by the notebook.npz Preprocessing_the_data.ipynb.

In order to completely run the model, first is necessary to run the notebook Preprocessing_the_data.ipynb, followed by the notebook Predicting_client_return.ipynb.

<a name="audio_ack"></a>
### Acknowledgements

I would like to express my gratitude to 365 Careers Team for the very good course.

<a name="webapp"></a>
## Web application

This is a web application where I used the World Bank API (https://data.worldbank.org/) to generate a dashboard to do a brief analysis of the advanced education population of the top 10 world economies. Link to the web application: https://adv-educ-pop.herokuapp.com/.

<a name="webapp_lib"></a>
### Libraries needed

The following libraries are used in this project:
1) flask
2) numpy
3) pandas
4) json
5) plotly
6) collections
7) request

<a name="webapp_files"></a>
### Files needed

There are several different files needed in order to understand the web application and to add them all here would be pointless.

<a name="webapp_ack"></a>
### Acknowledgements

I would like to express my gratitude to Udacity Team for the very good course.
