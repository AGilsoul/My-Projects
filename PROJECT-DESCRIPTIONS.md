# CS-Portfolio
A portfolio of all my programming projects that I feel showcase my skills as a programmer

Project Descriptions:

Python:

  K-Means Clustering: In this program I coded my own K-Means Clustering machine learning algorithm to classify three different types of iris flowers based on the length of their sepal and the width of their sepal. The program moves three different points, or centroids around the graph based on the data provided to locate them in the center of the three different flower clusters, and colors the flower data points based on the nearest centroid.

  BreastCancerClassifier: In this program I used the K-Nearest Neighbors machine learning algorithm from the sklearn library to determine whether patients have breast cancer based on a myriad of factors that were recorded for each patient. The algorithm is then tested, and a graph is shown displaying its accuracy on a validation set of patients with a different number of "K-Nearest Neighbors". It can be seen that the accuracy of the algorithm reaches about 96.5% when K is set to 10. The algorithm can be tested on a different arrangment of the data by changing the random state in the code.


C++:

  FRC2020: This folder contains the code written for my high school's First Robotics Competition (FRC) robot. It was mostly written by a team of five people including me. The majority of the important code is contained in the file named "bruh.cpp". The functionalities I programmed into the robot mainly revolved around its ball shooting capabilities. During the competition, our robot had to launch balls at a target. Included in this was vision processing, which I coded in to the robot to allow it to aim itself at a target using a "Limelight" camera. This allowed for increased accuracy over simply having the robot driver aim it themself. The methods which I coded are belonging to the Robot class and are named teleShoot, autoShoot, startAutoTarget, startTeleTarget, findTarget, and AutonomousPeriodic. In addition to this I also helped with some of the code in the TeleopPeriodic method. The methods that I wrote all use each other to aim the robot, and implement a Proportion Integral Derivative (PID) controller method to aim the robot without overadjusting.

  2420_FinalProject: In this project I coded a binary tree object, and then created a method for balancing the tree to optimize average query time for each node. The user first specifies the number of nodes they want in the tree. The program then creates a tree with all numbers up to that number, inserting them in a random order. The program then balances the tree, and displays a percentage difference in the average node distance between the balanced and unbalanced trees.
  
  1410(FinalProject): In this project I made a command line program that utilized a CSV file containing all production cars sold in the United States in 2020 to find the best car for a user to fit their needs based on certain preferances such as build, mileage, drivetrain, and so on. It implements a selection sort to sort cars based on horsepower, mileage, and price. The user is greated by a menu from which they can filter the list, sort the list, and do a google search for the cars currently in the list.
  
  Hashtable: This is a simple hashtable program that makes use of smart pointers to create a hashtable that sorts a specified number of random strings based on the first character in the string, which leads to improved search time over a basic array. The program creates the random strings and then displays the hashtable in the console.
  
  11.1_NoBrainer: This is where I originally coded my graph object and graph traversal methods. Like the above program, running the program will only show the test results, and the Graph.h file shows most of the code worth looking at.
  
  
Java:

  DataVisualization: In this project I created a binary search tree using Java rather than C++. I also coded a simple GUI that allows the user to select certain nodes on the tree, delete certain nodes, and insert new nodes.
