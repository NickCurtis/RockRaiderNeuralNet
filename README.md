# RockRaiderNeuralNet

The Rock Raider neural net is a ROS (http://www.ros.org/) compatible convolutional neural net that can be trained to look for any type of image. It uses the lasagne (https://github.com/Lasagne/Lasagne) library as its neural network base. It reads images using OpenCV (http://opencv.org/).

# Usage

To build a ROS workspace follow this tutorial: http://wiki.ros.org/catkin/Tutorials/create_a_workspace \
Then build the ROS module using this tutorial: http://wiki.ros.org/catkin/Tutorials/using_a_workspace \
Then with ROS, opencv, and lasagne installed, open up a terminal and type "rosrun vision_neural_net ReadImage.py"
