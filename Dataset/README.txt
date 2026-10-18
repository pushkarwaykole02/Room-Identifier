8-category household room categorization dataset
------------------------------------------------

This is the dataset that was used in the following paper:

Peter Uršič, Rok Mandeljc, Aleš Leonardis, and Matej Kristan.
Part-Based Room Categorization for Household Service Robots
IEEE International Conference on Robotics and Automation (ICRA), 2016


This dataset is an 8-category subset of the well-known MIT Indoor Scene
Recognition dataset from the following paper:
A. Quattoni and A.Torralba. Recognizing Indoor Scenes. IEEE Conference on
Computer Vision and Pattern Recognition (CVPR), 2009.


The following 8 categories are used:
- bathroom
- bedroom
- children_room
- closet
- corridor
- dining_room
- kitchen
- livingroom


In this subset, we include only the images from the above categories
that are included in the author-provided training/test split (i.e.,
the ones that are listed in TrainImages.txt and TestImages.txt from
http://web.mit.edu/torralba/www/indoor.html).


In addition, we revised the dataset labels, hence some images have been
renamed or removed, as outlined below:


Changes made to the images from training set:
---------------------------------------------
bathroom/dsc3101.jpg -> REMOVED (duplicate of livingroom/dsc3101.jpg from test set)
bathroom/indoor_0323.jpg -> REMOVED (duplicate of bedroom/indoor_0323.jpg from training set)
bathroom/indoor_0332.jpg -> REMOVED (duplicate of bedroom/indoor_0332.jpg from test set)
bedroom/child4.jpg -> children_room/child4.jpg
bedroom/dsc04183.jpg -> REMOVED (duplicate of kitchen/dsc04183.jpg from training set)
bedroom/hunter_room.jpg -> children_room/hunter_room.jpg
bedroom/indoor_0057.jpg -> livingroom/indoor_0057.jpg
bedroom/indoor_0113.jpg -> dining_room/indoor_0113.jpg
bedroom/indoor_0267.jpg -> livingroom/indoor_0267.jpg
bedroom/indoor_0289.jpg -> livingroom/indoor_0289.jpg
bedroom/indoor_0330.jpg -> livingroom/indoor_0330.jpg
bedroom/indoor_0338.jpg -> livingroom/indoor_0338.jpg
bedroom/indoor_0557.jpg -> children_room/indoor_0557.jpg
bedroom/madison_room.jpg ->children_room/madison_room.jpg
bedroom/strongcpl_t.jpg -> children_room/strongcpl_t.jpg
children_room/AT_01_3B_5486_10_l.jpg -> bedroom/AT_01_3B_5486_10_l.jpg
dining_room/easyst034.jpg -> bedroom/easyst034.jpg
dining_room/easyst048.jpg -> livingroom/easyst048.jpg
dining_room/interior005.jpg -> livingroom/interior005.jpg
kitchen/dining047.jpg -> dining_room/dining047.jpg
kitchen/indoor_0281.jpg -> livingroom/indoor_0281.jpg

Changes made to the test set:
-----------------------------
bathroom/img_1115.jpg -> bedroom/img_1115.jpg
bedroom/indoor_0418.jpg -> livingroom/indoor_0418.jpg
dining_room/easyst036.jpg -> livingroom/easyst036.jpg
dining_room/easyst039.jpg -> livingroom/easyst039.jpg
dining_room/easyst042.jpg -> livingroom/easyst042.jpg
dining_room/easyst047.jpg -> livingroom/easyst047.jpg
dining_room/homeoff002.jpg -> livingroom/homeoff002.jpg
