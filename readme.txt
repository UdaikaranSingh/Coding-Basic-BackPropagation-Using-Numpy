Names:
1. Udaikaran Singh
2. Wesley Kwan

Class: CSE 190
======================================
Which File to run:
1.mainFile.py

This files runs all the other files.
======================================
Source Files:
1.data.zip
2.gradientchecker.py
3.mainFile.py
4.checker.py
5.neuralnet.py 
6.ProblemC.py
7.ProblemD.py
8.ProblemE.py
9.ProblemF.py

======================================
What Each File Does:
1.gradientchecker.py
	This corresponds with part B of the write-up.
	This compares gradient approximation with theoretical gradient.
2.ProblemC.py
	This corresponds with part C of the write-up.
	In this, the model trains on the complete dataset for 100 epochs.
	The console prints accuracy and number of epochs needed.
	There is 1 plot saved from this under the name: "partC.png"
3.ProblemD.py
	This corresponds with part D of the write-up.
	This experiments with regularization.
	This saves 2 plots under the names: 
		"partD_0.001.png"
		"partD_0.0001.png"
4.ProblemE.py
	This corresponds with part E of the write-up.
	This experiments with activation functions.
	This saves 3 plots under the name:
		"partE_ReLU.png"
		"partE_sigmoid.png"
		"partE_tanh.png"
5.ProblemF.py
	This corresponds with part F of the write-up.
	This experiments with different network topologies.
	This saves 3 plots under the name:
		"partF_[784, 25, 10].png"
		"partF_[784, 47, 47, 10].png"
		"partF_[784, 100, 10].png"

======================================
Helper Files:
1.checker.py
2.neuralnet.py
3.data.zip