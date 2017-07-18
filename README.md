dmc-2017
==============================

This project, participant of Data Mining Cup 2017, analyzes the set in order to recognize existing relationships of dynamic prices strategies and creates the model that can predict revenues given certain training set.
Due to some problems with submission this project was not represented in ranking. Result error points equal 9734.45107402 which is 18.5% worse than 1st place result and matches 23rd place.

Raw data can be downloaded from http://www.data-mining-cup.de/en/dmc-competition/task/


Project Organization
------------
	├── LICENSE
	├── README.md				<- The top-level README for developers using this project.
	├── data
	│   ├── Uni_Polytechnic_Lviv_1.csv	<- Submission.
	│   │
	│   ├── external			<- Data from third party sources.
	│   │
	│   ├── interim				<- Intermediate data that has been transformed.
	│   │   └── items_v1.csv		<- Transformed items.
	│   │	
	│   ├── processed			<- The final, canonical data sets for modeling.
	│   │
	│   └── raw				<- The original, immutable data dump.
	│       ├── class.csv			<- Information for the classification time period.
	│       ├── items.csv			<- Attributes of all the products that do not change with time.
	│       ├── realclass.csv       	<- Answers
	│       └── train.csv			<- Information for the learning time period.
	│
	├── models				<- Trained and serialized models, model predictions, or model summary
	│   ├── model_lasso.pkl			<- Saved Lasso model.
	│   └── model_ridge.pkl			<- Saved Riege model.
	│
	├── notebooks				<- Jupyter notebooks.
	│   └── all.ipynb         		<- Jupyter notebook with all work.
	│
	├── reports				<- Generated analysis as HTML, PDF, LaTeX, etc.
	│  
	└── src					<- Source code for use in this project.
	    ├── evaluate.py			<- Source code for evaluation of error points.
	    ├── final.py			<- Source code for main work (starting with preprocessing ending with generation of submission.
	    └── func.py				<- Source code for function used in final.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
