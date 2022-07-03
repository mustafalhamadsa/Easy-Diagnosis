
** The website is deployed online and can be accessed through the following link:

http://identify1.pythonanywhere.com	
 
** Requirements for Easy Diagnosis flask application:

— Python version installed: 3.X , X>5

— Install MySQL workbench, create schema and import the .sql file from the Data Import option on the right.

—  Then apply the following steps: 

1-  If Python version installed: 3.X , X>5 is not installed , install it.

2-  Installing virtaulenv

	for Mac OS follow the following link:

		https://gist.github.com/dineshviswanath/af72af0ae2031cd9949f
		
	for windows OS follow the following link:
		
		https://virtualenv.pypa.io/en/stable/userguide/	
	---------------------------------------------------------------------------

3-  Add the two ED_APP.py and requirements.txt files and the two folders: static and templets to you virtaulenv directory NOT insid the virtaulenv folder .// by coping and adding all the needed files 

##--important note: use pip3 instead of pip always if the python version was 3.X--##
-----------------------------------------------------------------------------------------

4-  Activate your virtaulenv


....\APP-DIRECTORY>env\Scripts\activate.bat
-----------------------------------------------------------------------------------------

5- install all the app requirements by running the following command FIRST TIME :\\NEED THIS COMMAND JUST ONE TIME  
		
	pip install -r requirements.txt
-- OR USE FOR MAC --
	pip3 install -r requirements.txt

6-  Set the environment variable FLASK_APP to ED_APP.py and MYSQL_DATABASE_PASSWORD to your database connection password.
	
	for Mac OS:
		
		export NEWVAR=SOMETHING

EX: export FLASK_APP=ED_APP.py  //no spaces

EX: export MYSQL_DATABASE_PASSWORD=root12345  //no spaces

	for windows OS:

 		set NEWVAR=SOMETHING

7-  Run the app from by running the following command: 
	
	flask run 

-- OR USE FOR MAC --

	python3 -m flask run

8-  Copy the URL from the terminal and run it in your browser.   // (Press CTRL+C to quit)

9-  Login with admin account and create other users ( admins and medical specialists)  or create a registered user account.

— The database is initialized with an admin account and 11 active diagnosis models for CKD, Diabetes Mellitus, CHD, RA, Schizophrenia, TC, Asthma, Alzheimer, Hypothyroidism, Breast Cancer and ADHD.  

— admin account credentials:
	username : admin
	password: admin123




