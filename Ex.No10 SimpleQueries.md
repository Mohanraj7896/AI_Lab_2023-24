# Ex.No: 10  Logic Programming –  Simple queries from facts and rules
### DATE: 06/09/2025                                                                           
### REGISTER NUMBER : 212223060166
### AIM: 
To write a prolog program to find the answer of query. 
###  Algorithm:
 Step 1: Start the program <br> 
 Step 2: Convert the sentence into First order Logic  <br> 
 Step 3:  Convert the sentence into Horn clause form  <br> 
 Step 4: Add rules and predicates in a program   <br> 
 Step 5:  Pass the query to program. <br> 
 Step 6: Prolog interpreter shows the output and return answer. <br> 
 Step 8:  Stop the program.
### Task 1:
Construct the FOL representation for the following sentences <br> 
1.	John likes all kinds of food.  <br> 
2.	Apples are food.  <br> 
3.	Chicken is a food.  <br> 
4.	Sue eats everything Bill eats. <br> 
5.	 Bill eats peanuts  <br> 
   Convert into clause form and Prove that John like Apple by using Prolog. <br> 
### Program:
```
 likes(john,X):-food(X).
 food(apple).
 food(chicken).
 eats(sure,X):-eats(bill,X).
 eats(bill,peanuts).
```
### Output:<img width="1436" height="681" alt="Screenshot 2025-09-12 201647" src="https://github.com/user-attachments/assets/bfac2974-9d3d-4c32-b223-81ab2af62a1c" />


### Task 2:
Consider the following facts and represent them in predicate form: <br>              
1.	Steve likes easy courses. <br> 
2.	Science courses are hard. <br> 
3. All the courses in Have fun department are easy <br> 
4. BK301 is Have fun department course.<br> 
Convert the facts in predicate form to clauses and then prove by resolution: “Steve likes BK301 course”<br> 

### Program:
```
 likes(steve, X) :
easy_course(X).
 hard_course(science).
 easy_course(X) :
in_department(X, have_fun).
 in_department(bk301, have_fun).
```
### Output:<img width="1430" height="698" alt="Screenshot 2025-09-12 201747" src="https://github.com/user-attachments/assets/4b742035-3c0f-4b7a-87dd-da4165e9deca" />


### Task 3:
Consider the statement <br> 
“This is a crime for an American to sell weapons to hostile nations. The Nano , enemy of America has some missiles and its missiles were sold it by Colonal West who is an American” <br> 
Convert to Clause form and prove west is criminal by using Prolog.<br> 
### Program:
```
criminal(X):-
american(X),
 weapon(Y),
 hostile(Z),
 sells(X,Y,Z).
 weapon(Y):-
missile(Y).
 hostile(Z):-
enemy(Z,america).
 sells(west,Y,nano):-
missile(Y),
 owns(nano,Y).
 missile(m).
 owns(nano,m).
 enemy(nano,america).
 american(west).
```
### Output:<img width="1420" height="714" alt="Screenshot 2025-09-12 201946" src="https://github.com/user-attachments/assets/b503df81-9b5b-4161-ae26-b824fb1156bc" />


### Result:
Thus the prolog programs were executed successfully and the answer of query was found.
