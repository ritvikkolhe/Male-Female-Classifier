##### EXERCISE 1 :

#import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import pandas as pd

male = pd.read_csv('male_train_data.csv')
female = pd.read_csv('female_train_data.csv')

print(male.male_bmi[3])
print(female.female_bmi[7])
print(male.male_stature_mm[1])
print(female.female_stature_mm[9])

#### EXERCISE 2: 


female_bmi = female['female_bmi'].to_numpy()
female_stature_mm = female['female_stature_mm'].to_numpy()
A_female = np.array([female_bmi,female_stature_mm]).transpose()
A_female_bias = np.append(A_female,np.ones([len(female_bmi),1]),1)

male_bmi = male['male_bmi'].to_numpy()
male_stature_mm = male['male_stature_mm'].to_numpy()
A_male = np.array([male_bmi,male_stature_mm]).transpose()
A_male_bias = np.append(A_male,np.ones([len(male_bmi),1]),1)

b_male = np.ones(len(male_bmi))
b_female = -1 * (np.ones(len(female_bmi)))

A = np.concatenate((A_male_bias,A_female_bias),axis=0)
b = np.concatenate((b_male,b_female),axis=0)




### optimum weight vector

A_trans = A.transpose()
A_trans_A = np.matmul(A_trans, A)
inv_A_trans_A = inv(A_trans_A)
optimal_weight_vec = np.matmul(np.matmul(inv_A_trans_A,A_trans),b)

print(optimal_weight_vec)

### part 3

'''x = cp.Variable(3)
objective = cp.Minimize(cp.sum_squares(A*x - b))
constraints = []
prob = cp.Problem(objective,constraints)
result = prob.solve()
w1,w2,w0 = x.value'''


####EXERCISE 3 :
## PLOT FOR FEMALE CLASSES
plt.scatter(female_bmi,female_stature_mm)

## PLOT OF MALE CLASSES
plt.scatter(male_bmi,male_stature_mm)

plt.xlabel('BMI')
plt.ylabel('Stature')

w1,w2,w0 = optimal_weight_vec
x1 =np.linspace(min(female_bmi) ,max(female_bmi), 200)
slope_m = - (w1/w2)
y_intercept = - (w0/w2)
x2 = slope_m * x1 + y_intercept

plt.plot(x1,x2)


###EXERCISE 3 PART B 

female_test = pd.read_csv('female_test_data.csv')
female_bmi = female_test['female_bmi'].to_numpy()
female_stature_mm = female_test['female_stature_mm'].to_numpy()
A_female_test = np.array([female_bmi,female_stature_mm]).transpose()
A_female_bias_test = np.append(A_female_test,np.ones([len(female_bmi),1]),1)
b_female_test = -1 * (np.ones(len(female_bmi)))

male_test = pd.read_csv('male_test_data.csv')
male_bmi = male_test['male_bmi'].to_numpy()
male_stature_mm = male_test['male_stature_mm'].to_numpy()
A_male_test = np.array([male_bmi,male_stature_mm]).transpose()
A_male_bias_test = np.append(A_male_test,np.ones([len(male_bmi),1]),1)
b_male_test = np.ones(len(male_bmi))

A_test = np.concatenate((A_male_bias_test,A_female_bias_test),axis=0)
b_test = np.concatenate((b_male_test,b_female_test),axis=0)

sm = 0
for i in A_male_test:
    pred_male = i[1] - (-w1/w2)*i[0] - (-w0/w2)
    if pred_male>0:
        sm+=1

sf = 0
for j in A_female_test:
    pred_female = j[1] - (-w1/w2)*j[0] - (-w0/w2)
    if pred_female<0:
        sf+=1
    
success_rate = (sm+sf)/1002
print(success_rate)

### EXERCISE 4
A = A/100
answer=[]
lambda_list = np.arange(0.1, 10, 0.1)
for i in lambda_list:
    x = cp.Variable(3)
    objective = cp.Minimize(cp.sum_squares(A*x - b) + cp.sum_squares(i*x))
    constraints = []
    prob = cp.Problem(objective,constraints)
    result = prob.solve()
    answer.append(x.value)
    
x1 =np.linspace(min(female_bmi) ,max(female_bmi), 200)
## PLOT FOR FEMALE CLASSES
plt.scatter(female_bmi,female_stature_mm)

## PLOT OF MALE CLASSES
plt.scatter(male_bmi,male_stature_mm)

for j in answer:
    x2 = (- (j[0] / j[1]) * x1) - (j[2]/j[1])

    
    plt.plot(x1,x2)

    

plt.xlabel('BMI')
plt.ylabel('Stature')


### PART 3
x = cp.Variable(3)
objective = cp.Minimize(cp.sum_squares(A*x - b) + cp.sum_squares(0.1*x))
constraints = []
prob = cp.Problem(objective,constraints)
result = prob.solve()
a = x.value

theta_alpha = []
for i in range(-50,51):
    x = cp.Variable(3)
    objective = cp.Minimize(cp.sum_squares(A*x - b))
    temp=cp.sum_squares(x)
    constraints = [temp<= a+i*2]
    prob = cp.Problem(objective,constraints)
    result = prob.solve()
    theta_alpha.append(x.value)

theta_epsilon=[]
for i in range(0,100):
    x = cp.Variable(3)
    objective = cp.Minimize(cp.sum_squares(x))
    temp=cp.sum_squares(A*x-b)
    constraints = [temp<= a+i*2]
    prob = cp.Problem(objective,constraints)
    result = prob.solve()
    theta_epsilon.append(x.value)









