import numpy as np
from matplotlib import pyplot as plt 

x=np.arange(0,100)
print(x)
y=np.random.randint(10,100,100)
print(y)

threta0=10
threta1=0
learningRate=0.1

plt.plot(x,y,"o")

for i in x:
    total=0
    total1=0
    m=x.size
    for num in x:
        #预测值
        h=threta0+threta1*x[num]
        #实际值
        y_actual=y[num]
        total=total+(h-y_actual)
        total1=total1+(h-y_actual)*x[num]
    temp0=threta0-learningRate/m*total
    temp1=threta1-learningRate/m*total1
    if temp0==threta0:
        print("找到最优解threta0=%d,threta1=%d"%(temp0,temp1))
        break;
    threta0=temp0
    threta1=temp1
    print("i=%d,threta0=%d,threta1=%d"%(i,threta0,threta1))

h=threta0+threta1*x
plt.plot(x,h)

plt.show()

