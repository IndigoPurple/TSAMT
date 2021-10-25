import numpy as np
import matplotlib.pyplot as plt

# data = np.loadtxt('optimization_pose.csv', dtype=np.double, delimiter=',')
data = np.load('test_pose_period.npy')

pick = 3
pose3 = data[:,3]
# print(pose3.size)
pose5 = data[:,5]
pose6 = data[:,6]
pose = data[:,pick]
x = np.arange(1,pose.size+1,1)
y = np.polyfit(x, pose, 30)
curve = np.poly1d(y)
yvals= curve(x)
mean = np.mean(pose) * np.ones(pose.size)
diff = pose-mean

plt.figure(figsize=(10, 5))

Y = np.fft.fft(diff)

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}

plt.xlabel('Frame index', font1)
plt.ylabel('$θ_{1,1}$', font1)
Pyy = Y * np.conj(Y) / pose.size
print(np.argwhere(Pyy>0.05))

plt.plot(x[:40], Pyy[:40],'o')
for i in range(int(x.size/2)):
    xx = [x[i],x[i]]
    yy = [0,Pyy[i]]
    plt.plot(xx,yy,'b')
plt.xlabel('Frequency', font1)
plt.ylabel('Amplitude',font1)
plt.show()