import matplotlib.pyplot as plt
import numpy as np
fr = open('acu2.txt')
fr = fr.readlines()
b1 = []
b2 = []
b3 = []
b4 = []

for i in range(len(fr)):
	b1.append(fr[i].split()[0])
	b2.append(fr[i].split()[1])
	b3.append(fr[i].split()[2])
	b4.append(fr[i].split()[3])

plt.figure(figsize=(16, 12), dpi=164, facecolor="white")
axes = plt.subplot(111)
axes.cla() # Clear all the information in the coordinate
# Assign the font of the picture
font = {'family' : 'serif', 'color'  : 'darkred', 'weight' : 'normal', 'size'   : 16}
ax = plt.gca()
plt.ylabel(u'Accuracy')
plt.xlabel(u'Training times/1000 times')
l1 = plt.plot(b1, b2, 'x-')
l2 = plt.plot(b1, b3, 'o-')
l3 = plt.plot(b1, b4, '^-')
ax.set_yticks(np.linspace(0.5, 0.7, 5))
ax.set_yticklabels(('50%', '55%', '60%', '65%', '70%'))
ax.legend((l1[0], l2[0], l3[0]), (u'Testing accuracy', u'Testing amplitude accuracy', u'Training accuracy'))
plt.savefig('Training accuracy.png')