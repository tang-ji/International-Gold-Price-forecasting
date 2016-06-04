import matplotlib.pyplot as plt
import numpy as np
fr = open('acu.txt')
fr = fr.readlines()
b1 = []
b2 = []

for i in range(len(fr)):
	b1.append(fr[i].split()[0])
	b2.append(fr[i].split()[1])

plt.figure(figsize=(16, 12), dpi=244, facecolor="white")
axes = plt.subplot(111)
axes.cla() # Clear all the information in the coordinate
# Assign the font of the picture
font = {'family' : 'serif', 'color'  : 'darkred', 'weight' : 'normal', 'size'   : 25}
ax = plt.gca()
plt.ylabel(u'Accuracy')
plt.xlabel(u'Training times/1000 times')
l1 = plt.plot(b1[1000:], 'y')
l2 = plt.plot(b2[1000:], 'r')
ax.set_yticks(np.linspace(0.12, 0.20, 5))
ax.set_xticklabels((0, 20, 40, 60, 80, 100, 120, 140, 160))
ax.legend((l1[0], l2[0]), (u'Training variance', u'Testing variance'))
plt.savefig('Training variance.png')
