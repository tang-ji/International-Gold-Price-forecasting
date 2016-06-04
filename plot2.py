import matplotlib.pyplot as plt
fr = open('accuracy2.txt')
fr = fr.readlines()
b1 = []
b2 = []
for i in range(len(fr)):
	b1.append(fr[i].split()[0])
	b2.append(fr[i].split()[1])
plt.plot(b1)
plt.plot(b2)
plt.show()
