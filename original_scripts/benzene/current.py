import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp2d
import multiprocessing as mp
plt.switch_backend('agg')




spacing = 0.3
dimensions = (227, 231, 201)
boxSize = (spacing*(dimensions[0]-1)//2,spacing*(dimensions[1]-1)//2,spacing*(dimensions[2]-1)//2)

xArray = np.linspace(-boxSize[0],boxSize[0],dimensions[0])
yArray = np.linspace(-boxSize[1],boxSize[1],dimensions[1])
zArray = np.linspace(-boxSize[2],boxSize[2],dimensions[2])

def getCartesian(indices):
	out = [0,0,0]
	for i in range(3):
		out[i] = (indices[i]-(dimensions[i]-1)//2)*spacing
	return (out[0],out[1],out[2])



direcs = ["output_iter/td.0001000/","output_iter/td.0002000/","output_iter/td.0003000/"]


def plotJphi(directory):
	filename  = "current-x.dx"
	curX = pd.read_csv(directory+filename,nrows=np.prod(dimensions),skiprows=7,header=None,names=["curX"])
	curX = np.array(curX.values)[:,0]
	curX = curX.reshape(dimensions)

	current2D = np.zeros((dimensions[0],dimensions[1]))
	current = 0.0
	for index in np.ndindex(dimensions):
		cart = getCartesian(index)
		rho = np.sqrt(cart[0]**2 + cart[1]**2)
		if rho != 0:
			flux = curX[index]*cart[1]/rho
		else:
			flux =0
		current2D[index[0],index[1]] += flux*spacing
		current += flux*spacing**3

	filename = "current-y.dx"
	curX = pd.read_csv(directory+filename,nrows=np.prod(dimensions),skiprows=7,header=None,names=["curX"])
	curX = np.array(curX.values)[:,0]
	curX = curX.reshape(dimensions)

	for index in np.ndindex(dimensions):
		cart = getCartesian(index)
		rho = np.sqrt(cart[0]**2 + cart[1]**2)
		if rho != 0:
			flux = -curX[index]*cart[0]/rho
		else:
			flux = 0
		current2D[index[0],index[1]] += flux*spacing
		current += flux*spacing**3

	interp = interp2d(xArray,yArray,np.transpose(current2D),kind="cubic")

	numAngles = 100
	angles = np.linspace(0,2*np.pi,numAngles,endpoint=False)
	radii = np.arange(0.01,6,0.01)
	jphi = []
	for r in radii:
		integ = 0
		for a in angles:
			integ += interp(r*np.cos(a),r*np.sin(a))/(numAngles)
		jphi.append(1000*integ)
	return (radii,jphi)

p=mp.Pool()
data = p.map(plotJphi,direcs)
for i in range(len(data)):
	fig,ax = plt.subplots()
	ax.axis('off')
	plt.plot(data[i][0],data[i][1],linewidth=5.0)
	plt.ylim((-0.04,0.16))
	plt.axhline(y=0,color='black')
	plt.savefig("13_"+["t=100","t=200","t=300"][i]+".png")
	plt.close()

# plt.legend(["t=50","t=100","t=150","t=200","t=250"])
# plt.xlabel("Cylindrical radius")
# plt.title("Radial Current Distribution")
# plt.axvline(x=2.58,linestyle='--',color='red')
# plt.axhline(y=0,color='black')
# plt.savefig("jphi.png")
# plt.close()
