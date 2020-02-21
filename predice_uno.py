import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
import sklearn.metrics as cf

numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)

data = imagenes.reshape((n_imagenes, -1))

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def predict(data, target, inter):
	scaler = StandardScaler()
	x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.7)

	x_train = scaler.fit_transform(x_train)
	x_test = scaler.transform(x_test)

	cov = np.cov(x_train.T)
	valores, vectores = np.linalg.eig(cov)
	valores = np.real(valores)
	vectores = np.real(vectores)
	ii = np.argsort(-valores)
	valores = valores[ii]
	vectores = vectores[:,ii]

	pc1_train = np.dot(vectores[:,0],x_train.T)
	pc2_train = np.dot(vectores[:,1],x_train.T)
	pc3_train = np.dot(vectores[:,2],x_train.T)

	interval_x = [pc1_train[np.where(y_train == 1)][0]-inter, pc1_train[np.where(y_train == 1)][0]+inter]
	interval_y = [pc2_train[np.where(y_train == 1)][0]-inter, pc2_train[np.where(y_train == 1)][0]+inter]
	interval_z = [pc3_train[np.where(y_train == 1)][0]-inter, pc3_train[np.where(y_train == 1)][0]+inter]

	ix_tr = [np.where((pc1_train >= interval_x[0]) & (pc1_train <= interval_x[1]))]
	iy_tr = [np.where((pc2_train >= interval_y[0]) & (pc2_train <= interval_y[1]))]
	iz_tr = [np.where((pc3_train >= interval_z[0]) & (pc3_train <= interval_z[1]))]

	ii_tr = (np.intersect1d(ix_tr,np.intersect1d(iy_tr,iz_tr)))

	cov = np.cov(x_test.T)
	valores, vectores = np.linalg.eig(cov)
	valores = np.real(valores)
	vectores = np.real(vectores)
	ii = np.argsort(-valores)
	valores = valores[ii]
	vectores = vectores[:,ii]

	pc1_test = np.dot(vectores[:,0],x_test.T)
	pc2_test = np.dot(vectores[:,1],x_test.T)
	pc3_test = np.dot(vectores[:,2],x_test.T)

	ix = [np.where((pc1_test >= interval_x[0]) & (pc1_test <= interval_x[1]))]
	iy = [np.where((pc2_test >= interval_y[0]) & (pc2_test <= interval_y[1]))]
	iz = [np.where((pc3_test >= interval_z[0]) & (pc3_test <= interval_z[1]))]

	ii = (np.intersect1d(ix,np.intersect1d(iy,iz)))

	truth_train = np.copy(y_train)
	truth_test = np.copy(y_test)
	predict_train = np.zeros(len(truth_train))
	predict_test = np.zeros(len(truth_test))
	truth_train[np.where(y_train != 1)] = 0
	truth_test[np.where(y_test != 1)] = 0

	predict_test[ii] = 1
	predict_train[ii_tr] = 1


	return truth_train, truth_test, predict_train, predict_test

truth_train, truth_test, predict_train, predict_test = predict(data,target,3)
confusion_train = cf.confusion_matrix(truth_train,predict_train)
confusion_test = cf.confusion_matrix(truth_test,predict_test)

def f1(confusion):
	p = confusion[1,1]/(confusion[0,1] + confusion[1,0])
	r = confusion[1,1]/(confusion[1,1] + confusion[0,0])

	return 2*p*r/(p+r)

print(f1(confusion_train),f1(confusion_test))

plt.plot()
plt.subplot(121)
plt.imshow(confusion_train)
plt.text(0,0,'TN = {}'.format(confusion_train[0,0]))
plt.text(1,0,'FP = {}'.format(confusion_train[0,1]))
plt.text(0,1.25,'TP = {}'.format(confusion_train[1,1]))
plt.text(1,1.25,'FN = {}'.format(confusion_train[1,0]))
plt.title('F1 train = {:.3f}'.format(f1(confusion_train)))
plt.subplot(122)
plt.imshow(confusion_test)
plt.text(0,0,'TN = {}'.format(confusion_test[0,0]))
plt.text(1,0,'FP = {}'.format(confusion_test[0,1]))
plt.text(0,1.25,'TP = {}'.format(confusion_test[1,1]))
plt.text(1,1.25,'FN = {}'.format(confusion_test[1,0]))
plt.title('F1 test = {:.3f}'.format(f1(confusion_test)))
plt.savefig('matriz_confusion.png')



