# Computational Neuroscience

## `SpikingNetwork.py` usage

~~~python
SN = SpikingNetwork()
SN.T = 200
SN.N = 200
SN.vt = 2.4
SN.W /= 10
SN.parameter_initialisation()
SN.trial()

# Plotting

plt.figure(figsize=(20,20))
plt.pcolormesh(SN.v, cmap=plt.cm.get_cmap('PuBu'))
dots = np.ma.masked_where(SN.vv != 1, SN.vv)
plt.pcolormesh(dots, cmap=plt.cm.get_cmap('spring'))
plt.xlabel('Time (ms)')
plt.ylabel('Neuron number')
plt.title('Neuronal activity')
plt.xlim(0, SN.T)

plt.xlabel('Time (ms)')
plt.ylabel('Neuron number')
plt.title('Spike times');

# Output:
~~~

![](http://i.imgur.com/f749x4Z.png)



***
