import numpy as np

class SpikingNetwork():
    def __init__(self):
        # Defining network model parameters
        self.vt = 2.                     # Spiking threshold
        self.tau_m = float(10)                 # Membrane time constant [ms]
        self.g_m = 1                    # Neuron conductance
        self.Nsig = 1.0                 # Variance amplitude of current
        self.Nmean = 0.0                # Mean current to neurons
        self.tau_I = 10.                 # Time constant to filter the synaptic inputs
        self.N = 5000                   # Number of neurons in total
        self.NE = int(0.5*self.N)            # Number of excitatory neurons
        self.NI = int(0.5*self.N)            # Number of inhibitory neurons
        self.dt = 1                    # Simulation time bin [ms]
        self.T = int(300/self.dt)            # Simulation length 
        self.W = 100/float(self.N)             # Connectivity strength
        
        # Initialisation
        self.v = np.random.random((self.N, self.T))* self.vt    # membrane potential
        self.vv = np.zeros((self.N, self.T))                    # variable that notes if v crosses the threshold
        self.Iback = np.zeros(self.N)                 # building up the external current
        self.SP = 0                                   # recording spike times
        self.Ichem = np.zeros(self.N)                 # current coming from synaptic inputs
        self.Iext = np.zeros(self.N)                  # external current
        self.raster = np.array([])                    # save spike times for plotting
    
    def trial(self):
        for t in range(self.T-1):
            self.Iback = self.Iback + self.dt/self.tau_I*(-self.Iback + np.random.random(self.N)) # generate a colored noise for the current
            self.Iext = self.Iback/np.sqrt(1./(2*(self.tau_I/self.dt)))*self.Nsig+self.Nmean      # rescaling the noise current to have the correct mean and variance
            
            # current to excitatory neurons coming from the synaptic inputs
            self.Ichem[:self.NE] = self.Ichem[:self.NE] + self.dt/self.tau_I*\
            (-self.Ichem[:self.NE]+self.W*(np.sum(self.vv[:, t][:self.NE])-self.vv[:, t][:self.NE])-self.W*(np.sum(self.vv[:, t][self.NE:]))) 
            
            # current to inhibitory neurons coming from the synaptic inputs
            self.Ichem[self.NE:] = self.Ichem[self.NE:] + self.dt/self.tau_I*\
            (-self.Ichem[self.NE:]-self.W*(np.sum(self.vv[:, t][self.NE:])-self.vv[:, t][self.NE:])+self.W*(np.sum(self.vv[:, t][:self.NE])))
            
            self.Itot = self.Iext + self.Ichem
            
            # Integrate and Fire Model
            self.v[:,t+1] = self.v[:,t] + float(self.dt)/self.tau_m * (-self.g_m * self.v[:,t] + self.Itot)    # Euler method for IF neuron
            
            self.vv[:,t+1] = (self.v[:,t+1] < self.vt).astype(int)
            self.v[:,t+1] = self.v[:,t+1] * self.vv[:,t+1] 
            

