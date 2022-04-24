from arch_grid import *
import time
# =============================================================================
# PRE-PROCESSING
# =============================================================================        


# Network init
c = 1.0
A = np.array([[0,c],[c,0]])
coeff_energy = 0
coeff_time = 1
tau_pr = 1e-4 #eps^2
dt = 1e-4 #eps
T = 1

print("tau_pr = %f"%tau_pr)
print("dt = %f"%dt)
print("T = %i"%T)

Nb_Jobs = int(T/tau_pr)
N_proc = int(Nb_Jobs/20)

network1 = Network(A,coeff_energy,coeff_time)
# Cluster parameters : network | position | speed | power | Nb processors
cluster1 = Cluster(network1, 0, 50 , 1, N_proc, tau_pr)
cluster2 = Cluster(network1, 1, 100, 1, N_proc, tau_pr)
# Add clusters to the current network
network1.add_cluster(cluster1)
network1.add_cluster(cluster2)

# Jobs init
n = np.arange(0,Nb_Jobs)
np.random.seed(2)
rand_load = np.random.uniform(50,200,len(n))
rand_wtime = np.zeros(len(n))
#rand_wtime = np.random.uniform(0,10,len(n))
# Assign jobs
for i in n:
    network1.assign(cluster1.position, Job(i, rand_load[i], rand_wtime[i],  network1))

# =============================================================================
# SIMULATION
# =============================================================================   

t0 = 0
tf = 10
N = int((tf-t0)/dt)
tn = np.linspace(t0,tf,N+1)

t1 = time.time()
load1 = deque()
load2 = deque()
for t in tn:
    cluster1.update(dt)
    cluster2.update(dt)
    load1.append(100*cluster1.get_occup()/cluster1.total_proc)
    load2.append(100*cluster2.get_occup()/cluster2.total_proc)
    # print("SIMULATION TIME : %2.2f s" %(t))
t2 = time.time()
# =============================================================================
# POST-PROCESSING
# =============================================================================   

fig1, (ax1, ax2) = plt.subplots(2, sharex=True)
fig1.suptitle('Cluster 1')
ax1.plot(tn,load1,'r-')
ax1.set(xlabel='Time (s)', ylabel='CPU Occupation (%)')
ax1.set_ylim(-5,105); ax1.grid()
ax2.plot(tn,cluster1.liste_theta_star,'b-')
ax2.set(xlabel='Time (s)', ylabel='\u03B8* (sec)')
ax2.grid()
fig1.show()

fig2, (ax1, ax2) = plt.subplots(2, sharex=True)
fig2.suptitle('Cluster 2')
ax1.plot(tn,load2,'r-')
ax1.set(xlabel='Time (s)', ylabel='CPU Occupation (%)')
ax1.set_ylim(-5,105); ax1.grid()
ax2.plot(tn,cluster2.liste_theta_star,'b-')
ax2.set(xlabel='Time (s)', ylabel='\u03B8* (sec)')
ax2.grid()
fig2.show()

print('Duration of the simulation : %.3f seconds'%(t2-t1))