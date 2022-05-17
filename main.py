from arch_grid import *
import numpy.linalg as npl
import time
# =============================================================================
# PRE-PROCESSING
# =============================================================================        
Job.counter = 0
plt.close('all')
# Network init
c = 100000000000
A = np.array([[0,c],[c,0]])
coeff_energy = 0
coeff_time = 1
tau_pr = 1e-4 #eps^2
dt = 1e-2 #eps
T = 1


print("tau_pr = %f"%tau_pr)
print("dt = %f"%dt)
print("T = %i"%T)

Echelle_nb_Jobs = int(T/tau_pr)
Nb_Jobs = 1000
if abs(np.log(Echelle_nb_Jobs/Nb_Jobs)) > np.log(99.9) :
    print('Scale warning /!\ Nb_Jobs : %i & Scale : %i'%(Nb_Jobs,Echelle_nb_Jobs))
#N_proc = int(Nb_Jobs/20)
N_proc = 500
if abs(np.log(N_proc/Nb_Jobs)) > np.log(9.9) :
    print('Scale warning /!\ Nb_Jobs : %i & N_proc : %i'%(Nb_Jobs,N_proc))
#N_proc = int(Nb_Jobs/20)

network1 = Network(A,coeff_energy,coeff_time)
# Cluster parameters : network | position | speed | power | Nb processors
cluster1 = Cluster(network1, 0, 50 , 1, N_proc, tau_pr)
cluster2 = Cluster(network1, 1, 100, 1, N_proc, tau_pr, delta = 1e-3)
# Add clusters to the current network
network1.add_cluster(cluster1)
network1.add_cluster(cluster2)

# Jobs init
n = np.arange(0,Nb_Jobs)
np.random.seed(2)
rand_load = np.random.uniform(50,150,len(n))
# rand_load = np.ones(len(n))*100
rand_wtime = np.zeros(len(n))
# rand_wtime = np.random.uniform(0,10,len(n))
# Assign jobs
for i in n:
    job = Job(rand_load[i], rand_wtime[i],  network1)
    network1.assign(cluster1.position, job)

# =============================================================================
# SIMULATION
# =============================================================================   

t0 = 0
tf = 50
N = int((tf-t0)/dt)
tn = np.linspace(t0,tf,N+1)

t1 = time.time()
nb_job_soumis = 0
load1 = []
list1 = []
load2 = []
list2 = []
f = 100
per = 1/f
stack = 0
n = 1
for t in tn:
    if np.isclose(stack, per) :
        if stack >= per :
            for i in range(n):
                network1.assign(cluster1.position, Job(225, 0,  network1))
                network1.assign(cluster2.position, Job(225, 0,  network1))
            stack -= per
        # nb_job_soumis += 1
        # print(t)
    stack += dt
    cluster1.update(dt)
    # list1.append(cluster1.get_len())
    cluster2.update(dt)
    # list2.append(cluster2.get_len())
    load1.append(100*cluster1.get_occup()/cluster1.total_proc)
    load2.append(100*cluster2.get_occup()/cluster2.total_proc)
    if np.isclose(t%1, 0) :
        print("SIMULATION TIME : %2.2f s" %(t))
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
# ax3.plot(tn,list1,'g-')
# ax3.set(xlabel='Time (s)', ylabel='Waiting queue')
# ax3.grid()
fig1.show()

fig2, (ax1, ax2) = plt.subplots(2, sharex=True)
fig2.suptitle('Cluster 2')
ax1.plot(tn,load2,'r-')
ax1.set(xlabel='Time (s)', ylabel='CPU Occupation (%)')
ax1.set_ylim(-5,105); ax1.grid()
ax2.plot(tn,cluster2.liste_theta_star,'b-')
ax2.set(xlabel='Time (s)', ylabel='\u03B8* (sec)')
ax2.grid()
# ax3.plot(tn,list2,'g-')
# ax3.set(xlabel='Time (s)', ylabel='Waiting queue')
# ax3.grid()
fig2.show()

print('Duration of the simulation : %.3f seconds'%(t2-t1))

# dt = np.array([1e-1,5e-2,1e-2,5e-3,1e-3])
# time = np.array([2.335,4.715,12.874,23.014,100.1])

# X = np.array([np.ones(len(dt)),np.log(dt)]).T
# y = np.log(time)

# line = npl.solve(X.T.dot(X),X.T.dot(y))

# plt.close('all')
# plt.plot(np.log(dt),np.log(time),'r*-',label='Log of time simulation')
# plt.plot(np.log(dt),-(3/4)*np.log(dt)+line[0],'--',label='o($dt^{-3/4}$)')
# plt.title('Time of simulation, depending on the time step')
# plt.xlabel('log(dt)')
# plt.ylabel('log(t)')
# plt.legend()
# plt.grid()
# plt.show()
