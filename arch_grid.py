import numpy as np
import matplotlib.pyplot as plt
from random import sample
from collections import deque
import sys

tol = 1e-6
eps = 5e-2
# =============================================================================
# CLASS NETWORK
# =============================================================================

class Network:
    def __init__(self, transfer_matrix, E, T):
        self.number_clusters = 0
        self.clusters = deque()
        self.transfer_matrix = transfer_matrix
        # renormalization of coefficients if they do not sum to 1
        self.energy_coef = E / (E + T)
        self.time_coef   = T / (E + T)
        
    def add_cluster(self, cluster):
        self.clusters.append(cluster)
        self.number_clusters += 1
        
    def assign(self, cluster, job):
        job.cluster = cluster
        self.clusters[cluster].add_to_queue(job)
        
    def get_number_cluster(self):
        return self.number_clusters
    
    def get_transfer_time(self,origin,destination):
        # return transfer time from original cluster to destination cluster
        return self.transfer_matrix[origin,destination]
        
# =============================================================================
# CLASS JOB
# =============================================================================
        
class Job:
    def __init__(self, load, wtime, network, run = False, wait = True, end = False):
        self.counter += 1
        self.network = network
        self.id = self.counter
        self.cluster = 0
        self.load = load          # q
        self.waiting_time = wtime # theta
        # flags
        self.is_running = run
        self.is_waiting = wait
        self.is_ended = end

    def run(self, dt):
        # decrease the load according to the ODE, explicit Euler scheme
        self.load -= self.network.clusters[self.cluster].computation_speed*dt*(self.load>1e-6)
        
    def wait(self, dt, network, old_cluster, new_cluster):
        # decrease the waiting time according to the ODE, explicit Euler scheme
        if self.is_running and old_cluster != new_cluster :
            transfer_time = network.get_transfer_time(old_cluster,new_cluster)
            self.waiting_time += - self.waiting_time + transfer_time
        else :
            self.waiting_time -= dt

# =============================================================================
# CLASS CLUSTER
# =============================================================================
        
class Cluster:
    def __init__(self, network, pos, speed, power, N_proc, tau_pr, delta = 0, security_rate = 0.9):
        self.network = network
        self.position = pos # position of the cluster within the network
        self.computation_speed = speed
        self.energetic_power = power
        self.total_proc = N_proc
        self.max_proc = int(np.floor(security_rate*N_proc))
        self.run_job = deque()    # list of jobs currently running on the cluster
        self.wait_job = deque()   # list of jobs currently in the waiting queue
        self.theta_star = 0
        self.liste_theta_star = deque()
        self.tau_pr = tau_pr
        self.delta = delta
        
    def energy_consumption(self, rate):
        # rate : current rate of occupation of the cluster (between 0 and 1)
        # the energy consumption is considered to be piecewise linear here
        return (rate > 0.5 )*(rate*7 - 3) + (rate <= 0.5 )*rate + self.delta
        
    def add_to_queue(self, job):
        self.wait_job.append(job)
        job.is_waiting = True
        job.is_running = False
        
    def get_len(self):
        return len(self.wait_job)
        
    def get_occup(self):
        # we consider 1 job per processor
        # the utilization is equal to the number of running jobs
        return len(self.run_job)
        
    def Decision(self, network, job):
        # optimizing speed and energy consumption of the network with transfers
        E = network.energy_coef
        T = network.time_coef
        list1 = [] # energy
        list2 = [] # time
        list3 = [] # mixed
        # loop on clusters of the network
        for i in range(network.get_number_cluster()):
            # case 1 : the observed cluster is the current cluster of the job
            if i == job.cluster:
                # job.load*self.energetic_power*
                list1.append(self.energy_consumption(min(self.get_occup()+self.get_len(),self.max_proc ) / self.total_proc))
                list2.append(job.load / self.computation_speed)
            # case 2 : else...
            else :
                cl = network.clusters[i]
                # job.load*cl.energetic_power*
                list1.append((1-eps)*cl.energy_consumption(min(cl.get_occup()+cl.get_len(),cl.max_proc ) / cl.total_proc))
                list2.append(job.load/cl.computation_speed 
                             - cl.theta_star 
                             + network.get_transfer_time(job.cluster,i))
            list3.append(E*list1[i] + T*list2[i])
        # print(list3)
        return np.argmin(list3) # pos of the best cluster   
    
    def update(self,dt):
        
        Nb_job_test = int(dt/self.tau_pr)
        # Nb_job_test = 1
        
        waiting_time = deque()
        
        ## TRANSFER FROM QUEUE TO RUN STATE
        # availibility of the cluster : max processors - occupied processors
        avail = self.max_proc - self.get_occup()
        # update status of jobs from the queue
        
        for job in self.wait_job:
            #if self.position == 1 :
            #   print(job.waiting_time)
            if job.waiting_time <= 0 and (avail > 0) :
                job.is_running = True
                job.is_waiting = False
                self.run_job.append(job)
                avail -= 1
        
        self.wait_job = deque([job for job in self.wait_job if job.is_waiting])

        for i in range(len(self.wait_job)):
            if self.wait_job[i].waiting_time <= 0 : # not transfering or on waiting
                waiting_time.append(self.wait_job[i].waiting_time) 
        
        if len(waiting_time) > 0:
            self.theta_star = min(waiting_time)
            self.liste_theta_star.append(self.theta_star)
        else:
            self.theta_star = 0
            self.liste_theta_star.append(self.theta_star)
            
        sample_for_test = deque(sample(self.run_job,min(Nb_job_test,len(self.run_job))))
        # sample_for_test = deque(sample(self.run_job,len(self.run_job)))
        other_jobs = deque([job for job in self.run_job if job not in sample_for_test])
        
        ## RUN THE CURRENT JOBS 
        for job in sample_for_test:
            job.run(dt)
            new_cluster = self.Decision(self.network, job)
            job.wait(dt, self.network, self.position, new_cluster)
            # proceed the transfer
            if new_cluster != job.cluster :
                self.network.assign(new_cluster,job)
        
        for job in other_jobs:
            job.run(dt)
            job.wait(dt, self.network, self.position,self.position)
        
        self.run_job = deque([job for job in self.run_job if job.cluster == self.position and job.load > tol])
        
        ## COMPUTE WAITING TIME FOR JOBS IN THE QUEUE
        for job in self.wait_job:
            # update the waiting time of waiting jobs
            job.wait(dt, self.network, self.position,self.position)