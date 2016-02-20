########################################################################################################################
# Heuristic Solver for the Tool Switching Problem
#
#   Exercises for WS15 Computational Techniques
#   Group members (alphabetical): Morariu, Smolka, Zeba
########################################################################################################################

import copy
import random
import math
import itertools
import collections

########################################################################################################################
#
# Neighborhood Structures
# (Exercise 3)
#
########################################################################################################################
def singleRandomSwapNeighborhood(job_sequence,resolution=1):
    for i in range(1,len(job_sequence)*resolution):
        new_sequence = copy.deepcopy(job_sequence)
        a,b=0,0
        while a==b:
            a=random.randrange(0,len(job_sequence))
            b=random.randrange(0,len(job_sequence))
        new_sequence[a],new_sequence[b]=new_sequence[b],new_sequence[a]
        yield new_sequence

def singleNeighborSwapNeighborhood(job_sequence):
    for i in range(1,len(job_sequence)):
        new_sequence = copy.deepcopy(job_sequence)
        new_sequence[i],new_sequence[i-1]=new_sequence[i-1],new_sequence[i]
        yield new_sequence

def dualNeighborSwapNeighborhood(job_sequence):
    for i in range(1,len(job_sequence)):
        new_sequence = copy.deepcopy(job_sequence)
        new_sequence[i],new_sequence[i-1]=new_sequence[i-1],new_sequence[i]

        for sequence in singleNeighborSwapNeighborhood(new_sequence):
            yield sequence

def singlePairSwapNeighborhood(job_sequence):
    for i in range(len(job_sequence)):
        for j in range(i+1,len(job_sequence)):
            if i!=j:
                new_sequence = copy.deepcopy(job_sequence)
                new_sequence[i],new_sequence[j]=new_sequence[j],new_sequence[i]
                yield new_sequence

def singleSliceSwapNeighborhood(job_sequence,k=None):
    if k==None:
        k=len(job_sequence)/10

        if k==0:
            k=min(3,len(job_sequence))

    for i in range(len(job_sequence)/k):
        for j in range(i+1,len(job_sequence)/k):
            if i!=j:
                new_sequence = copy.deepcopy(job_sequence)
                new_sequence[i*k:i*k+k],new_sequence[j*k:j*k+k]=new_sequence[j*k:j*k+k],new_sequence[i*k:i*k+k]
                yield new_sequence

def singleSliceRandomizeNeighborhood(job_sequence,k=None):
    if k==None:
        k=len(job_sequence)/10

        if k==0:
            k=min(3,len(job_sequence))

    for i in range(len(job_sequence)/k):
        new_sequence = copy.deepcopy(job_sequence)
        new_slice = []
        old_slice = copy.deepcopy(new_sequence[i*k:i*k+k])
        while len(old_slice):
            choice=random.choice(old_slice)
            new_slice.append(choice)
            old_slice.remove(choice)

        new_sequence[i*k:i*k+k] = new_slice
        yield new_sequence

def rotatingNeighborhood(job_sequence):
    new_sequence = collections.deque(job_sequence)
    for i in range(len(job_sequence)):
        new_sequence.rotate(1)
        yield list(new_sequence)

def recursiveRandomSwapNeighborhood(job_sequence,resolution=1,depth=3):
    for new_sequence in singleRandomSwapNeighborhood(job_sequence):
        yield new_sequence

        if depth:
            for other_sequence in recursiveRandomSwapNeighborhood(new_sequence,resolution,depth-1):
                yield other_sequence

def multiRandomSwapNeighborhood(job_sequence,resolution=100,multi_resolution=0.1):
    for i in range(len(job_sequence)*resolution):
        new_sequence = copy.deepcopy(job_sequence)

        for j in range(int(len(job_sequence)*multi_resolution)):
            a,b=0,0
            while a==b:
                a=random.randrange(0,len(job_sequence))
                b=random.randrange(0,len(job_sequence))
            new_sequence[a],new_sequence[b]=new_sequence[b],new_sequence[a]
        yield new_sequence

def multiRandomNeighborSwapNeighborhood(job_sequence,resolution,depth):
    for k in range(len(job_sequence)*resolution):
        new_sequence = copy.deepcopy(job_sequence)
        for l in range(depth):
            i=random.randrange(1,len(job_sequence))
            new_sequence[i],new_sequence[i-1]=new_sequence[i-1],new_sequence[i]
        yield new_sequence

def combinedNeighborhood(job_sequence,neighborhood_generator_function_a,neighborhood_generator_function_b):
    for sequence_a in neighborhood_generator_function_a(job_sequence):
        if random.uniform(0,1) > 0.1:
            continue

        for sequence_b in neighborhood_generator_function_b(sequence_a):
            if random.uniform(0,1) > 0.1:
                continue

            yield sequence_b

########################################################################################################################
#
# Class for loading and solving ToSP problems with various (meta)heuristics
#
########################################################################################################################
class HeuristicSolver:
    """Heuristic Solver for the Tool Switching Problem

       Exercises for WS15 Computational Techniques
       Group members (alphabetical): Morariu, Smolka, Zeba
    """

    def __init__(self):
        self.n = 0 #Jobs
        self.m = 0 #Tools
        self.A = None #n x m Incidence Matrix
        self.C = None #Machine capacity

        self.log = None
        self.console_output = True

        self.stats = {"total_objective_evaluations":0,
                      "total_improvements":0,
                      "total_iterations":0,
                      "total_neighborhoods":0,
                      "objective_evaluations_since_last_improvement":0,
                      "best_objective_value": -1,
                      "improvements_by_neighborhood": {} }

    def setC(self,c):
        """Set machine (tool) capacity"""
        self.C=c

    def getToolsForJob(self,j):
        """Returns the set of tools job j requires."""
        #TODO: Maybe precompute
        return set([i for i in range(self.m) if self.A[j][i] == 1])


    ########################################################################################################################
    #
    # Tool Sequence Optimization - Evaluation of Objective Function
    #
    ########################################################################################################################
    def minimizeSwitchesForJobSequence(self,job_sequence,abort_after_max=None):
        """For the given job sequence, return a sequence of tool configurations that tries to
        minimize the number of total tool switches.

        (Exercise 1)

        Adaption: Incremental switch count evaluation: Option to abort after exceeding a certain switch count

        Algorithm:

           1. Initialize the first tool configuration with the tools needed by the first job
           2. Until the first configuration is full:
                 Add a tool that is not yet in the configuration, and will be needed the soonest

           3. For each job in the sequence, after the first:
                 Find the tools that need to be swapped in (set difference of new needed tools minus current configuration)
                 Sort the tools the can be swapped out by for how long they will not be needed
                 While there are tools that need to be swapped in:
                     Switch out the next longest not needed tool
                     Switch in the next needed tool
                     Increment switch count
        """

        self.onEvaluateObjectiveFunction()

        #TODO: More efficient implementation just for evaluating objective function

        i=1
        def rankTool(t): #Assigns a score to a tool based on the time until it will be needed again
            for k in range(i+1,len(job_sequence)):
                if t in self.getToolsForJob(job_sequence[k]):
                    return (k-(i+1))
            return len(job_sequence)

        switches=0

        #Fill up empty slots of tool_sequence[0] with tools by ranking them by how soon they will be needed
        first_tools=self.getToolsForJob(job_sequence[0])

        additional_tools=sorted([tool for tool in range(self.m) if not tool in first_tools],key=rankTool)
        additional_tools_index=0

        while len(first_tools) < self.C and additional_tools_index < len(additional_tools):
            first_tools.add(additional_tools[additional_tools_index])
            additional_tools_index+=1

        tool_sequence=[first_tools]

        #We know which tools we must swap in after every job, so we search for the optimal tools to swap out.
        for i in range(1,len(job_sequence)):
            current_tools=tool_sequence[i-1]
            needed_tools=self.getToolsForJob(job_sequence[i])
            switch_in_tools=needed_tools.difference(current_tools)

            next_tools=copy.deepcopy(current_tools)

            switch_candidates = sorted([tool for tool in current_tools if not tool in needed_tools],key=rankTool,reverse=True)
            switch_candidate_index = 0

            for needed in switch_in_tools:
                if len(next_tools) >= self.C:
                    if switch_candidate_index < len(switch_candidates):
                        next_tools.remove(switch_candidates[switch_candidate_index])
                        switch_candidate_index += 1
                    else:
                        raise ValueError("Machine capacity too small for job")

                next_tools.add(needed)
                switches += 1

                if abort_after_max != None and switches >= abort_after_max:
                    return False,switches

            tool_sequence.append(next_tools)

        return (tool_sequence,switches)

    ########################################################################################################################
    #
    # Construction Heuristics
    #
    ########################################################################################################################
    def constructJobSequenceLinear(self):
        return [i for i in range(self.n)]

    def constructJobSequenceRandom(self):
        jobs=[i for i in range(self.n)]
        job_sequence=[]
        while len(jobs):
            j=random.choice(jobs)
            job_sequence.append(j)
            jobs.remove(j)
        return job_sequence

    def constructJobSequenceGreedy(self):
        """Construct a job sequence minimizing tool switches using a greedy algorithm.

           (Exercise 2)

           Algorithm:
                1. Find pair of jobs with the largest common needed tool denominator
                2. Add pair to the job sequence
                2. For each remaining job:
                    Find the remaining job with the largest common needed tool denominator with the last job added to the job sequence
                    Add the chosen job to the sequence
        """
        jobs=set([i for i in range(self.n)])
        job_sequence=[]

        best_pair=None
        best_pair_intersect=-1

        for j in jobs:
            for l in jobs:
                if j!=l:
                    tools_j=self.getToolsForJob(j)
                    tools_l=self.getToolsForJob(l)
                    intersect=len(tools_j.intersection(tools_l))
                    if intersect > best_pair_intersect:
                        best_pair_intersect=intersect

                        #Load the job needing more tools first (first loading is free)
                        if len(tools_j)>len(tools_l):
                            best_pair=(j,l)
                        else:
                            best_pair=(l,j)

        best_pair_a,best_pair_b=best_pair
        jobs.remove(best_pair_a)
        jobs.remove(best_pair_b)

        job_sequence=[best_pair_a,best_pair_b]

        while len(jobs):
            last_job=job_sequence[-1]
            last_tools=self.getToolsForJob(last_job)
            best=None
            best_intersect=-1

            for j in jobs:
                tools_j=self.getToolsForJob(j)
                intersect=len(last_tools.intersection(tools_j))
                if intersect > best_intersect:
                    best=j
                    best_intersect=intersect

            job_sequence.append(best)
            jobs.remove(best)

        return job_sequence

    def constructJobSequencesGreedyJobClustering(self):
        """Construct a number of job sequences minimizing tool switches using an advanced greedy algorithm.

           (Exercise 2)

           Algorithm:
                1. Group jobs into clusters based on intersection count
                2. Greedily order jobs within clusters to maximize neighbor intersects
                3. Permutate clusters to create job sequence variants
        """
        jobs=set([i for i in range(self.n)])
        clusters = [set() for i in range(self.n)]
        mates = [0 for i in range(self.n)]

        for j in jobs:
            best_mate = None
            best_mate_intersect = 0
            for l in jobs:
                if j!=l:
                    tools_j=self.getToolsForJob(j)
                    tools_l=self.getToolsForJob(l)
                    intersect=len(tools_j.intersection(tools_l))
                    if intersect > best_mate_intersect:
                        best_mate_intersect=intersect
                        best_mate=l

            mates[j]=best_mate

        added=True
        while added:
            added=False
            for job,mate in enumerate(mates):
                if not mate in clusters[job]:
                    clusters[job].add(mate)
                    added=True

                for k,cluster in enumerate(clusters):
                    if mate in cluster:
                        if not job in cluster:
                            cluster.add(job)
                            added=True

        superclusters=[]

        for cluster in clusters:
            intersected=False
            for supercluster in superclusters:
                if len(supercluster.intersection(cluster))!=0:
                    intersected=True
                    for item in cluster:
                        supercluster.add(item)

            if not intersected:
                superclusters.append(cluster)

        for i in range(len(superclusters)):
            for j in range(len(superclusters)):
                if i!=j and len(superclusters[i].intersection(superclusters[j])):
                    raise

        ordered_superclusters = []

        for cluster in superclusters:
            ordered_cluster=[]

            best_pair=None
            best_pair_intersect=-1

            for j in cluster:
                for l in cluster:
                    if j!=l:
                        tools_j=self.getToolsForJob(j)
                        tools_l=self.getToolsForJob(l)
                        intersect=len(tools_j.intersection(tools_l))
                        if intersect > best_pair_intersect:
                            best_pair_intersect=intersect

                            #Load the job needing more tools first (first loading is free)
                            if len(tools_j)>len(tools_l):
                                best_pair=(j,l)
                            else:
                                best_pair=(l,j)

            best_pair_a, best_pair_b = best_pair
            ordered_cluster.append(best_pair_a)
            ordered_cluster.append(best_pair_b)
            cluster.remove(best_pair_a)
            cluster.remove(best_pair_b)

            while(len(cluster)):
                last_job=ordered_cluster[-1]
                last_tools=self.getToolsForJob(last_job)
                best=None
                best_intersect=-1

                for j in cluster:
                    tools_j=self.getToolsForJob(j)
                    intersect=len(last_tools.intersection(tools_j))
                    if intersect > best_intersect:
                        best=j
                        best_intersect=intersect

                ordered_cluster.append(best)
                cluster.remove(best)

            ordered_superclusters.append(ordered_cluster)

        job_sequences=[]

        for permutation in itertools.permutations(ordered_superclusters):
            job_sequence=[]
            for cluster in permutation:
                for item in cluster:
                    job_sequence.append(item)
            job_sequences.append(job_sequence)

        return job_sequences

    def constructJobSequenceGreedyRandomized(self, alpha=None):
        """Construct a job sequence mimizing tool switches using a randomized greedy algorithm.

           (Exercise 6.a)

           Algorithm:
                1. For each remaining job:
                    CL=Remaining jobs
                    RCL=Remaining jobs with promising intersection
                    Add a random job from RCL to the job sequence
        """
        if alpha==None:
            alpha=random.uniform(0,1)

        jobs=set([i for i in range(self.n)])
        job_sequence=[]

        while len(jobs):
            max_intersect=0
            min_intersect=100000

            for j in jobs:
                tools_j=self.getToolsForJob(j)
                if len(job_sequence):
                    intersect=len(self.getToolsForJob(job_sequence[-1]).intersection(tools_j))
                else:
                    intersect=len(tools_j)

                max_intersect=max(max_intersect,intersect)
                min_intersect=min(min_intersect,intersect)

            RCL=[]
            for j in jobs:
                tools_j=self.getToolsForJob(j)
                if len(job_sequence):
                    intersect=len(self.getToolsForJob(job_sequence[-1]).intersection(tools_j))
                else:
                    intersect=len(tools_j)

                if intersect >= max_intersect - alpha * (max_intersect-min_intersect):
                    RCL.append(j)

            chosen=random.choice(RCL)

            job_sequence.append(chosen)
            jobs.remove(chosen)

        return job_sequence

    ########################################################################################################################
    #
    # Local Search
    #
    ########################################################################################################################
    def nextBestJobSequence(self,job_sequence,neighborhood_generator,first_improvement=True):
        """Try to improve the job sequence by evaluating the given neighborhood
           (Exercise 3)
        """

        best_job_sequence = job_sequence
        _, min_switches = self.minimizeSwitchesForJobSequence(best_job_sequence)

        improved=False
        for sequence in neighborhood_generator:
            #Incremental evaluation aborts if switch count is higher/equal than current minimum
            _, new_min_switches = self.minimizeSwitchesForJobSequence(sequence, min_switches)

            if new_min_switches < min_switches:
                best_job_sequence=sequence
                min_switches=new_min_switches
                improved=True

                if not neighborhood_generator.__name__ in self.stats["improvements_by_neighborhood"]:
                    self.stats["improvements_by_neighborhood"][neighborhood_generator.__name__]=0
                self.stats["improvements_by_neighborhood"][neighborhood_generator.__name__]+=1

                if first_improvement:
                    break

        return best_job_sequence,min_switches,improved

    def bestJobSequenceLocalSearch(self,job_sequence,neighborhood_generator_function):
        """Perform local search iterations on the job sequence until no more improvement can be made in switch count
           (Exercise 3)
        """
        improved = True
        while improved:
            self.onEvaluateNeighborhood()
            neighborhood=neighborhood_generator_function(job_sequence)
            job_sequence,min_switches,improved=self.nextBestJobSequence(job_sequence,neighborhood)
            self.onNewBest(min_switches)
            #print(min_switches)

        return job_sequence,min_switches

    ########################################################################################################################
    #
    # Advanced Heuristics
    #
    ########################################################################################################################
    def bestJobSequenceVariableNeighborhoodSearch(self,job_sequence,neighborhood_generator_function_cycle, combine_step=False):
        """Perform local search iterations on the job sequence, by searching in a cycle of neighborhoods, until no more improvement can be made in switch count
           (Exercise 5)
        """
        improved = True
        while improved:
            #print("")
            self.onIterate()
            for i in range(len(neighborhood_generator_function_cycle)):
                self.onEvaluateNeighborhood()
                neighborhood_generator_function=neighborhood_generator_function_cycle[i]
                #print(" "*i + neighborhood_generator_function.__name__ )

                neighborhood=neighborhood_generator_function(job_sequence)
                job_sequence,min_switches,improved=self.nextBestJobSequence(job_sequence,neighborhood)

                if improved:
                    #print(" " *i + "New minimum: %d @ %d evals"%(min_switches,self.stats["total_objective_evaluations"]))
                    self.onNewBest(min_switches)
                    break
                else:
                    #Swap the neighborhood to the end of the cycle, because it may not perform well in the current function area
                    pass

        if combine_step:
            improved = True
            while improved:
                #print("")
                self.onIterate()
                for i in range(len(neighborhood_generator_function_cycle)):
                    self.onEvaluateNeighborhood()

                    for j in range(len(neighborhood_generator_function_cycle)):
                        if i==j:
                            continue

                        neighborhood=combinedNeighborhood(job_sequence,neighborhood_generator_function_cycle[i],neighborhood_generator_function_cycle[j])
                        job_sequence,min_switches,improved=self.nextBestJobSequence(job_sequence,neighborhood)

                        if improved:
                            self.onNewBest(min_switches)
                            break
                        else:
                            pass

                    if improved:
                        break

        return job_sequence,min_switches

    def bestJobSequenceGRASP(self,iterations=100):
        """Randomized greedy neighborhood search
           (Exercise 6.a)
        """
        cycle=[singleNeighborSwapNeighborhood,singleRandomSwapNeighborhood]

        best_job_sequence = self.constructJobSequenceGreedyRandomized()
        _,best_switches=self.minimizeSwitchesForJobSequence(best_job_sequence)
        best_job_sequence, best_switches = self.bestJobSequenceVariableNeighborhoodSearch(best_job_sequence, cycle)

        for i in range(iterations):
            self.onIterate()
            job_sequence = self.constructJobSequenceGreedyRandomized()
            _,switches=self.minimizeSwitchesForJobSequence(job_sequence)

            job_sequence, switches = self.bestJobSequenceVariableNeighborhoodSearch(job_sequence, cycle)

            if switches < best_switches:
                best_job_sequence=job_sequence
                best_switches=switches
                self.onNewBest(best_switches)

        return best_job_sequence

    def bestJobSequenceGVNS(self,job_sequence,shaker_neighborhood_function,local_neighborhood_cycle,iterations=3,k_max=100):
        """Generalized variable neighborhood search
           (Exercise 6.b)
        """

        best_job_sequence=job_sequence
        _,best_switches=self.minimizeSwitchesForJobSequence(best_job_sequence)

        for i in range(iterations):
            self.onIterate()
            k=0

            #print("")
            #print("Iteration %d"%i)

            while k < k_max:
                self.onEvaluateNeighborhood()

                shaker_neighborhood=shaker_neighborhood_function(job_sequence,1,k)
                for shake_sequence in shaker_neighborhood:
                    break

                job_sequence,switches=self.bestJobSequenceVariableNeighborhoodSearch(job_sequence,local_neighborhood_cycle)

                if switches<best_switches:
                    #print("New best %d, k=%d"%(switches,k))
                    self.onNewBest(switches)
                    best_job_sequence=job_sequence
                    best_switches=switches
                    k=1
                else:
                    k+=1
                    #print("No new best k=%d"%k)

        return best_job_sequence

    def bestJobSequenceSimulatedAnnealing(self,job_sequence,neighborhood_generator_function,iterations=1000,start_T=1.0):
        """Simulated Annealing
           (Exercise 6.c)
        """
        def acceptance_probability(x,y,T):
            if T==0:
                return y<x
            return 1 if y<x else math.exp(-float(y-x)/T)

        max_evaluations=iterations*25
        evaluations=0

        _, x = self.minimizeSwitchesForJobSequence(job_sequence)

        best_sequence = job_sequence
        best_switches = x

        for k in range(iterations):
            self.onIterate()
            self.onEvaluateNeighborhood()
            neighborhood=neighborhood_generator_function(job_sequence)
            for new_job_sequence in neighborhood:
                evaluations+=1
                if evaluations>max_evaluations:
                    break

                _, y = self.minimizeSwitchesForJobSequence(new_job_sequence)
                T=start_T - start_T * (float(k)/float(iterations))

                if acceptance_probability(x,y,T) >= random.uniform(0,1):
                    job_sequence=new_job_sequence
                    #print(y,y<x,T,self.stats["total_objective_evaluations"])
                    x=y

                    if x < best_switches:
                        self.onNewBest(x)
                        best_switches=x
                        best_sequence = job_sequence
                    break

            if evaluations>max_evaluations:
                break

        return best_sequence

    ########################################################################################################################
    #
    # Experimental Heuristics
    #
    ########################################################################################################################
    def bestJobSequenceGreedyClusteringAllVNS(self,cycle):
        sequences=self.constructJobSequencesGreedyJobClustering()
        best_job_sequence = sequences[0]
        _,best_switches=self.minimizeSwitchesForJobSequence(best_job_sequence)
        best_job_sequence, best_switches = self.bestJobSequenceVariableNeighborhoodSearch(best_job_sequence, cycle)

        for job_sequence in sequences:
            self.onIterate()
            _,switches=self.minimizeSwitchesForJobSequence(job_sequence)

            job_sequence, switches = self.bestJobSequenceVariableNeighborhoodSearch(job_sequence, cycle)

            if switches < best_switches:
                best_job_sequence=job_sequence
                best_switches=switches
                self.onNewBest(best_switches)

        return best_job_sequence

    def bestJobSequenceLocalDecreasingSlices(self,job_sequence,iterations=100):
        """ Experimental!
           (Exercise 3)
        """

        last_improved_size=len(job_sequence)/2

        for i in range(iterations):
            self.onIterate()
            while True:
                slice_size=last_improved_size
                improved = False
                depth=0
                while not improved and slice_size >= 1:
                    self.onEvaluateNeighborhood()
                    neighborhood=singleSliceSwapNeighborhood(job_sequence,slice_size)
                    job_sequence,min_switches,improved=self.nextBestJobSequence(job_sequence,neighborhood)
                    slice_size -= 1

                if not improved:
                    slice_size=len(job_sequence)/2
                    break
                else:
                    last_improved_size=slice_size
                    self.onNewBest(min_switches)

        return job_sequence,min_switches

    def bestJobSequenceMultipleSimulatedAnnealing(self,neighborhood_generator_function,total_iterations=1000):
        solutions=[]
        solutions.append(self.constructJobSequenceGreedy())
        solutions.extend(self.constructJobSequencesGreedyJobClustering())

        for i in range(5):
            solutions.append(self.constructJobSequenceGreedyRandomized())

        for i in range(10):
            solutions.append(self.constructJobSequenceRandom())

        best_solution=solutions[0]
        _, best_switches = self.minimizeSwitchesForJobSequence(best_solution)


        while len(solutions) > 1:
            iterations = total_iterations / len(solutions)

            #print("%d solutions in pool."%len(solutions))
            #print("%d iterations allocated for each."%iterations)

            for i, sequence in enumerate(solutions):
                        #print("\tOptimizing %d of %d"%(i,len(solutions)))
                solutions[i] = self.bestJobSequenceSimulatedAnnealing(sequence,neighborhood_generator_function,iterations)

            def rateSolution(k):
                _, switches = self.minimizeSwitchesForJobSequence(k)
                return switches

            solutions=sorted(solutions,key=rateSolution,reverse=False)

            new_solutions=[]
            for i in range(int((0.5+len(solutions))/2)):
                new_solutions.append(solutions[i])

            #print("OLD",[rateSolution(i) for i in solutions])
            #print("NEW",[rateSolution(i) for i in new_solutions])
            solutions=new_solutions

        iterations = total_iterations / len(solutions)
        solutions[0] = self.bestJobSequenceSimulatedAnnealing(sequence,neighborhood_generator_function,iterations)

        return solutions[0]

    def bestJobSequenceVariableNeighborhoodSimulatedAnnealing(self,job_sequence,neighborhood_generator_function_cycle, iterations=1000):
        """
        """
        def sanneal(seq,neigh,iters):
            _,old_min_switches=self.minimizeSwitchesForJobSequence(seq)
            new_seq = self.bestJobSequenceSimulatedAnnealing(seq,neigh,iters)
            _,new_min_switches=self.minimizeSwitchesForJobSequence(new_seq)

            return new_seq,new_min_switches,(new_min_switches<old_min_switches)

        improved = True
        while improved:
            #print("")
            self.onIterate()
            for i in range(len(neighborhood_generator_function_cycle)):
                self.onEvaluateNeighborhood()
                neighborhood_generator_function=neighborhood_generator_function_cycle[i]
                #print(" "*i + neighborhood_generator_function.__name__ )

                neighborhood=neighborhood_generator_function(job_sequence)
                job_sequence,min_switches,improved=sanneal(job_sequence,neighborhood_generator_function,iterations)

                if improved:
                    #print(" " *i + "New minimum: %d @ %d evals"%(min_switches,self.stats["total_objective_evaluations"]))
                    self.onNewBest(min_switches)
                    break
                else:
                    #Swap the neighborhood to the end of the cycle, because it may not perform well in the current function area
                    pass

        return job_sequence,min_switches

    ########################################################################################################################
    #
    # Utility, Output, Statistics and Debugging
    #
    ########################################################################################################################
    def loadFromFile(self,filename): #TODO: File format sanity checks.
        """Expects data format according to http://www.unet.edu.ve/~jedgar/ToSP/ToSP.htm"""
        lines=open(filename,"r").readlines()

        self.n=len(lines)
        self.m=-1

        jobs={}

        for line in lines:
            parts=line.replace("#","").split(":")
            j=int(parts[0]) #Extract job index
            jobs[j]=set([int(i) for i in parts[1].split(",")]) #Extract tools
            self.m=max(self.m,max(jobs[j])+1) #Update problem tool count

        #Construct the incidence matrix.
        self.A=[[1 if t in jobs[j] else 0 for t in range(self.m)] for j in range(self.n)]

    def setLogFile(self,filename):
        self.log = open(filename,"w")
        self.writeLog("iteration,objective_value")

    def closeLogFile(self):
        self.log.close()

    def writeLog(self,msg):
        if self.log != None:
            self.log.write(msg+"\n")

        if self.console_output:
            print(msg)

    def onEvaluateObjectiveFunction(self):
        self.stats["total_objective_evaluations"]+=1
        self.stats["objective_evaluations_since_last_improvement"]+=1

    def onEvaluateNeighborhood(self):
        self.stats["total_neighborhoods"]+=1

    def onIterate(self):
        self.stats["total_iterations"]+=1

    def onNewBest(self,objective_value):
        if objective_value < self.stats["best_objective_value"] or self.stats["best_objective_value"]==-1:
            self.stats["total_improvements"]+=1
            self.stats["objective_evaluations_since_last_improvement"]=0
            self.stats["best_objective_value"]=objective_value
            self.writeLog("%d,%d"%(self.stats["total_iterations"],objective_value))

    def writeReport(self,best_job_sequence):
        tool_sequence, min_switches = self.minimizeSwitchesForJobSequence(best_job_sequence)

        self.writeLog("=======================================================================================================")

        self.debugVerifySolution(best_job_sequence,tool_sequence)
        self.writeLog("Solution passed logic verification.")
        self.writeLog("")

        self.writeLog("Tool sequence:")


        st=0
        for i,tool_set in enumerate(tool_sequence):
            swaps=0
            if i>=1:
                swaps=len(tool_set)-len(tool_set.intersection(tool_sequence[i-1]))
                st+=swaps
            self.writeLog("(%2d switches) [%s]" % (swaps,",".join(["%2d"%i for i in sorted([i for i in tool_set])]),))

        self.writeLog("")
        self.writeLog("Job sequence:\n[%s]" % ",".join([str(i) for i in best_job_sequence]))

        self.writeLog("=======================================================================================================")
        self.writeLog("Objective value:              %d"%min_switches)
        self.writeLog("Total objective evaluations:  %d"%(self.stats["total_objective_evaluations"]-1))
        self.writeLog("Wasted objective evaluations: %d"%(self.stats["objective_evaluations_since_last_improvement"]-1))
        self.writeLog("Total iterations:             %d"%self.stats["total_iterations"])
        self.writeLog("Total neighborhoods:          %d"%self.stats["total_neighborhoods"])
        self.writeLog("Total improvements:           %d"%self.stats["total_improvements"])

    def debugBestSequenceExhaustive(self):
        start=self.constructJobSequenceLinear()
        best_sequence=start
        _,best_switches=self.minimizeSwitchesForJobSequence(start)
        for permutation in itertools.permutations(start):
            _,switches=self.minimizeSwitchesForJobSequence(permutation)
            if switches<best_switches:
                best_switches=switches
                best_sequence=permutation

        return best_sequence


    def debugVerifyJobSequence(self,job_sequence):
        if set(job_sequence) != set([i for i in range(self.n)]) or len(job_sequence) != self.n:
            raise

    def debugVerifySolution(self,job_sequence,tool_sequence):
        i=0

        self.debugVerifyJobSequence(job_sequence)

        for j in job_sequence:
            needed=self.getToolsForJob(j)
            for t in needed:
                if not t in tool_sequence[i]:
                    raise
            if len(tool_sequence[i])>self.C:
                raise
            i+=1

########################################################################################################################
#
# Experimental: Genetic Algorithm
#
########################################################################################################################
class Individual:
    def __init__(self,seq):
        self.sequence=seq

    sequence=[]

    objective_value=0
    distance_sum=0

    objective_score=0
    distance_score=0


    def distance(self,other):
        dist=0
        for i in range(len(self.sequence)):
            if self.sequence[i]!=other.sequence[i]:
                dist+=1
        return dist

class HeuristicSolverGenetic(HeuristicSolver):
    population=[]
    population_size_max = 300
    mutation_rate = 0.15
    reproduction_rate = 0.9

    def seed(self,n):
        for i in range(n):
            self.population.append(Individual(self.constructJobSequenceRandom()))

    def iterate(self):
        self.computeScores()
        print("%d best score in %d"%(self.getBest().objective_value,len(self.population)))
        print("%f avg. objective"%(sum([i.objective_value for i in self.population])/float(len(self.population))))

        reproduction_distribution = self.buildReproductionDistribution()
        offspring = []

        for i in range(int(len(self.population)*self.reproduction_rate)):
            a = self.select(reproduction_distribution)
            b=a

            while a==b:
                b = self.select(reproduction_distribution)

            offspring.append(Individual(self.recombine(self.population[a].sequence,self.population[b].sequence)))

        print("%d offspring"%len(offspring))

        if len(self.population)+len(offspring) > self.population_size_max:
            death_distribution = self.buildDeathDistribution()

            must_die = len(self.population)+len(offspring) - self.population_size_max
            dying = set()
            while len(dying) < must_die:
                dying.add(self.select(death_distribution))

            new_population = []
            for i, individual in enumerate(self.population):
                if not i in dying:
                    new_population.append(individual)

            print("%d died"%(len(self.population)-len(new_population)))
            self.population = new_population

        self.population.extend(offspring)

        for i in range(int(len(self.population)*self.mutation_rate)):
            index = random.randrange(0,len(self.population)-1)
            self.population[index].sequence = self.mutate(self.population[index].sequence)

        print("%d mutations"%(int(len(self.population)*self.mutation_rate)))
        print("")

    def getBest(self):
        best_individual=self.population[0]
        best_objective=best_individual.objective_value

        for individual in self.population:
            if individual.objective_value < best_objective:
                best_objective = individual.objective_value
                best_individual = individual

        return best_individual

    def fastForward(self):
        #Simulated annealing for all
        pass

    def computeScores(self):
        total_objective_sum = 0
        total_distance_sum = 0
        for individual in self.population:
            _,individual.objective_value=self.minimizeSwitchesForJobSequence(individual.sequence)
            individual.distance_sum=0

            for other in self.population:
                individual.distance_sum+=individual.distance(other)

            total_objective_sum+=individual.objective_value
            total_distance_sum+=individual.distance_sum

        avg_objective = total_objective_sum / float(len(self.population))
        avg_distance_sum = total_distance_sum / float(len(self.population))

        for individual in self.population:
            individual.objective_score = 1 / ( float(individual.objective_value**2) / avg_objective**2)
            individual.distance_score = float(individual.distance_sum) / avg_distance_sum
            individual.score = (3*individual.objective_score+individual.distance_score)/4.0
            individual.p_reproduce = (1.0/len(self.population)) * individual.score
            individual.p_death = (1.0/len(self.population)) * (1.0/individual.score)

            #print(individual.objective_value,individual.score)

    def buildReproductionDistribution(self):
        distribution=[]

        value=0
        for i,individual in enumerate(self.population):
            distribution.append( (value,value+individual.p_reproduce,i) )
            value+=individual.p_reproduce

        return distribution

    def buildDeathDistribution(self):
        distribution=[]

        value=0
        for i,individual in enumerate(self.population):
            distribution.append( (value,value+individual.p_death,i) )
            value+=individual.p_death

        return distribution

    def select(self,distribution):
        r = random.uniform(0,1)

        for start,end,index in distribution:
            if r>=start and r<end:
                return index

        return random.randrange(0,len(self.population)-1)

    def recombine(self,sequence_a,sequence_b,initial_interchange_size=None):
        #Find largest pair of portions that can be interchanged
        if initial_interchange_size==None:
            interchange_size = len(sequence_a)/2
        else:
            interchange_size = initial_interchange_size
        found=False

        while not found and interchange_size > 2:
            interchange_size -= 1
            for i in range(0,len(sequence_a)-interchange_size):
                for j in range(0,len(sequence_b)-interchange_size):
                    set_a=set(sequence_a[i:i+interchange_size])
                    set_b=set(sequence_b[j:j+interchange_size])

                    if set_a==set_b:
                        found=True
                        interchange_locus_a=i
                        interchange_locus_b=j
                        break

                if found:
                    break

        if not found:
            return sequence_a

        new_sequence=copy.deepcopy(sequence_a)
        new_sequence[interchange_locus_a:interchange_locus_a+interchange_size]=copy.deepcopy(sequence_b[interchange_locus_b:interchange_locus_b+interchange_size])

        if interchange_size > 3:
            return self.recombine(new_sequence,sequence_b,interchange_size-1)
        else:
            return new_sequence

    def mutate(self,sequence):
        for new_sequence in singleRandomSwapNeighborhood(sequence):
            return new_sequence
