########################################################################################################################
#
# Exercise 4
#
# This file is concerned with benchmarking the implemented heuristics and parameters, which are implemented
# in the file "tosp.py"
#
########################################################################################################################

import tosp
import time

def constructGreedy(solver):
    return solver.constructJobSequenceGreedy()

def constructRandom(solver):
    return solver.constructJobSequenceRandom()

def constructGreedyClusteringBest(solver):
    seqs = solver.constructJobSequencesGreedyJobClustering()
    best_seq=seqs[0]
    _,best_switches = p.minimizeSwitchesForJobSequence(seqs[0])
    for i in range(1,len(seqs)):
        _,switches = p.minimizeSwitchesForJobSequence(seqs[i])
        if switches<best_switches:
            best_seq=seqs[i]
            best_switches=switches

    return best_seq

def constructGreedyRandomized(solver):
    return solver.constructJobSequenceGreedyRandomized()

def doNotImprove(solver,job_sequence):
    return job_sequence

def benchmarkLocal(solver,job_sequence,neighborhood_generator_function):
    best_sequence,_=solver.bestJobSequenceLocalSearch(job_sequence,neighborhood_generator_function)
    return best_sequence

def benchmarkGreedyClusteringAllVNS(solver):
    cycle=[tosp.singleRandomSwapNeighborhood,
           tosp.singleSliceSwapNeighborhood,
           tosp.singleSliceRandomizeNeighborhood,
           tosp.rotatingNeighborhood,
           tosp.singleNeighborSwapNeighborhood,
           tosp.singlePairSwapNeighborhood,
           tosp.dualNeighborSwapNeighborhood]

    return p.bestJobSequenceGreedyClusteringAllVNS(cycle)

def benchmarkVNS(solver,job_seq):
    _,switches=p.minimizeSwitchesForJobSequence(job_seq)

    cycle=[tosp.singleRandomSwapNeighborhood,
           tosp.singleSliceSwapNeighborhood,
           tosp.singleSliceRandomizeNeighborhood,
           tosp.rotatingNeighborhood,
           tosp.singleNeighborSwapNeighborhood,
           tosp.singlePairSwapNeighborhood,
           tosp.dualNeighborSwapNeighborhood]

    job_seq,switches=p.bestJobSequenceVariableNeighborhoodSearch(job_seq, cycle)
    return job_seq

def benchmarkVNSSimulatedAnnealing(solver,job_seq):
    _,switches=p.minimizeSwitchesForJobSequence(job_seq)

    cycle=[tosp.singleRandomSwapNeighborhood,
           tosp.singleSliceSwapNeighborhood,
           tosp.singleSliceRandomizeNeighborhood,
           tosp.rotatingNeighborhood,
           tosp.singleNeighborSwapNeighborhood,
           tosp.singlePairSwapNeighborhood,
           tosp.dualNeighborSwapNeighborhood]

    job_seq,switches=p.bestJobSequenceVariableNeighborhoodSimulatedAnnealing(job_seq, cycle)
    return job_seq

def benchmarkGRASP(solver):
    return p.bestJobSequenceGRASP()

def benchmarkGVNS(solver,job_sequence):
    cycle=[tosp.singleRandomSwapNeighborhood,
           tosp.singleSliceSwapNeighborhood,
           tosp.singleSliceRandomizeNeighborhood,
           tosp.rotatingNeighborhood,
           tosp.singleNeighborSwapNeighborhood]

    return p.bestJobSequenceGVNS(job_sequence,tosp.multiRandomSwapNeighborhood,cycle)

def benchmarkSimulatedAnnealing(solver,job_sequence,neighborhood_generator_function):
    return p.bestJobSequenceSimulatedAnnealing(job_sequence,neighborhood_generator_function)

def bestJobSequenceMultipleSimulatedAnnealing(solver):
    return p.bestJobSequenceMultipleSimulatedAnnealing(tosp.singleRandomSwapNeighborhood)

def benchmarkExhaustive(solver):
    return solver.debugBestSequenceExhaustive()

benchmarks=[([constructRandom,constructGreedy,constructGreedyClusteringBest,constructGreedyRandomized], doNotImprove, []),
            ([constructRandom,constructGreedy,constructGreedyClusteringBest], benchmarkLocal, [tosp.singleNeighborSwapNeighborhood]),
            ([constructRandom,constructGreedy,constructGreedyClusteringBest], benchmarkVNS, []),
            ([None], benchmarkGreedyClusteringAllVNS, []),
            ([None], benchmarkGRASP, []),
            ([constructGreedyClusteringBest], benchmarkGVNS, []),
            ([constructRandom,constructGreedy,constructGreedyClusteringBest], benchmarkSimulatedAnnealing, [tosp.singleRandomSwapNeighborhood]),
            ([constructRandom,constructGreedy,constructGreedyClusteringBest], benchmarkSimulatedAnnealing, [tosp.singleNeighborSwapNeighborhood])],
            ([constructGreedy], benchmarkVNSSimulatedAnnealing, [])]
            #([None],bestJobSequenceMultipleSimulatedAnnealing, [])]


datasets=[("matrix_10j_10to_NSS_0",4),
          ("matrix_40j_30to_NSS_0",15),
          ("matrix_40j_60to_NSS_0",20)]

repetitions = 1

improvements_by_neighborhood = {}

for filename,C in datasets:
    table=open("tables/"+filename+".csv","w")
    table.write("improvement_heuristic;construction_heuristic;neighborhood;objective;evals;wasted_evals;neighborhoods;iterations;wallclock\n")
    print(filename)

    for constructors, benchmark, parameters in benchmarks:
        for constructor in constructors:
            sum_objective=0
            sum_evaluations=0
            sum_wasted_evaluations=0
            sum_neighborhoods=0
            sum_iterations=0
            sum_wallclock=0

            benchmark_name = benchmark.__name__
            improvement_heuristic = benchmark_name
            construction_heuristic = ""
            if constructor!=None:
                benchmark_name += "_" + constructor.__name__
                construction_heuristic = constructor.__name__

            print("\t"+benchmark_name)

            for r in range(repetitions):
                title="%s_%s"%(benchmark_name,filename)

                p=tosp.HeuristicSolver()
                p.console_output=False
                p.setLogFile("reports/%s_%d.log"%(title,r))
                p.loadFromFile("datasets/"+filename+".txt")
                p.setC(C)

                begin=time.time()
                if constructor==None:
                    job_seq=benchmark(p)
                else:
                    job_seq=constructor(p)
                    job_seq=benchmark(p,job_seq,*parameters)
                end=time.time()

                _,best_objective = p.minimizeSwitchesForJobSequence(job_seq)
                p.stats["total_objective_evaluations"]-=1

                sum_objective+=best_objective
                sum_evaluations+=p.stats["total_objective_evaluations"]
                sum_wasted_evaluations+=p.stats["objective_evaluations_since_last_improvement"]
                sum_wallclock+=(end-begin)
                sum_neighborhoods+=p.stats["total_neighborhoods"]
                sum_iterations+=p.stats["total_iterations"]

                p.writeReport(job_seq)
                p.closeLogFile()

                for neighborhood, improvements in p.stats["improvements_by_neighborhood"].iteritems():
                    if not neighborhood in improvements_by_neighborhood:
                        improvements_by_neighborhood[neighborhood]=0
                    improvements_by_neighborhood[neighborhood]+=improvements

            neighborhood_table = open("tables/neighborhood_ranking.csv","w")
            neighborhood_table.write("neighborhood;improvements\n")
            for neighborhood in sorted(improvements_by_neighborhood,key=improvements_by_neighborhood.get):
                neighborhood_table.write("%s;%d\n"%(neighborhood,improvements_by_neighborhood[neighborhood]))
            neighborhood_table.close()

            neighborhood_parameter_text=""
            if len(parameters):
                neighborhood_parameter_text="_".join([param.__name__ for param in parameters])

            table.write("%s;%s;%s;%.3f;%.3f;%.3f;%d;%d;%.3f\n"%(improvement_heuristic,
                                                             construction_heuristic,
                                                             neighborhood_parameter_text,
                                                             sum_objective/float(repetitions),
                                                             sum_evaluations/float(repetitions),
                                                             sum_wasted_evaluations/float(repetitions),
                                                             sum_neighborhoods/float(repetitions),
                                                             sum_iterations/float(repetitions),
                                                             sum_wallclock/float(repetitions)))
            table.flush()

    table.close()
    print("")
