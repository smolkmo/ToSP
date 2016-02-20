########################################################################################################################
#
# This script demonstrates the usage of tosp.py to solve a problem and print the solution
#
########################################################################################################################

import tosp

solver = tosp.HeuristicSolver()

print("Loading problem: matrix_40j_30to_NSS_0, C=15;")
solver.loadFromFile("datasets/matrix_40j_30to_NSS_0.txt")
solver.setC(15)

print("Constructing intitial solution (greedy);")
job_sequence = solver.constructJobSequenceGreedy()
_, switches = solver.minimizeSwitchesForJobSequence(job_sequence)
print("\t %d switches (obj. f. value)"%switches)

print("Improving using simulated annealing on singleRandomSwapNeighborhood;")
print("iteration,new_best_objective")
job_sequence = solver.bestJobSequenceSimulatedAnnealing(job_sequence,tosp.singleRandomSwapNeighborhood)
_, switches = solver.minimizeSwitchesForJobSequence(job_sequence)

solver.writeReport(job_sequence)
