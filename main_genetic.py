########################################################################################################################
#
# This script demonstrates the usage of tosp.py to solve a problem and print the solution
#
########################################################################################################################

import tosp

solver = tosp.HeuristicSolverGenetic()

print("Loading problem: matrix_40j_30to_NSS_0, C=15;")
solver.loadFromFile("datasets/matrix_40j_30to_NSS_0.txt")
solver.setC(15)

solver.seed(300)

for i in range(5000):
    solver.iterate()
    solver.writeReport(solver.getBest())
