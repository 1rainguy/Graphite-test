import subprocess

# 1️⃣ Write a small TSP instance in TSPLIB format
with open("example.tsp", "w") as f:
    f.write("NAME: example\n")
    f.write("TYPE: TSP\n")
    f.write("DIMENSION: 5\n")
    f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
    f.write("NODE_COORD_SECTION\n")
    f.write("1 0 0\n")
    f.write("2 1 5\n")
    f.write("3 5 2\n")
    f.write("4 6 6\n")
    f.write("5 8 3\n")
    f.write("EOF\n")

# 2️⃣ Write LKH parameter file
with open("example.par", "w") as f:
    f.write("PROBLEM_FILE = example.tsp\n")
    f.write("OUTPUT_TOUR_FILE = example.tour\n")

# 3️⃣ Call the LKH executable
subprocess.run(["LKH", "example.par"])
