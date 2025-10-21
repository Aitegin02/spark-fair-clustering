LAB 1 G30HW1
Homework 1

The purpose of the first homework is to get acquainted with Spark and with its use to implement MapReduce algorithms. In preparation for the homework, you must perform and test the set up of your machine, following the instructions given in this site. The homework concerns a variant of the classical Lloyd's algorithm for k-means clustering, which enforces a fairness constraint on the solution based on extra demographic information attached to the input point
"""

#Libraries import

import sys
from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans
from pyspark.mllib.linalg import Vectors

#Inputs processing

if len(sys.argv) != 5:
    print("Usage: G30HW1.py <input_path> <L> <K> <M>")
    sys.exit(1)

input_path = sys.argv[1]
L = int(sys.argv[2]) #number of partitions
K = int(sys.argv[3]) #number of clusters
M = int(sys.argv[4]) #number of iterations

#Read using Spark and format test file, save in RDD

#Get a proper input format
def parse_line(line):
  parts = line.strip().split(",")
  point = tuple(map(float, parts[:-1]))
  group = parts[-1]
  return (point, group)

#Read file
sc = SparkContext(appName="G30HW1")
data = sc.textFile(input_path,L)
inputPoints = data.map(parse_line).cache()

#Counting points
N = inputPoints.count()
NA = inputPoints.filter(lambda x: x[1] == "A").count()
NB = N - NA

print(f"N:{N}")
print(f"NA:{NA}")
print(f"NB:{NB}")

"""1. We need to leave RDD without categories
2. Launch KMeans and get centroids and print them
"""

#Kmeans and search of centroids (formulas)

#consider only coordinates
onlyPoints =inputPoints.map(lambda x: Vectors.dense(x[0]))

#Kmeans
model = KMeans.train(onlyPoints, k=K, maxIterations = M)

#centroids
centroids = [tuple(c) for c in model.clusterCenters]

print ("centroids:")
for i,c in enumerate(centroids):
  print(f"Centroid {i}: {c}")

"""1. function MRComputeStandardObjective
2. function MRComputeFairObjective
"""

#Standard and Fair functions computing

#Standard function
def MRComputeStandardObjective(rdd, centroids):
  def closestCentroid(point):
    return min(
        sum ((p - c) ** 2 for p, c in zip(point, centroid)) for centroid in centroids
      )
  sum_distance = rdd.map(lambda x: closestCentroid(x[0])).sum()
  count = rdd.count()
  return sum_distance / count

#Fair function
def MRComputerFairObjective(rdd, centroids):
    def closestCentroid(point):
      return min(
        sum ((p - c) ** 2 for p, c in zip(point, centroid)) for centroid in centroids
        )
    def avg_distance(label):
      group_rdd = rdd.filter (lambda x: x[1] == label)
      count = group_rdd.count()
      if count == 0: return 0
      total_distance = group_rdd.map(lambda x: closestCentroid(x[0])).sum()
      return total_distance / count

    avg_distance_A = avg_distance("A")
    avg_distance_B = avg_distance("B")
    return max(avg_distance_A,avg_distance_B)

standard_obj = MRComputeStandardObjective(inputPoints, centroids)
fair_obj = MRComputerFairObjective(inputPoints, centroids)

"""1. Compute MRPrintStatistics"""

def MRPrintStatistics(rdd, centroids):
  def cluster(point):
    closest_distance = [(i, sum((p - c) ** 2 for p,c in zip (point, centroids[i]))) for i in range(len(centroids))]
    return min(closest_distance, key=lambda x: x[1])[0]

  #to find which group to which cluster belongs
  cluster_to_point = rdd.map(lambda x: ((cluster(x[0]), x[1]), 1))
  #the number of points in each group in each cluser
  counts = cluster_to_point.reduceByKey(lambda x,y: x + y)

  #gather in the dictionary in the following format
  stats = {}
  for ((cluster_idx, group), count) in counts.collect():
    if cluster_idx not in stats:
      stats[cluster_idx] = {'A': 0, 'B': 0}
    stats[cluster_idx][group] = count
  return stats

  for i in range(len(centroids)):
    NAi = stats.get(i, {}).get('A', 0)
    NBi = stats.get(i, {}).get('B', 0)
    print (f"{centroids[i]}, {NAi}, {NBi}")

stats = MRPrintStatistics(inputPoints, centroids)
print(f"======= EXAMPLE OF OUTPUT FOR L = {L}, K = {K}, M = {M} =======\n")
print(f"Input file = {input_path}, L = {L}, K = {K}, M = {M}")
print(f"N = {N}, NA = {NA}, NB = {NB}")
print(f"Delta(U,C) = {standard_obj:.6f}")
print(f"Phi(A,B,C) = {fair_obj:.6f}")

for i, c in enumerate(centroids):
    NAi = stats.get(i, {}).get('A', 0)
    NBi = stats.get(i, {}).get('B', 0)
    print(f"i = {i}, center = ({c[0]:.6f},{c[1]:.6f}), NA{i} = {NAi}, NB{i} = {NBi}")
