
**Assignment of Homework 1 (deadline: April 11)**

The purpose of the first homework is to get acquainted with Spark and with its use to implement MapReduce algorithms. In preparation for the homework, you must perform and test the set up of your machine, following the instructions given in this site. The homework concerns a variant of the classical Lloyd's algorithm for k-means clustering, which enforces a fairness constraint on the solution based on extra demographic information attached to the input points.  

K-MEANS CLUSTERING. Given a set of points 𝑈⊂ℝ𝐷
 and an integer 𝐾, k-means clustering aims at determining a set 𝐶⊂ℝ𝐷
 of 𝐾
 centroids which minimize the following objective function:
 
Δ(𝑈,𝐶)=(1/|𝑈|)∑𝑢∈𝑈(dist(𝑢,𝐶))2


where (dist(𝑢,𝐶))2
 denotes the squared Euclidean distance between 𝑢
 and the centroid of 𝐶
 closest to 𝑢
. Since the problem is NP-hard, current efficient solutions seek approximate solutions. Lloyd's algorithm is widely used to this purpose. It works as follows. Let 𝑀
 be an integer parameter:

Compute an initial set 𝐶={𝑐1,𝑐2,…,𝑐𝐾}
 of centroids
Repeat M times
2.1. Partition 𝑈
 into 𝐾
 clusters 𝑈1,𝑈2,⋯,𝑈𝐾
, where 𝑈𝑖
 consists of the points of 𝑈
 whose closest centroid is 𝑐𝑖
  (assume that ties are broken in favor of the smallest index).
2.2. For 1≤𝑖≤𝐾
, compute a new centroid 𝑐𝑖
 as the average of the points of 𝑈𝑖

Note that, rather than using a user-defined parameter 𝑀
 to specify the number of iterations, suitable convergence conditions are often employed to decide when to stop the iterations.

SPARK IMPLEMENTATION of Lloyd's ALGORITHM. In the RDD-based API of the mllib package, Spark provides an implementation of LLoyd's algorithm for k-means clustering. In particular, the algorithm is implemented by method train of class KMeans which receives in input the points stored as an RDD of Vector in Java, and of NumPy arrays in Python, the number k of clusters, and the number of iterations. The method computes an initial set of centroids using, as a default, algorithm kmeans|| (a parallel variant of kmeans++), and then executes the specified number of iterations. As output the method returns the final set of centroids, represented as an instance of class KMeansModel. For this latter class, method clusterCenters will return the centroids as an array of Vector in Java and list of NumPy arrays in Python. Refer to the official Spark documentation on clustering (RDD-based API) for more details.
FAIR K-MEANS CLUSTERING. A fair variant of Lloyd's algorithm was proposed in the paper Socially Fair k-Means Clustering (ACM FAccT'21), by M. Ghadiri, S. Samadi, and S. Vempala. Suppose that the input set of points 𝑈
 is split into two demographic groups 𝐴,𝐵
 (e.g., corresponding to female and male individuals). That is 𝑈=𝐴∪𝐵
. The new algorithm addresses a socially fair version of k-means where the set 𝐶
 of centroids aims at minimizing the following objective function:

Φ(𝐴,𝐵,𝐶)=max{(1/|𝐴|)∑𝑎∈𝐴(dist(𝑎,𝐶))2,(1/|𝐵|)∑𝑏∈𝐵(dist(𝑏,𝐶))2}
   

The new algorithm is a simple variation of Lloyd's algorithm. The only difference concerns Step 2.2, where the computation of the new centroids 𝑐1,𝑐2,…,𝑐𝐾
 from the current partition 𝑈1,𝑈2,…,𝑈𝐾
 is performed through a gradient descent protocol. You will implement this fair variant of Lloyd's algorithm in Homework 2, while in this homework you will be only concerned with the new objective function. 

REPRESENTATION OF POINTS

(JAVA Users) Points must be represented as instances of the class Vector of package mllib.linalg and can be manipulated through static methods offered by the class Vectors in the same package. For example, method Vectors.dense(x) transforms an array x of double into an instance of class Vector, while method Vectors.sqdist(x,y) computes the squared Euclidean distance between two instances, x and y, of class Vector. Details on these classes can be found in the Spark Java API. 
Warning. Make sure to use the classes from the org.apache.spark.mllib package. There are classes with the same name in org.apache.spark.ml package which are functionally equivalent, but incompatible with those of the org.apache.spark.mllib package.

(PYTHON Users) Points must be represented as tuples of float (i.e., point = (x1, x2, ...)). Although Spark provides the class Vector also for Python (see pyspark.mllib package), its performance is very poor and its more convenient to use tuples, especially for points from low-dimensional spaces.
INPUT FORMAT. To implement the algorithms assume that the input set 𝑈
 is given in input as a file, where each row contains one point stored with the coordinates separated by comma (',') and, at then end, a character, A or B, representing the point's demographic group. For instance, a point 𝑝
 = (1.5,6.0) ∈ℝ2
  of group A, will occur in the file as line: 1.5,6.0,A.

ASSIGNMENT for HW1. For a given input set of points 𝑈
, split in two demographic groups 𝐴,𝐵
, you must compare the values of the two objective functions Δ(𝑈,𝐶)
 and Φ(𝐴,𝐵,𝐶)
, where the set 𝐶
 of centroids is computed with the standard Lloyd's algorithm, ignoring the demographic group attributes. Also, you will print some statistics concerning the distribution of the two groups among the clusters. You must assume that the input set 𝑈
 is potentially very large, therefore it must be stored into an RDD, whose elements are pairs (𝑝,𝑔𝑝)
, where 𝑝∈ℝ𝐷
 is a point and 𝑔𝑝∈{𝐴,𝐵}
 is a character representing its demographic group.

Specifically, for the homework you must do the following tasks.

1) Write a method/function MRComputeStandardObjective that takes in input the set 𝑈=𝐴∪𝐵
 and a set 𝐶
 of centroids, and returns the value of the objective function Δ(𝑈,𝐶)
 described above, thus ignoring the demographic groups.

2) Write a method/function MRComputeFairObjective that takes in input the set 𝑈=𝐴∪𝐵
 and a set 𝐶
 of centroids, and returns the value of the objective function Φ(𝐴,𝐵,𝐶)
 described above.

3) Write a method/function MRPrintStatistics that takes in input the set 𝑈=𝐴∪𝐵
 and a set 𝐶
 of centroids, and computes and prints the triplets (𝑐𝑖,𝑁𝐴𝑖,𝑁𝐵𝑖)
, for 1≤𝑖≤𝐾=|𝐶|
, where 𝑐𝑖
 is the 𝑖
-th centroid in 𝐶
, and 𝑁𝐴𝑖,𝑁𝐵𝑖
 are the numbers of points of 𝐴
 and 𝐵
, respectively, in the cluster 𝑈𝑖
 centered in 𝑐𝑖
.

4) Write a program GxxHW1.java (for Java users) or GxxHW1.py (for Python users), where xx is your 2-digit group number (e.g., 04 or 25), which receives in input, as command-line arguments, a path to the file storing the input points, and 3 integers 𝐿,𝐾,𝑀
, and does the following:

Prints the command-line arguments and stores  𝐿,𝐾,𝑀
 into suitable variables. 
Reads the input points into an RDD -called inputPoints-, subdivided into 𝐿
 partitions.
Prints the number 𝑁
 of points, the number 𝑁𝐴
 of points of group A, and the number 𝑁𝐵
 of points of group B (hence, 𝑁=𝑁𝐴+𝑁𝐵
). 
Computes a set 𝐶
 of 𝐾
 centroids by using the Spark implementation of the standard Lloyd's algorithm for the input points, disregarding the points' demographic groups, and using 𝑀
 as number of iterations. 
Prints the values of the two objective functions Δ(𝑈,𝐶)
 and Φ(𝐴,𝐵,𝐶)
, computed by running  MRComputeStandardObjective and MRComputeFairObjective, respectively.
Runs MRPrintStatistics.
OUTPUT FORMAT: In the same section as this page you will find some datasets and outputs corresponding to specific input configurations using these datasets. Your program must STRICTLY ADHERE TO THE OUTPUT FORMAT used in these examples. Any deviation from this format will be penalized.

5) Test your program using the examples provided in the same section as this page.

6) Prepare a 1-page word file named GxxHW1analysis.docx, which provides a bound to the local space indicator M_L of the MapReduce setting, required by the method/function MRPrintStatistics. The bound must be justified explicitly encompassing every transformation used by the method/function.

IMPORTANT WARNING. The input set 𝑈
 as well as any intermediate data of size similar to |𝑈|
 can be store exclusively in distributed form, using RDDs. Otherwise, the evaluation will be penalized. One the other hand, you can safely assume that 𝐾
 is small enough that 𝐾
 points, or, in general, collections of size 𝐾
, can be stored into local structures (e.g., arrays, lists, etc.)

SUBMISSION INSTRUCTIONS. Each group must submit a zip file containing its program (GxxHW1.java or GxxHW1.py) and the file GxxHW1analysis.docx. Please use a zip even for single and small files. Only one student per group must submit the file in Moodle Exam using the link provided in the Homework 1 section. Make sure that your code is free from compiling/run-time errors and that you use the file/variable names in the homework description, otherwise your score will be penalized.
