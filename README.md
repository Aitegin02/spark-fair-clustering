# Spark Fair Clustering

A project exploring **scalable and fairness clustering** using **Apache Spark** and **PySpark**.  
Includes implementations of **Fair K-Means** and **Count-Min Sketch streaming** 

### Fair K-Means
- Implemented a fairness-aware variant of Lloyd’s algorithm.
- Ensures balanced group representation across clusters.
- Optimized with **NumPy vectorization**, **broadcast variables**, and **RDD minimization**.
- Tested on datasets up to **10M samples**, achieving ~2× faster runtime.

### Streaming Analytics (Count-Min Sketch)
- Built a Spark Streaming pipeline to detect **top-K frequent elements** in unbounded data.
- Used **hash-based sketching** for memory-efficient approximate counting.

## Tech Stack
Python / PySpark / Apache Spark / NumPy / Matplotlib / CloudVeneto Cluster


