import sys
import time
import numpy as np
from pyspark import SparkContext, StorageLevel
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.mllib.linalg import Vectors
import numpy as _np


def parse_line(line):
    parts = line.strip().split(",")
    point = tuple(map(float, parts[:-1]))
    group = parts[-1]
    sum_sq = sum(p ** 2 for p in point)
    return (point, group, sum_sq)


def closest_centroid(point, centroids_np):
    point_np = np.array(point)
    distances = np.linalg.norm(centroids_np - point_np, axis=1)
    return np.argmin(distances)


def MRComputeFairObjectiveFast(rdd, centroids):
    C   = _np.array(centroids)            # (K,d)
    C2  = _np.sum(C * C, axis=1)          # (K,)
    bcC  = rdd.context.broadcast(C)
    bcC2 = rdd.context.broadcast(C2)
    def part_stats(iterator):
        C   = bcC.value
        C2  = bcC2.value
        SA = SB = 0.0
        nA = nB = 0
        block, labels = [], []
        for pt, grp, _ in iterator:
            block.append(pt); labels.append(grp)
            if len(block) == 4096:        # process chunk
                X  = _np.array(block)                     # (B,d)
                d2 = (_np.sum(X*X,1)[:,None] + C2 - 2*X.dot(C.T)).min(1)
                for dst, g in zip(d2, labels):
                    if g == 'A': SA += dst; nA += 1
                    else:        SB += dst; nB += 1
                block[:], labels[:] = [], []

        # leftovers
        if block:
            X  = _np.array(block)
            d2 = (_np.sum(X*X,1)[:,None] + C2 - 2*X.dot(C.T)).min(1)
            for dst, g in zip(d2, labels):
                if g == 'A': SA += dst; nA += 1
                else:        SB += dst; nB += 1
        yield (SA, nA, SB, nB)
    SA, nA, SB, nB = (
        rdd.mapPartitions(part_stats)
           .reduce(lambda a, b: (a[0]+b[0], a[1]+b[1],
                                 a[2]+b[2], a[3]+b[3]))
    )

    avgA = SA / nA if nA else 0.0
    avgB = SB / nB if nB else 0.0
    return max(avgA, avgB)


def computeVectorX(fixed_a, fixed_b, alpha, beta, ell, k):
    gamma = 0.5
    x_dist = [0.0] * k
    power = 0.5
    t_max = 10
    for _ in range(t_max):
        f_a, f_b = fixed_a, fixed_b
        power /= 2
        for i in range(k):
            denom = gamma * alpha[i] + (1 - gamma) * beta[i]
            temp = 0.0 if denom == 0 else (1 - gamma) * beta[i] * ell[i] / denom
            x_dist[i] = temp
            f_a += alpha[i] * temp * temp
            temp2 = ell[i] - temp
            f_b += beta[i] * temp2 * temp2
        if abs(f_a - f_b) < 1e-8:
            break
        gamma = gamma + power if f_a > f_b else gamma - power
    return x_dist


def MRFairLloyd(rdd, K, M):
    total_A, total_B = rdd.map(lambda x: (1, 0) if x[1] == 'A' else (0, 1)).reduce(lambda a, b: (a[0]+b[0], a[1]+b[1]))

    only_points = rdd.map(lambda x: Vectors.dense(x[0]))
    model = KMeans.train(only_points, k=K, maxIterations=0)
    centroids = [tuple(c) for c in model.clusterCenters]
    dim = len(centroids[0])

    for it in range(M):
        iter_start = time.time()
        bc = rdd.context.broadcast(np.array(centroids))

        def assign_partition(iterator):
            for point, group, sum_sq in iterator:
                cid = np.argmin(np.linalg.norm(bc.value - np.array(point), axis=1))
                yield (cid, (point, group, sum_sq))

        zero = (0, np.zeros(dim), 0.0, 0, np.zeros(dim), 0.0)

        def seq_op(acc, x):
            point, group, sum_sq = x
            if group == 'A':
                return (
                    acc[0] + 1, acc[1] + np.array(point), acc[2] + sum_sq,
                    acc[3], acc[4], acc[5]
                )
            else:
                return (
                    acc[0], acc[1], acc[2],
                    acc[3] + 1, acc[4] + np.array(point), acc[5] + sum_sq
                )

        def comb_op(a, b):
            return (
                a[0] + b[0], a[1] + b[1], a[2] + b[2],
                a[3] + b[3], a[4] + b[4], a[5] + b[5]
            )

        stats = rdd.mapPartitions(assign_partition).aggregateByKey(zero, seq_op, comb_op).collectAsMap()

        new_centroids = []
        total_shift = 0.0
        for i in range(K):
            cntA, sumA, sqA, cntB, sumB, sqB = stats.get(i, zero)
            muA = sumA / cntA if cntA > 0 else None
            muB = sumB / cntB if cntB > 0 else None

            if muA is not None and muB is not None:
                ell = np.linalg.norm(muA - muB)
                if ell == 0:
                    new_centroids.append(tuple(muA))
                    continue
                fixedA = sqA - cntA * np.sum(muA ** 2)
                fixedB = sqB - cntB * np.sum(muB ** 2)
                alpha = [cntA / total_A]
                beta = [cntB / total_B]
                x = computeVectorX(fixedA / total_A, fixedB / total_B, alpha, beta, [ell], 1)[0]
                c_i = (1 - x / ell) * muA + (x / ell) * muB
                total_shift += np.linalg.norm(c_i - bc.value[i])
                new_centroids.append(tuple(c_i))
            elif muA is not None:
                new_centroids.append(tuple(muA))
            elif muB is not None:
                new_centroids.append(tuple(muB))
            else:
                new_centroids.append(centroids[i])

        centroids = new_centroids
        #print(f"Iteration {it+1}/{M} took {int((time.time() - iter_start)*1000)} ms, shift = {total_shift:.4f}")
        if total_shift < 1e-4:
            break

    return centroids

if __name__ == '__main__':
    if len(sys.argv) != 5:
        sys.exit(1)

    path, L, K, M = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    sc = SparkContext(appName="G30HW2")

    # load and cache
    raw = sc.textFile(path, L).cache()
    N  = raw.count()
    NA = raw.filter(lambda x: x.split(',')[-1] == 'A').count()
    NB = N - NA

    # parse
    parsed_rdd = raw.map(parse_line).persist(StorageLevel.MEMORY_ONLY)
    parsed_rdd.count()                       # materialise

    #Dense-vector
    points_rdd = parsed_rdd.map(lambda x: Vectors.dense(x[0]))

    # standard Lloyd
    t0 = time.time()
    std_model     = KMeans.train(points_rdd, k=K, maxIterations=M)
    std_centroids = [tuple(c) for c in std_model.clusterCenters]
    t1 = time.time()

    #fair Lloyd (your gradient step)
    t2 = time.time()
    fair_centroids = MRFairLloyd(parsed_rdd, K, M)
    t3 = time.time()

    t4 = time.time()
    obj_std  = MRComputeFairObjectiveFast(parsed_rdd, std_centroids)
    t5 = time.time()

    t6 = time.time()
    obj_fair = MRComputeFairObjectiveFast(parsed_rdd, fair_centroids)
    t7 = time.time()

    print(f"Input file = {path}, L = {L}, K = {K}, M = {M}")
    print(f"N = {N}, NA = {NA}, NB = {NB}")
    print("Fair Objective with Standard Centers =", round(obj_std , 4))
    print("Fair Objective with Fair Centers =", round(obj_fair, 4))
    print("Time to compute standard centers =", int((t1 - t0)*1000), "ms")
    print("Time to compute fair centers =", int((t3 - t2)*1000), "ms")
    print("Time to compute objective with standard centers =", int((t5 - t4)*1000), "ms")
    print("Time to compute objective with fair centers =", int((t7 - t6)*1000), "ms")


    sc.stop()
