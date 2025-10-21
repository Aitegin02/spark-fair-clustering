from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
import sys, random, threading

# arguments
assert len(sys.argv) == 6, "USAGE: G30HW3.py portExp T D W K"
port = int(sys.argv[1])
T = int(sys.argv[2])
D = int(sys.argv[3])
W = int(sys.argv[4])
K = int(sys.argv[5])

# setup
random.seed(42)
P = 8191  # prime for hashing

# sketches and counts
true_counts = {}
cm = [[0] * W for _ in range(D)]
cs = [[0] * W for _ in range(D)]

# hash generators
def make_hashes(d, m):
    funcs = []
    for _ in range(d):
        a = random.randint(1, P-1)
        b = random.randint(0, P-1)
        funcs.append(lambda x, a=a, b=b: ((a*x + b) % P) % m)
    return funcs

def make_sketch_pair(d, w):
    hs, gs = [], []
    for _ in range(d):
        a1 = random.randint(1, P-1)
        b1 = random.randint(0, P-1)
        hs.append(lambda x, a=a1, b=b1: ((a*x + b) % P) % w)
        a2 = random.randint(1, P-1)
        b2 = random.randint(0, P-1)
        gs.append(lambda x, a=a2, b=b2: 1 if ((a*x + b) % P) % 2 == 0 else -1)
    return hs, gs

h = make_hashes(D, W)
hh, gg = make_sketch_pair(D, W)

# estimators
def est_cm(u): return min(cm[i][h[i](u)] for i in range(D))
def est_cs(u):
    vals = [gg[i](u) * cs[i][hh[i](u)] for i in range(D)]
    vals.sort()
    m = D//2
    return (vals[m-1]+vals[m])/2 if D%2==0 else vals[m]

# processing state
total = [0]
stop_event = threading.Event()

# RDD processing
def proc(rdd):
    if stop_event.is_set(): return
    pairs = rdd.map(lambda x: (int(x),1)).reduceByKey(lambda a,b: a+b).collect()
    batch = sum(f for _,f in pairs)
    total[0] += batch
    for u,f in pairs:
        true_counts[u] = true_counts.get(u,0) + f
        for i in range(D):
            cm[i][h[i](u)] += f
            cs[i][hh[i](u)] += gg[i](u)*f
    if total[0] >= T:
        stop_event.set()

# Main
conf = SparkConf().setMaster("local[*]").setAppName("G30HW3")
# disable WAL on Windows to avoid native-hadoop errors
conf = conf.set("spark.streaming.receiver.writeAheadLog.enable","false")
sc = SparkContext(conf=conf)
ssc = StreamingContext(sc, 0.01)
sc.setLogLevel("ERROR")


dstream = ssc.socketTextStream("algo.dei.unipd.it", port)
dstream.foreachRDD(lambda t,r: proc(r))
ssc.start()
stop_event.wait()
ssc.stop(False,False)


# compute heavy hitters
freqs = sorted(true_counts.items(), key=lambda x:-x[1])
th = freqs[K-1][1] if len(freqs)>=K else freqs[-1][1]
hits = [(u,f) for u,f in freqs if f>=th]
er_cm = [abs(f-est_cm(u))/f for u,f in hits]
er_cs = [abs(f-est_cs(u))/f for u,f in hits]
avg_cm = sum(er_cm)/len(er_cm) if er_cm else 0
avg_cs = sum(er_cs)/len(er_cs) if er_cs else 0

#output
print(f"Port = {port} T = {T} D = {D} W = {W} K = {K}")
print(f"Number of processed items = {total[0]}")
print(f"Number of distinct items = {len(true_counts)}")
print(f"Number of Top-K Heavy Hitters = {len(hits)}")
print(f"Avg Relative Error for Top-K Heavy Hitters with CM = {avg_cm}")
print(f"Avg Relative Error for Top-K Heavy Hitters with CS = {avg_cs:.15E}")
if K<=10:
    print("Top-K Heavy Hitters:")
    for u,f in sorted(hits[:K], key=lambda x:x[0]):
        print(f"Item {u} True Frequency = {f} Estimated Frequency with CM = {est_cm(u)}")
