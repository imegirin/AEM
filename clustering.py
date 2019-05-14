import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from copy import deepcopy
import seaborn as sns
from random import shuffle
import random
from copy import deepcopy
from math import log, e
import numpy as np
import threading
import pickle
import time
import math
import os

class DistanceMatrix:

    def __init__(self, df, filename, recalculate=False):
        self.filename=filename
        self.matrix = self.distanceArray(df, recalculate)

    def distance(self, a, b):
        return math.sqrt(math.pow(a[0]-b[0], 2) + math.pow(a[1]-b[1], 2))

    def distanceArray(self, df, recalculate=False):
        if recalculate or not self.filename in os.listdir():
            tmp = np.zeros((df.shape[0], df.shape[0]))
            for i in range(df.shape[0]):
                for k in range(df.shape[0]):
                    #if i<k:
                    val = self.distance(tuple(df.iloc[i][['x','y']]),
                            tuple(df.iloc[k][['x','y']]))
                    tmp[i,k]= val
                    tmp[k,i] = val
            pickle.dump(tmp, open(self.filename, 'wb'))
            return tmp
        else:
            try:
                return pickle.load(open(self.filename, 'rb'))
            except:
                print('file does not exist')

    def getDist(self, el1, el2):
        return self.matrix[min(el1, el2),max(el1,el2)], el2             

    def distance_point_group(self, point, group):
        return np.sum(self.matrix[point,group])

    def findNearestNode(self, array, searchspace):
        tmpdict={}
        for item in array:
            tmpdict[item]= min([self.getDist(item, e) for e in np.setdiff1d(
                searchspace,[item])])

        return min(tmpdict.items(), key=lambda x: x[1])

class Clustering:
    def __init__(self, num_groups=10, filename='objects20_06.data', k=30):   
        self.df = pd.read_csv(filename, sep=' ', names=['x','y'], index_col=False)
        self.dm = DistanceMatrix(self.df, "matrix.p", recalculate=False)
        self.point_group_vector = np.full((num_groups, self.df.shape[0]), False)
        self.candidates = self.get_candidates(k)
        self.cache = {}

    def empty_point_group_vector(self):
        self.point_group_vector = np.full(self.point_group_vector.shape, False)
        self.cache= {}

    def MST(self, nodes):
        tree=[]
        freeset=nodes[:]
        size=0
        edges=[]

        if tree==[]:
            tree.append(freeset.pop())
        while freeset!=[]:
            intreepoint, (dist, newpoint) = self.dm.findNearestNode(tree, freeset)
            tree.append(newpoint)
            freeset.remove(newpoint)
            edges.append((intreepoint, newpoint))
            size+=dist

        return size, edges

    def distance_edges_in_group(self, groupID):
        points = np.where(self.point_group_vector[groupID])[0]
        agg = 0
        for p in points:
            for e in points:
                if p!=e:
                    agg+=self.dm.getDist(p,e)[0]
        num_edges = len(points)*(len(points)-1)/2.
        return agg/2., num_edges

    def mean_distance(self):
        sums, num_of_edges = np.sum([self.distance_edges_in_group(groupID) for groupID \
             in range(len(self.point_group_vector))], axis=0)
        return sums/num_of_edges

    def distance_to_group(self, node, groupID):
        group = np.where(self.point_group_vector[groupID]==True)[0]
        dist = self.dm.distance_point_group(node, group)
        return dist

    def show(self, data, trees):
        for tree, i in zip(trees,range(len(trees))):
            for point in tree:
                plt.plot([data.iloc[point[0]].x, data.iloc[point[1]].x],
                        [data.iloc[point[0]].y, data.iloc[point[1]].y],
                        color=f'C{i}',linewidth=3)
        plt.scatter(data['x'], data['y'])
        plt.show()
    
    def showClusters(self, data, clusters):
        for i, cluster in enumerate(clusters):
            plt.scatter(data.loc[cluster].x, data.loc[cluster].y, \
                c=np.hstack(np.random.rand(3,1)))
        plt.show()

    def initialize_clusters(self, method='greed'):
        def greed(n):
            freeset=[i for i in range(len(self.dm.matrix))]
            shuffle(freeset)
            clusters=[[freeset.pop()] for _ in range(n)]

            counter=0
            while freeset!=[]:
                _, (_,neighbor) = self.dm.findNearestNode(clusters[counter%n], freeset)
                neighbor = int(neighbor)
                alt = []
                for cluster in clusters:
                    t = [self.dm.getDist(neighbor, node)[0] for node in cluster]
                    alt.append(min(t))

                clusters[alt.index(min(alt))].append(neighbor)
                freeset.remove(neighbor)
                counter+=1
            
            return clusters

        def greed2(n):
            freeset=[i for i in range(len(self.dm.matrix))]
            shuffle(freeset)
            clusters=[[freeset.pop()] for _ in range(n)]

            while freeset!=[]:
                p = freeset.pop()
                tp = []
                for cluster in clusters:
                    tmp=min([self.dm.getDist(p, node)[0] for node in cluster])
                    tp.append(tmp)
                clusters[tp.index(min(tp))].append(p)

            return clusters

        def regret(n):
            freeset=[i for i in range(len(self.dm.matrix))]
            shuffle(freeset)
            clusters=[[freeset.pop()] for _ in range(n)]

            counter=0
            while len(freeset)>1:
                _, (_,neighbor) = self.dm.findNearestNode(clusters[counter%n], freeset)
                _, (_,alternative) = self.dm.findNearestNode(clusters[counter%n],
                        np.setdiff1d(freeset,neighbor))
                others_a = []
                others_n = []
                for cluster in clusters:
                    t = [self.dm.getDist(neighbor, node)[0] for node in cluster+[alternative]]
                    others_n.append(min(t))

                    t = [self.dm.getDist(alternative, node)[0] for node in cluster+[neighbor]]
                    others_a.append(min(t))

                if min(others_n)<min(others_a):
                    clusters[others_n.index(min(others_n))].append(neighbor)
                    freeset.remove(neighbor)
                else:
                    clusters[others_a.index(min(others_a))].append(alternative)
                    freeset.remove(alternative)
                counter+=1

            _, (_,neighbor) = self.dm.findNearestNode(clusters[counter%n], freeset)
            alt = []
            for cluster in clusters:
                t = [self.dm.getDist(neighbor, node)[0] for node in cluster]
                alt.append(min(t))

            clusters[alt.index(min(alt))].append(neighbor)
            freeset.remove(neighbor)
            return clusters

        def regret2(n):
            freeset=[i for i in range(len(self.dm.matrix))]
            shuffle(freeset)
            clusters=[[freeset.pop()] for _ in range(n)]

            while freeset!=[]:
                print()
                if len(freeset)>1:
                    p1 = freeset.pop()
                    p2 = freeset.pop()

                    t =[]
                    for cluster in clusters:
                        t.append(min([self.dm.getDist(p2, node) for node in cluster]))
                    p2t = t.index(min(t))
                    tmpcluster = clusters[p2t]+[p2]

                    t = []
                    for cluster in clusters:
                        t.append(min([self.dm.getDist(p1, node) for node in cluster]))
                    p1t1 = t.index(min(t))

                    t = []
                    for cluster in clusters+[tmpcluster]:
                        t.append(min([self.dm.getDist(p1, node) for node in cluster]))
                    p1t2 = t.index(min(t))

                    if p1t1==p1t2:
                        freeset.append(p2)
                        clusters[p1t1].append(p1)
                    else:
                        freeset.append(p1)
                        clusters[p2t].append(p2)
                else:
                    p1 = freeset.pop()
                    t = []
                    for cluster in clusters:
                        t.append(min([self.dm.getDist(p1, node) for node in cluster]))
                    p1t = t.index(min(t))
                    clusters[p1t].append(p1)

            return clusters

        def random(n):
            freeset = [i for i in range(len(self.dm.matrix))]
            shuffle(freeset)
            clusters = [[] for _ in range(n)]
            c = 0
            while freeset!=[]:
                    clusters[c%n].append(freeset.pop())     
                    c+=1
            return clusters  

        if method=='greed':
            clusters = greed(len(self.point_group_vector))
        elif method=='greed2':
            clusters = greed2(len(self.point_group_vector))
        elif method=='regret':
            clusters = regret(len(self.point_group_vector))
        elif method=='regret2':
            clusters = regret2(len(self.point_group_vector))
        elif method=='random':
            clusters = random(len(self.point_group_vector))

        for i, c in enumerate(clusters):
            self.point_group_vector[i][c]=True

        return clusters

    def add_to_group(self, node, groupID):
        self.point_group_vector[groupID][node] = True

    def delete_from_group(self, node, groupID):
        self.point_group_vector[groupID][node] = False

    def get_candidates(self, k):
        return [np.argpartition(self.dm.matrix[item], k)[1:k+1] for item in range(len(self.dm.matrix))]

    def get_candidate_groups(self, node):
        nearest_neighbors = self.candidates[node]
        return set([np.where(self.point_group_vector[:,element]==True)[0][0] for element in nearest_neighbors])

    def get_candidate_groups_list(self,node):
        nearest_neighbors = self.candidates[node]
        return list([np.where(self.point_group_vector[:,element]==True)[0][0] for element in nearest_neighbors])

    def entropy(self, labels, base=None):
        n_labels = len(labels)
        if n_labels <= 1:
            return 0
        _,counts = np.unique(labels, return_counts=True)
        probs = counts / n_labels
        n_classes = np.count_nonzero(probs)
        if n_classes <= 1:
            return 0

        ent = 0.
        base = e if base is None else base
        for i in probs:
            ent -= i * log(i, base)

        return ent

    def get_entropy_list(self):
        return [self.entropy(self.get_candidate_groups_list(i)) for i in range(len(self.dm.matrix))]

    def destroy(self, clusters, p):
        new_clusters = list(map(list, clusters))
        n = int(np.round(np.sum(self.point_group_vector)*p))
        freeset = []
        entropy_list = list(np.argsort(self.get_entropy_list()))

        for _ in range(n):
            item = entropy_list.pop()
            for i, c in enumerate(new_clusters):
                if item in c:
                    c.remove(item)
                    freeset.append(item)

        return new_clusters, freeset

    def repair(self, n, clusters, freeset):
        clusters = list(map(list, clusters))
        shuffle(freeset)

        counter=np.random.randint(n)
        while freeset!=[]:
            _, (_,neighbor) = self.dm.findNearestNode(clusters[counter%n], freeset)
            neighbor = int(neighbor)
            alt = []
            for cluster in clusters:
                t = [self.dm.getDist(neighbor, node)[0] for node in cluster]
                element = min(t) if t!=[] else 0.
                alt.append(element)

            idx = alt.index(min(alt))
            clusters[idx].append(neighbor)
            freeset.remove(neighbor)
            counter+=1
        
        return clusters

    def steep_local_search(self, candidates=True, caching=True):
        point_space = list(range(len(self.dm.matrix)))
        shuffle(point_space)

        best_sol = None
        for i in point_space:
            current_cluster_id = np.where(self.point_group_vector[:,i]==True)[0][0]
            current_cluster_dist = self.distance_to_group(i, current_cluster_id)

            if candidates:
                check_groups = self.get_candidate_groups(i)
            else:
                check_groups = range(len(self.point_group_vector))

            for another_id in [anID for anID in check_groups if anID != current_cluster_id]:
                another_cluster_dist = self.distance_to_group(i, another_id)

                if caching:
                    gain = self.cache.get((i, current_cluster_id, another_id))
                    if gain==None:
                        gain = current_cluster_dist - another_cluster_dist
                        self.cache[(i,current_cluster_id,another_id)]=gain
                else:
                    gain = current_cluster_dist - another_cluster_dist

                if best_sol==None and gain>0:
                    best_sol = (gain, i, current_cluster_id, another_id)
                if best_sol != None and gain > best_sol[0]:
                    best_sol = (gain, i, current_cluster_id, another_id)
        if best_sol!=None:   
            self.delete_from_group(best_sol[1], best_sol[2])
            self.add_to_group(best_sol[1], best_sol[3])
            self.cache={}
            return 200

    def greedy_local_search(self, candidates=True):
        point_space = list(range(len(self.dm.matrix)))
        shuffle(point_space)
        for i in point_space:
            current_cluster_id = np.where(self.point_group_vector[:,i]==True)[0][0]
            current_cluster_dist = self.distance_to_group(i, current_cluster_id)

            if candidates:
                check_groups = self.get_candidate_groups(i)
            else:
                check_groups = range(len(self.point_group_vector))

            for another_id in [anID for anID in check_groups if anID != current_cluster_id]:
                another_cluster_dist = self.distance_to_group(i, another_id)

                if current_cluster_dist>another_cluster_dist:
                    self.delete_from_group(i, current_cluster_id)
                    self.add_to_group(i, another_id)
                    return 200

        return None

    def local_search(self, clusters, function, candidates=True, caching=True):
        code_response = 200
        while code_response!=None:
            code_response = function(candidates, caching)

        clusters = [np.where(arr)[0] for arr in self.point_group_vector]
        return clusters

    def perturbation(self, clusters, n):
        for i in np.random.randint(len(clusters), size=n):
            clusters = list(map(list, clusters))
            if len(clusters[i]):
                v = clusters[i].pop(np.random.randint(len(clusters[i])))
                clusters[np.random.choice([j for j in range(len(clusters)) if j!=i])].append(v)
        return clusters

    def iterative_local_search(self, clusters, run_time, perturb_size=8):
        start = time.time()
        while time.time()-start<run_time:
            another_solution = deepcopy(self)
            another_solution.point_group_vector = self.point_group_vector.copy()
            another_solution.empty_point_group_vector()
            perturbated = self.perturbation(clusters, perturb_size)
            for i, c in enumerate(perturbated):
                another_solution.point_group_vector[i][c]=True

            another_clusters = another_solution.local_search(perturbated, another_solution.steep_local_search, candidates=True)
            if self.mean_distance()>another_solution.mean_distance():
                self.point_group_vector = another_solution.point_group_vector
                clusters = another_clusters

        return clusters

    def large_neighborhood_local_search(self, clusters, run_time, p=.3):
        start = time.time()
        while time.time()-start<run_time:
            another_solution = deepcopy(self)
            another_solution.point_group_vector = self.point_group_vector.copy()
            another_solution.empty_point_group_vector()
            
            new_clusters, freeset = self.destroy(clusters, p)
            new_clusters = self.repair(len(self.point_group_vector), new_clusters, freeset)
            for i, c in enumerate(new_clusters):
                another_solution.point_group_vector[i][c]=True

            another_clusters = another_solution.local_search(new_clusters, another_solution.steep_local_search, candidates=True)
            if self.mean_distance()>another_solution.mean_distance():
                self.point_group_vector = another_solution.point_group_vector
                clusters = another_clusters
        return clusters

def multiple_start_local_search(n=100):
    times = []
    scores = []
    #best_solution = None
    for _ in range(n):
        clustering = Clustering(20)
        clusters = clustering.initialize_clusters('random')
        start = time.time()
        clusters = clustering.local_search(clusters, clustering.steep_local_search, \
            candidates=True, caching=False)
        end = time.time()
        md = clustering.mean_distance()
        scores.append(md)
        times.append(end-start)
        print('time is',end-start)
        # if md<=np.min(scores):
        #     best_solution = clusters
    # self.empty_point_group_vector()
    # for i, c in enumerate(best_solution):
    #     self.point_group_vector[i][c]=True
    
    return np.min(scores)


# t = []
# for i in range(1):
#     alg = Clustering(20)
#     clusters = alg.initialize_clusters('random')
#     print(alg.mean_distance())
#     start = time.time()
#     alg.multiple_start_local_search(2)
#     end = time.time()
#     t.append(end-start)
# print(np.mean(t))
# print(alg.mean_distance())

def run():
    if False:
        times=[]
        scores=[]
        trees=[]
        last_best_score=999999
        for _ in range(100):
            start = time.time()
            clusters = dm.greed(10)
            end = time.time()
            times.append(end-start)

            outcome = [dm.MST(cluster) for cluster in clusters]
            score = sum([o[0] for o in outcome])
            scores.append(score)
            if score<last_best_score:
                last_best_score=score
                trees = [o[1] for o in outcome]

        print('time avg: {}, min: {}, max: {}'.format(np.mean(times), min(times), max(times)))
        print('score avg: {}, min: {}, max: {}'.format(np.mean(scores), min(scores), max(scores)))

        dm.show(df, trees)
    else:
        steep_best_score=99999
        steep_best_sol_greed = []
        steep_scores= []
        steep_time = []

        greed_best_score=99999
        greed_best_sol_greed=[]
        greed_scores=[]
        greed_time=[]
        for _ in range(5):
            dm=DistanceMatrix(df, 'matrix.p')

            #steepest
            clusters = dm.greed(20)
            clusters1 = deepcopy(clusters)
            start = time.time()
            clusters1 = dm.local_search(clusters,dm.steep_local_search)
            end = time.time()
            score = dm.meanDistForAllGroups(clusters1)
            steep_scores.append(score)
            steep_time.append(end-start)
            if score<steep_best_score:
                steep_best_score=score
                steep_best_sol_greed=clusters1

            dm=DistanceMatrix(df, 'matrix.p')

            #greedy
            clusters2 = deepcopy(clusters)
            start = time.time()
            clusters2 = dm.local_search(clusters2,dm.greedy_local_search)
            end = time.time()
            score = dm.meanDistForAllGroups(clusters2)
            greed_scores.append(score)
            greed_time.append(end-start)
            if score<greed_best_score:
                greed_best_score=score
                greed_best_sol_greed=clusters2
        
        print("steepest greed")
        print("mean score: {}   min score: {}   max score: {}".format(\
            np.mean(steep_scores), min(steep_scores), max(steep_scores)))
        print("mean time: {}   min time: {}   max time: {}".format(\
            np.mean(steep_time), min(steep_time), max(steep_time)))

        print("greedy greed")
        print("mean score: {}   min score: {}   max score: {}".format(\
            np.mean(greed_scores), min(greed_scores), max(greed_scores)))
        print("mean time: {}   min time: {}   max time: {}".format(\
            np.mean(greed_time), min(greed_time), max(greed_time)))


    #random initializer
        steep_best_score=99999
        steep_best_sol_random = []
        steep_scores= []
        steep_time = []

        greed_best_score=99999
        greed_best_sol_random=[]
        greed_scores=[]
        greed_time=[]
        for _ in range(1):
            dm=DistanceMatrix(df, 'matrix.p')

            #steepest
            clusters = dm.random(20)
            clusters1 = deepcopy(clusters)
            start = time.time()
            clusters1 = dm.local_search(clusters,dm.steep_local_search)
            end = time.time()
            score = dm.meanDistForAllGroups(clusters1)
            steep_scores.append(score)
            steep_time.append(end-start)
            if score<steep_best_score:
                steep_best_score=score
                steep_best_sol_random=clusters1
            dm.scorePerCluster={}

            dm=DistanceMatrix(df, 'matrix.p')

            #greedy
            clusters2 = deepcopy(clusters)
            start = time.time()
            clusters2 = dm.local_search(clusters2,dm.greedy_local_search)
            end = time.time()
            score = dm.meanDistForAllGroups(clusters2)
            greed_scores.append(score)
            greed_time.append(end-start)
            if score<greed_best_score:
                greed_best_score=score
                greed_best_sol_random=clusters2
            dm.scorePerCluster={}
        
        print("steepest random")
        print("mean score: {}   min score: {}   max score: {}".format(\
            np.mean(steep_scores), min(steep_scores), max(steep_scores)))
        print("mean time: {}   min time: {}   max time: {}".format(\
            np.mean(steep_time), min(steep_time), max(steep_time)))

        print("greedy random")
        print("mean score: {}   min score: {}   max score: {}".format(\
            np.mean(greed_scores), min(greed_scores), max(greed_scores)))
        print("mean time: {}   min time: {}   max time: {}".format(\
            np.mean(greed_time), min(greed_time), max(greed_time)))


        dm.showClusters(df, steep_best_sol_greed)
        dm.showClusters(df, greed_best_sol_greed)
        dm.showClusters(df, steep_best_sol_random)
        dm.showClusters(df, greed_best_sol_random)

def test_cache_candidates(n):

    m1_time=[]
    m1_score=[]
    m1_solution=None
    m2_time=[]
    m2_score=[]
    m2_solution=None
    m3_time=[]
    m3_score=[]
    m3_solution=None
    m4_time=[]
    m4_score=[]
    m4_solution=None
    df = None
    for _ in range(n):
        clustering = Clustering(20)
        clusters = clustering.initialize_clusters('greed')
        start = time.time()
        clusters = clustering.local_search(clusters, clustering.steep_local_search, \
            candidates=False, caching=False)
        end = time.time()
        md = clustering.mean_distance()
        m1_score.append(md)
        m1_time.append(end-start)
        if md<np.min(m1_score):
            m1_score=clusters

        clustering = Clustering(20)
        clusters = clustering.initialize_clusters('greed')
        start = time.time()
        clusters = clustering.local_search(clusters, clustering.steep_local_search, \
            candidates=False, caching=True)
        end = time.time()
        md = clustering.mean_distance()
        m2_score.append(md)
        m2_time.append(end-start)
        if md<np.min(m2_score):
            m2_score=clusters

        clustering = Clustering(20)
        clusters = clustering.initialize_clusters('greed')
        start = time.time()
        clusters = clustering.local_search(clusters, clustering.steep_local_search, \
            candidates=True, caching=False)
        end = time.time()
        md = clustering.mean_distance()
        m3_score.append(md)
        m3_time.append(end-start)
        if md<np.min(m3_score):
            m3_score=clusters

        clustering = Clustering(20)
        clusters = clustering.initialize_clusters('greed')
        start = time.time()
        clusters = clustering.local_search(clusters, clustering.steep_local_search, \
            candidates=True, caching=True)
        end = time.time()
        md = clustering.mean_distance()
        m4_score.append(md)
        m4_time.append(end-start)
        if md<np.min(m4_score):
            m4_score=clusters

        df = clustering.df

    # Clustering.showClusters(df, m1_solution)
    # Clustering.showClusters(df, m2_solution)
    # Clustering.showClusters(df, m3_solution)
    # Clustering.showClusters(df, m4_solution)
    print("mean score: {}   min score: {}   max score: {}".format(\
        np.mean(m1_score), min(m1_score), max(m1_score)))
    print("mean time: {}   min time: {}   max time: {}".format(\
        np.mean(m1_time), min(m1_time), max(m1_time)))
    print()
    print("mean score: {}   min score: {}   max score: {}".format(\
        np.mean(m2_score), min(m2_score), max(m2_score)))
    print("mean time: {}   min time: {}   max time: {}".format(\
        np.mean(m2_time), min(m2_time), max(m2_time)))
    print()
    print("mean score: {}   min score: {}   max score: {}".format(\
        np.mean(m3_score), min(m3_score), max(m3_score)))
    print("mean time: {}   min time: {}   max time: {}".format(\
        np.mean(m3_time), min(m3_time), max(m3_time)))
    print()
    print("mean score: {}   min score: {}   max score: {}".format(\
        np.mean(m4_score), min(m4_score), max(m4_score)))
    print("mean time: {}   min time: {}   max time: {}".format(\
        np.mean(m4_time), min(m4_time), max(m4_time)))

def test_alternative_local_search(n):
    m1_time=[]
    m1_score=[]
    m2_time=[]
    m2_score=[]
    m3_time=[]
    m3_score=[]

    def test_msls(n, iterations=2):
        for i in range(iterations):
            print(i)
            #clustering = Clustering(20)
            start = time.time()
            md = multiple_start_local_search(n)
            end = time.time()
            m1_score.append(md)
            m1_time.append(end-start)

    def test_ils(run_time, iterations=2):
        for i in range(iterations):
            print(i)
            clustering = Clustering(20)
            clusters = clustering.initialize_clusters('random')
            start = time.time()
            clusters = clustering.iterative_local_search(clusters, run_time, 8)
            end = time.time()
            md = clustering.mean_distance()
            m2_score.append(md)
            m2_time.append(end-start)

    def test_lnls(run_time,iterations=2):
        for i in range(iterations):
            print(i)
            clustering = Clustering(20)
            clusters = clustering.initialize_clusters('random')
            start = time.time()
            clusters = clustering.large_neighborhood_local_search(clusters, run_time, p=.3)
            end = time.time()
            md = clustering.mean_distance()
            m3_score.append(md)
            m3_time.append(end-start)

    iterations = 10
    test_msls(100, iterations)
    print('finished msls')
    run_time = np.mean(m1_time)
    t1 = threading.Thread(target=test_ils, args=(run_time, iterations))
    t2 = threading.Thread(target=test_lnls, args=(run_time, iterations))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print("mean score: {}   min score: {}   max score: {}".format(\
        np.mean(m1_score), min(m1_score), max(m1_score)))
    print("mean time: {}   min time: {}   max time: {}".format(\
        np.mean(m1_time), min(m1_time), max(m1_time)))
    print()
    print("mean score: {}   min score: {}   max score: {}".format(\
        np.mean(m2_score), min(m2_score), max(m2_score)))
    print("mean time: {}   min time: {}   max time: {}".format(\
        np.mean(m2_time), min(m2_time), max(m2_time)))
    print()
    print("mean score: {}   min score: {}   max score: {}".format(\
        np.mean(m3_score), min(m3_score), max(m3_score)))
    print("mean time: {}   min time: {}   max time: {}".format(\
        np.mean(m3_time), min(m3_time), max(m3_time)))

#test_cache_candidates(1)
test_alternative_local_search(2)
