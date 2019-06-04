import numpy as np
from fbpca import pca
from heapq import heappush, heappop, heappushpop



def euclidean(p1, p2):
    return np.sqrt(np.sum(np.power(p2 - p1, 2)))


class FTTree:
    def __init__(self, points, dist_fn=euclidean,  root=None, root_ind=None, inds=None, distances=None,max=None, max_idx=None):
        if inds is None:
            inds = list(range(points.shape[0]))
        self.inds = inds

        if root is None:
            idx = np.random.choice(len(points))
            root = points[idx]
            root_ind = inds[idx]


        if not len(points):
            raise ValueError('Points can not be empty.')

        self.left = None
        self.right = None

        self.numFeatures = len(points[0])
        self.root = root #center for further splits
        self.ind = root_ind #global data location
        self.points = points

        self.dist_fn = dist_fn

        if distances is None or max is None or max_idx is None:
            distances = [self.dist_fn(self.root, p) for p in points]
            max_idx = distances.index(np.max(distances))
            max = distances[max_idx]

        self.max = max
        self.furthest = self.points[max_idx]

        self.dists_to_furthest = [self.dist_fn(self.furthest, p) for p in points]
        #self.split = float(max)/2.

        left_points = []
        right_points = []
        left_dists = []
        left_max = None
        left_inds = []
        right_inds = []
        right_root = None
        max_right = 0
        max_left = 0


        for point, distance, fardistance, idx in zip(points, distances, self.dists_to_furthest, inds):
            if distance > fardistance:
                right_points.append(point)
                right_inds.append(idx)
                if distance > max_right:
                    right_root = point
                    right_root_ind = idx
                    max_right = distance
            else:
                left_points.append(point)
                left_inds.append(idx)
                left_dists.append(distance)
                if distance > max_left:
                    max_left = distance


        self.left_inds = left_inds
        self.right_inds = right_inds
        # print(left_inds)
        # print(right_inds)
        # print(len(left_points))
        # print(len(right_points))
        # print(left_points)
        # print(right_points)
        # print(right_root)
        # print(right_root_ind)
        if len(left_points) > 1:
            self.left = FTTree(points=left_points, dist_fn=self.dist_fn,
                               inds=left_inds, root=self.root, root_ind=self.ind,
                               distances=left_dists, max=None)

        if len(right_points) > 0:
            self.right = FTTree(points=right_points, dist_fn=self.dist_fn,
                                inds=right_inds, root=right_root,
                                root_ind = right_root_ind)
    def _is_leaf(self):
        return (self.left is None) and (self.right is None)

    def traverse(self):
        nodes_to_visit = [self]
        result = [(-1*float("inf"), self.ind)]

        while len(nodes_to_visit) > 0:
            # print([n.ind for n in nodes_to_visit])
            # print([len(n.points) for n in nodes_to_visit])
            node = nodes_to_visit.pop(0)


            if node is None:
                continue

            if node.right is not None:
                heappush(result, (-1*node.max,node.right.ind))
            nodes_to_visit.append(node.right)
            nodes_to_visit.append(node.left)

        self.nodeHeap = result
        return(result)






    def __str__(self):
        if self._is_leaf():
            return(str(self.ind))
        else:
            result = str(self.ind) + ', ['
            result += str(self.left) + ', '
            result += str(self.right) + ']'

        return(result)
