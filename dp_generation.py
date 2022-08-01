#!/usr/bin/python
# -*- coding: utf-8 -*-

import random
from shapely.geometry import Polygon, Point
from scipy.spatial import ConvexHull, Delaunay, convex_hull_plot_2d
from numpy import random
import math


def uniformSample(poly):
    (minx, miny, maxx, maxy) = poly.bounds
    while True:
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if poly.contains(p):
            return (p.x, p.y)


minimum_prob = 1e-3


def sqEuclidean(cell_1, cell_2):
    return (cell_1[0] - cell_2[0]) ** 2 + (cell_1[1] - cell_2[1]) ** 2


def getClosestCell(cell, vertices):
    (lat_cell, lng_cell) = cell
    min_dist = None
    min_cell = None
    for (lat, lng) in vertices:
        dist = sqEuclidean((lat, lng), (lat_cell, lng_cell))
        if min_dist == None or dist < min_dist:
            min_dist = dist
            min_cell = (lat, lng)
    return min_cell


def getCorners(cell):
    (lat, lng) = cell
    return [
        (lat - 0.5, lng - 0.5),
        (lat - 0.5, lng + 0.5),
        (lat + 0.5, lng - 0.5),
        (lat + 0.5, lng + 0.5),
    ]


def generateDPTrajectory(cell_trajectory):
    global dp_transition, dp_emission
    (prev_lat, prev_lng) = (int(cell_trajectory[0][0]), int(cell_trajectory[0][1]))

    posterior = [[0 for j in range(LNG_BLOCKS)] for i in range(LAT_BLOCKS)]
    posterior[prev_lat][prev_lng] = 1

    prior = [[0 for j in range(LNG_BLOCKS)] for i in range(LAT_BLOCKS)]
    for lat in range(LAT_BLOCKS):
        for lng in range(LNG_BLOCKS):
            for ((tran_lat, tran_lng), tran_prob) in getDPTransition(
                (lat, lng)
            ).items():
                prior[tran_lat][tran_lng] += posterior[lat][lng] * tran_prob

    result = []
    for (idx, (lat_cell, lng_cell, ts)) in tqdm(
        list(zip(range(len(cell_trajectory)), cell_trajectory))
    ):
        (true_lat, true_lng) = (int(lat_cell), int(lng_cell))

        sorted_prior = []
        for lat in range(LAT_BLOCKS):
            for lng in range(LNG_BLOCKS):
                sorted_prior += [(lat, lng, prior[lat][lng])]
        sorted_prior = sorted(sorted_prior, key=lambda x: x[2], reverse=True)

        location_set = []
        set_sum = 0
        for (i, (x, y, pr)) in enumerate(sorted_prior):
            if set_sum >= 1 - delta_dp:
                break
            location_set.append((x, y, pr))
            set_sum += pr

        points = []
        for (x, y, _) in location_set:
            points += getCorners((x, y))
        points = np.array(points)

        try:
            c_hull = ConvexHull(points)
        except:
            print(idx, "c_hull Error", points)
            points = [
                (x, y)
                for (x, y) in [
                    (prev_lat - NEIGHBOR_RANGE, prev_lng - NEIGHBOR_RANGE),
                    (prev_lat - NEIGHBOR_RANGE, prev_lng + NEIGHBOR_RANGE),
                    (prev_lat + NEIGHBOR_RANGE, prev_lng - NEIGHBOR_RANGE),
                    (prev_lat + NEIGHBOR_RANGE, prev_lng + NEIGHBOR_RANGE),
                ]
                if cellInRange((x, y))
            ]
            points = np.array([(x, y) for (x, y) in points])
            c_hull = ConvexHull(points)
        c_vertices = [
            (points[vertex, 0], points[vertex, 1]) for vertex in c_hull.vertices
        ]

        if not Polygon(c_vertices).contains(Point(lat_cell, lng_cell)):

            (lat_cell, lng_cell) = getClosestCell((lat_cell, lng_cell), c_vertices)

        vertex_set = {}
        for x in c_vertices:
            for y in c_vertices:
                if x == y:
                    continue
                vertex_set[(x[0] - y[0], x[1] - y[1])] = 1
        s_points = np.array(list(vertex_set.keys()))

        s_hull = ConvexHull(s_points)

        s_vertices = [
            (s_points[vertex, 0], s_points[vertex, 1]) for vertex in s_hull.vertices
        ]
        p = Polygon(s_vertices)

        t_value = None
        l = 1
        while True:

            sampled_points = [uniformSample(p) for i in range(l)]
            t_sum = 0
            for (x, y) in sampled_points:
                t_sum += x * x + y * y
            new_t = (t_sum / l) ** -0.5
            if t_value == None or abs(new_t - t_value) > 1e-3:
                t_value = new_t
                l += 1
            else:
                break
        normalzied_vertices = [(x * t_value, y * t_value) for (x, y) in s_vertices]
        sampled_point = uniformSample(Polygon(normalzied_vertices))
        noise_r = random.gamma(3, epsilon ** -1)
        (final_lat, final_lng) = (
            lat_cell + sampled_point[0] / t_value * noise_r,
            lng_cell + sampled_point[1] / t_value * noise_r,
        )
        while not cellInRange((final_lat, final_lng)):
            sampled_point = uniformSample(Polygon(normalzied_vertices))
            noise_r = random.gamma(3, epsilon ** -1)
            (final_lat, final_lng) = (
                lat_cell + sampled_point[0] / t_value * noise_r,
                lng_cell + sampled_point[1] / t_value * noise_r,
            )
        (final_lat, final_lng) = (int(final_lat), int(final_lng))

        posterior = [[0 for j in range(LNG_BLOCKS)] for i in range(LAT_BLOCKS)]
        prob_from = [[0 for j in range(LNG_BLOCKS)] for i in range(LAT_BLOCKS)]
        sum_prob = 0
        for lat in range(LAT_BLOCKS):
            for lng in range(LNG_BLOCKS):
                prob_from[lat][lng] = (
                    epsilon ** 2
                    / 2
                    / s_hull.area
                    * math.e
                    ** (
                        -epsilon
                        * t_value
                        * math.sqrt((final_lat - lat) ** 2 + (final_lng - lng) ** 2)
                    )
                )
                sum_prob += prior[lat][lng] * prob_from[lat][lng]

        if sum_prob <= 0:
            posterior[final_lat][final_lng] = 1
        else:
            for lat in range(LAT_BLOCKS):
                for lng in range(LNG_BLOCKS):
                    posterior[lat][lng] = (
                        prior[lat][lng] * prob_from[lat][lng] / sum_prob
                    )

        #

        sorted_posterior = []
        for lat in range(LAT_BLOCKS):
            for lng in range(LNG_BLOCKS):
                sorted_posterior += [(lat, lng, posterior[lat][lng])]
        sorted_posterior = sorted(sorted_posterior, key=lambda x: x[2], reverse=True)

        prior = [[0 for j in range(LNG_BLOCKS)] for i in range(LAT_BLOCKS)]
        dp_distribution = getDPTransition((final_lat, final_lng))

        for lat in range(LAT_BLOCKS):
            for lng in range(LNG_BLOCKS):
                for ((tran_lat, tran_lng), tran_prob) in dp_distribution.items():
                    prior[tran_lat][tran_lng] += posterior[lat][lng] * tran_prob

        result.append((final_lat, final_lng, ts))

    return result
