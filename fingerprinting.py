import glob
import json
from datetime import datetime
import haversine as hs
from haversine import Unit
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from geopy.distance import distance
import copy
from tqdm import tqdm
from numpy import random
import seaborn as sb
import math
import pickle
from time import time
from multiprocessing import Pool
import multiprocessing as mp
import sys
import psutil
import similaritymeasures
import scipy.stats as st

max_count = mp.cpu_count()
from functools import partial
import similaritymeasures

LAT_MIN = 39.6797
LAT_MAX = 40.1280
LNG_MIN = 116.0287
LNG_MAX = 116.7064

LAT_BLOCKS = 1000
LNG_BLOCKS = 1000
NEIGHBOR_RANGE = 2
lat_interval = (LAT_MAX - LAT_MIN) / LAT_BLOCKS
lng_interval = (LNG_MAX - LNG_MIN) / LNG_BLOCKS


VALID_TRAJECTORY_LENGTH = 50


def gpsInRange(gps):
    global LAT_MIN, LAT_MAX, LNG_MIN, LNG_MAX
    return LAT_MIN <= gps[0] <= LAT_MAX and LNG_MIN <= gps[1] <= LNG_MAX


def gpsInRangeEntry(gps_entry):
    global LAT_MIN, LAT_MAX, LNG_MIN, LNG_MAX
    return LAT_MIN <= gps_entry[0] <= LAT_MAX and LNG_MIN <= gps_entry[1] <= LNG_MAX


def gpsInRangeTrajectory(gps_trajectory):
    for gps_entry in gps_trajectory:
        if not gpsInRangeEntry(gps_entry):
            return False
    return True


def haversine(coord1, coord2):
    R = 6372800
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )

    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def isValidTrajectory(gps_trajectory):
    global VALID_TRAJECTORY_LENGTH
    return len(gps_trajectory) >= VALID_TRAJECTORY_LENGTH and gpsInRangeTrajectory(
        gps_trajectory
    )


def getCell(gps):
    global LAT_MIN, LAT_MAX, LNG_MIN, LNG_MAX, LAT_BLOCKS, LNG_BLOCKS
    assert gpsInRange(gps)
    lat_step = (LAT_MAX - LAT_MIN) / LAT_BLOCKS
    lng_step = (LNG_MAX - LNG_MIN) / LNG_BLOCKS
    lat_id = int((gps[0] - LAT_MIN) / lat_step)
    lng_id = int((gps[1] - LNG_MIN) / lng_step)
    return lat_id, lng_id


def cellize(gps_entry):
    lat_id, lng_id = getCell(gps_entry[:2])
    return lat_id, lng_id, gps_entry[2]


def cellizeTrajectory(gps_trajectory):
    cell_trajectory = []
    for gps_entry in gps_trajectory:
        cell_trajectory.append(cellize(gps_entry))
    return cell_trajectory


def generateTransitionProbability(cell_trajectories):
    global LAT_BLOCKS, LNG_BLOCKS
    transition_matrix = [
        [defaultdict(lambda: 0) for _ in range(LNG_BLOCKS)] for _ in range(LAT_BLOCKS)
    ]
    emission_matrix = defaultdict(lambda: 0)
    for cell_trajectory in cell_trajectories:
        for (lat, lng, _), (current_lat, current_lng, _) in zip(
            cell_trajectory, cell_trajectory[1:]
        ):
            transition_matrix[lat][lng][(current_lat, current_lng)] += 1
            emission_matrix[(lat, lng)] += 1
    emission_matrix = {k: v for k, v in emission_matrix.items()}
    transition_matrix = [
        [{k: v for k, v in transition_matrix[i][j].items()} for j in range(LNG_BLOCKS)]
        for i in range(LAT_BLOCKS)
    ]
    return emission_matrix, transition_matrix


def cellInRange(cell):
    return 0 <= cell[0] < LAT_BLOCKS and 0 <= cell[1] < LNG_BLOCKS


def getCenter(cell):
    global LAT_MIN, LNG_MIN, lat_interval, lng_interval
    assert cellInRange(cell)
    lat = (0.5 + cell[0]) * lat_interval + LAT_MIN
    lng = (0.5 + cell[1]) * lng_interval + LNG_MIN
    return lat, lng


def getEmission(cell):
    global NEIGHBOR_RANGE, emission
    cell_emissions = {}
    lat_cell, lng_cell = cell
    for new_lat in range(lat_cell - NEIGHBOR_RANGE, lat_cell + NEIGHBOR_RANGE + 1):
        for new_lng in range(lng_cell - NEIGHBOR_RANGE, lng_cell + NEIGHBOR_RANGE + 1):
            if cellInRange((new_lat, new_lng)):
                em = emission.get((new_lat, new_lng))
                cell_emissions[(new_lat, new_lng)] = em if em else 0
    sum_emissions = sum(cell_emissions.values())
    if sum_emissions <= 0:
        return {key: 1 / len(cell_emissions) for key in cell_emissions.keys()}
    else:
        return {key: value / sum_emissions for key, value in cell_emissions.items()}


def getTransition(cell):
    global NEIGHBOR_RANGE, transition
    lat_cell, lng_cell = cell
    total_count = sum(transition[lat_cell][lng_cell].values())
    cell_transitions = {}
    if total_count <= 0:
        for new_lat in range(lat_cell - NEIGHBOR_RANGE, lat_cell + NEIGHBOR_RANGE + 1):
            for new_lng in range(
                lng_cell - NEIGHBOR_RANGE, lng_cell + NEIGHBOR_RANGE + 1
            ):
                if cellInRange((new_lat, new_lng)):
                    cell_transitions[(new_lat, new_lng)] = 1
        return {
            key: 1 / len(cell_transitions.keys())
            for key, value in cell_transitions.items()
        }
    else:
        return {
            key: value / total_count
            for key, value in transition[lat_cell][lng_cell].items()
        }


def calcSqDistance(cell_1, cell_2):
    (lat_1, lng_1), (lat_2, lng_2) = cell_1, cell_2
    return (lat_1 - lat_2) ** 2 + (lng_1 - lng_2) ** 2


def getCtgTransition(prev_cell, true_cell, tau, dist_filter=True):
    candidates = getTransition(prev_cell)
    tau_candidates = dict(filter(lambda x: x[1] >= tau, candidates.items()))

    if dist_filter:
        dist = calcSqDistance(prev_cell, true_cell)
        tau_dist_candidates = dict(
            filter(
                lambda x: calcSqDistance(x[0], true_cell) <= dist,
                tau_candidates.items(),
            )
        )
        return candidates, tau_candidates, tau_dist_candidates
    else:
        return candidates, tau_candidates


def sampleProportionallyWithTruth(candidates, truth, p):
    if sum(candidates.values()) <= 0:
        sampled_cell = list(candidates.keys())[random.randint(len(candidates))]
    else:
        if truth:
            total = sum(candidates.values()) - candidates[truth]
            candidates = {key: p * value / total for key, value in candidates.items()}
            candidates[truth] = 1 - p
        else:
            total = sum(candidates.values())
            candidates = {key: value / total for key, value in candidates.items()}
        sampled_cell = list(candidates.keys())[
            random.choice(
                range(len(candidates)), p=[candidates[key] for key in candidates.keys()]
            )
        ]

    is_FP = False if truth == sampled_cell else True
    return sampled_cell, is_FP


def getClosestPoint(cell, candidates):
    closest_pt = None, None
    min_dist = None
    for cand in candidates:
        dist = calcSqDistance(cell, cand)
        if min_dist == None or dist < min_dist:
            min_dist = dist
            closest_pt = cand
    return closest_pt


def sampleCandidates(
    prev_cell, true_cell, p, tau, dist_filter=True, replace=True, DEBUG=False
):
    global transition, emission
    candidates, tau_candidates, tau_dist_candidates = getCtgTransition(
        prev_cell, true_cell, tau, dist_filter=True
    )
    if DEBUG:
        print(prev_cell, true_cell, p, tau, dist_filter, replace)
    if DEBUG:
        print(candidates, tau_candidates, tau_dist_candidates)
    is_FP = False

    if true_cell in tau_dist_candidates.keys():
        if DEBUG:
            print("In tau_dist")
        if len(tau_dist_candidates) > 1:
            if DEBUG:
                print("tau_dist >1")

            sampled_cell, is_FP = sampleProportionallyWithTruth(
                tau_dist_candidates, true_cell, p
            )
        elif len(tau_candidates) > 1:
            if DEBUG:
                print("tau > 1")

            sampled_cell, is_FP = sampleProportionallyWithTruth(
                tau_candidates, true_cell, p
            )
        else:
            if DEBUG:
                print("not satisfied tau, remain truth")

            sampled_cell, is_FP = true_cell, None
    else:
        if DEBUG:
            print("Not in tau_dist")
        if len(tau_candidates) > 1:
            if DEBUG:
                print("tau > 1")

            if replace:
                temp_true_cell = getClosestPoint(true_cell, tau_candidates)
                sampled_cell, _ = sampleProportionallyWithTruth(
                    tau_candidates, temp_true_cell, p
                )
            else:
                sampled_cell, _ = sampleProportionallyWithTruth(
                    tau_candidates, None, None
                )
            is_FP = True
        else:
            if DEBUG:
                print("not satisfied tau, Keep truth")

            sampled_cell, is_FP = true_cell, None
    if DEBUG:
        print(sampled_cell, "FP", is_FP)
    return sampled_cell, is_FP


def sampleCenter(cell):
    global LAT_MIN, LNG_MIN, lat_interval, lng_interval
    assert cellInRange(cell)
    lat, lng = getCenter(cell)
    return lat, lng


def fingerPrint(cell_trajectory, tau, p, theta, dist_filter=True, DEBUG=False):
    fp_trajectory = []
    fp_flag = []

    assert p >= 0

    if p == 0:
        if DEBUG:
            print("p = 0, return origin")
        for lat_cell, lng_cell, time in cell_trajectory:
            lat, lng = sampleCenter((lat_cell, lng_cell))
            fp_trajectory.append((lat, lng, time))
            fp_flag.append(False)
        return fp_trajectory, fp_flag

    block_count = 0

    fp_count = 0

    p_current = p

    lat_cell, lng_cell, time = cell_trajectory[0]
    if DEBUG:
        print("First cell truth: %5d, %5d, %10.2f" % (lat_cell, lng_cell, time))

    distribution = getEmission((lat_cell, lng_cell))

    (sampled_lat, sampled_lng), is_FP = sampleProportionallyWithTruth(
        distribution, (lat_cell, lng_cell), p_current
    )
    if DEBUG:
        print("Sampled: ", sampled_lat, sampled_lng, "FP" if is_FP else "--")

    prev_lat, prev_lng, prev_time = sampled_lat, sampled_lng, time

    lat, lng = sampleCenter((sampled_lat, sampled_lng))

    fp_trajectory.append((lat, lng, time))

    if is_FP == True:
        fp_flag.append(True)
        fp_count += 1
        if DEBUG:
            print("FP!")
    else:
        fp_flag.append(False)

    block_count += 1

    for true_lat, true_lng, true_time in cell_trajectory[1:]:
        if DEBUG:
            print("Prev:", prev_lat, prev_lng, "Truth: ", true_lat, true_lng)

        (sampled_lat, sampled_lng), is_FP = sampleCandidates(
            (prev_lat, prev_lng), (true_lat, true_lng), p_current, tau, dist_filter
        )

        if DEBUG:
            print(
                "Sampled: ",
                sampled_lat,
                sampled_lng,
                "FP" if is_FP else "--",
                "(p =",
                p_current,
                ")",
            )

        prev_lat, prev_lng, prev_time = sampled_lat, sampled_lng, time

        lat, lng = sampleCenter((sampled_lat, sampled_lng))

        fp_trajectory.append((lat, lng, true_time))

        if is_FP == True:
            fp_flag.append(True)
            fp_count += 1
            if DEBUG:
                print("FP!")
        else:
            fp_flag.append(False)

        block_count += 1

        if block_count >= math.ceil(1 / p):
            length = len(fp_trajectory)
            if DEBUG:
                print("FP_COUNT: %d, Expected: %.2f" % (fp_count, p * length))
            if fp_count > p * length:
                p_current = p * (1 - theta)
            elif fp_count < p * length:
                p_current = p * (1 + theta)
            else:
                p_current = p
            if p_current >= 1:
                p_current = 1
            block_count = 0
    return fp_trajectory, fp_flag


def BSIntegratedFingerprinting(
    cell_trajectory,
    tau,
    p,
    theta,
    dist_filter,
    bs_ids,
    bs_fp_entries,
    bs_id,
    DEBUG=False,
):
    fp_trajectory = []
    fp_flag = []

    assert p >= 0

    if p == 0:
        if DEBUG:
            print("p = 0, return origin")
        for lat_cell, lng_cell, time in cell_trajectory:
            lat, lng = sampleCenter((lat_cell, lng_cell))
            fp_trajectory.append((lat, lng, time))
            fp_flag.append(False)
        return fp_trajectory, fp_flag

    block_count = 0

    fp_count = 0

    bs_index = 0

    bs_fp_count = len(bs_fp_entries)
    trajectory_length = len(cell_trajectory)
    p_expected = (p * trajectory_length - (bs_fp_count - bs_id)) / (
        trajectory_length - bs_fp_count
    )
    if p_expected <= 0:
        for lat_cell, lng_cell, time in cell_trajectory:
            lat, lng = sampleCenter((lat_cell, lng_cell))
            fp_trajectory.append((lat, lng, time))
            fp_flag.append(False)
        return fp_trajectory, fp_flag

    p_current = p_expected if p_expected <= 1 else 1

    lat_cell, lng_cell, time = cell_trajectory[0]
    if DEBUG:
        print("First cell truth: %5d, %5d, %10.2f" % (lat_cell, lng_cell, time))

    is_code = False
    is_FP = False
    if bs_index < bs_id and bs_ids[bs_index] == 0:
        sampled_lat, sampled_lng = lat_cell, lng_cell
        lat, lng = sampleCenter((sampled_lat, sampled_lng))
        bs_index += 1
        is_code = True
    elif bs_index < len(bs_ids) and bs_ids[bs_index] == 0:
        lat, lng, _ = bs_fp_entries[bs_index]
        sampled_lat, sampled_lng = getCell((lat, lng))
        lat, lng = sampleCenter((sampled_lat, sampled_lng))
        bs_index += 1
        is_code = True
    else:

        distribution = getEmission((lat_cell, lng_cell))

        (sampled_lat, sampled_lng), is_FP = sampleProportionallyWithTruth(
            distribution, (lat_cell, lng_cell), p_current
        )
        if DEBUG:
            print("Sampled: ", sampled_lat, sampled_lng, "FP" if is_FP else "--")

        lat, lng = sampleCenter((sampled_lat, sampled_lng))

        block_count += 1

    prev_lat, prev_lng, prev_time = sampled_lat, sampled_lng, time

    fp_trajectory.append((lat, lng, time))

    fp_flag.append(is_FP)
    if is_FP == True:
        fp_count += 1
        if DEBUG:
            print("FP!")

    for i, (true_lat, true_lng, true_time) in enumerate(cell_trajectory):

        if i == 0:
            continue

        if block_count >= math.ceil(1 / p_expected):
            length = len(fp_trajectory)
            if DEBUG:
                print(
                    "FP_COUNT: %d, Expected: %.2f"
                    % (fp_count, p_expected * (length - bs_index))
                )
            if fp_count > p_expected * (length - bs_index):
                p_current = p_expected * (1 - theta)
                if DEBUG:
                    print("MORE than expected, p <-", p_current)
            elif fp_count < p_expected * (length - bs_index):
                p_current = p_expected * (1 + theta)
                if DEBUG:
                    print("LESS than expected, p <-", p_current)
            else:
                p_current = p_expected
                if DEBUG:
                    print("Equal to expected, p <-", p_current)
            block_count = 0
            if p_current >= 1:
                p_current = 1
            if DEBUG:
                print("Final p: ", p_current)

        is_code = False
        is_FP = False
        if bs_index < bs_id and bs_ids[bs_index] == i:
            sampled_lat, sampled_lng = true_lat, true_lng
            lat, lng = sampleCenter((sampled_lat, sampled_lng))
            bs_index += 1
            is_code = True
        elif bs_index < len(bs_ids) and bs_ids[bs_index] == i:
            lat, lng, _ = bs_fp_entries[bs_index]
            sampled_lat, sampled_lng = getCell((lat, lng))
            lat, lng = sampleCenter((sampled_lat, sampled_lng))
            bs_index += 1
            is_code = True
        else:
            if DEBUG:
                print("Prev:", prev_lat, prev_lng, "Truth: ", true_lat, true_lng)

            (sampled_lat, sampled_lng), is_FP = sampleCandidates(
                (prev_lat, prev_lng), (true_lat, true_lng), p_current, tau, dist_filter
            )

            if DEBUG:
                print(
                    "Sampled: ",
                    sampled_lat,
                    sampled_lng,
                    "FP" if is_FP else "--",
                    "(p =",
                    p_current,
                    ")",
                )

            prev_lat, prev_lng, prev_time = sampled_lat, sampled_lng, time

            lat, lng = sampleCenter((sampled_lat, sampled_lng))

            block_count += 1

        fp_trajectory.append((lat, lng, true_time))

        fp_flag.append(is_FP)
        if is_FP == True:
            if is_code == False:
                fp_count += 1
            if DEBUG:
                print("FP!")

    return fp_trajectory, fp_flag


def BSAnalysisOnSuspects(
    colluding_trajectory, real_trajectory, suspects, indexes, bs_block_size, bs_blocks
):

    fp_count_in_block = 0
    for i, index in enumerate(indexes):

        lat, lng, time = colluding_trajectory[index]
        if getCell((lat, lng)) != tuple(real_trajectory[index][:2]):
            fp_count_in_block += 1
        if (i + 1) % bs_block_size == 0:
            if fp_count_in_block > bs_block_size / 2:
                for suspect in suspects:
                    if suspect % (bs_blocks + 1) == i // bs_block_size:
                        return suspect
            fp_count_in_block = 0
    for suspect in suspects:
        if suspect % (bs_blocks + 1) == bs_blocks:
            return suspect
    return suspects[0]
