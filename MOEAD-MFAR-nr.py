import numpy as np
import matplotlib.pyplot as plt
import random
import math
import os
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import statistics
from pymoo.indicators.hv import HV

'''THE_PATH_WHERE_THE_PICTURE_IS_SAVED'''
figures_dir = r'D:\SAG_MOEAD_exp_fig\SAG_MOEAD_8.2.nr\fig'

"""parameters"""
N = 100
Objective_num = 2
Weight_vector_num = N
Neighbor_num = 5 

Evolution_epoch = 100
Xy_mutation_rate = 0.5
Power_mutation_rate = 0.1

connection_mutation_rate = 0.5

Remove_Rate = 1

UAV_mutation_rate = 0.5

IOT_bd_cate_mutation_rate = 0.5

"Communication System Environmental Parameters"

B = 1000000

P0_IOT_UAV = 0.000142

P0_UAV_LEO = 0.000142
Fading_UAV_LEO = 3.16

N0 = -174


def dbm_to_watt_per_hz(dbm_per_hz):
    watts_per_hz = 10 ** ((dbm_per_hz - 30) / 10) / 1000
    return watts_per_hz

N0 = dbm_to_watt_per_hz(N0)

"""uav-related-parameters"""

Num_UAV = 1

Length_UAV_area = 1000

Width_UAV_area = 1000

Coordinate_UAV_max = [Length_UAV_area, Width_UAV_area]
Coordinate_UAV_min = [0, 0]

Height_UAV = 100

Connect_distance_max = 300

Safe_distance_UAV = 30

Pmax_UAV_Transmit = 50

Pmin_UAV_Transmit = 0

Capacity_UAV = 500000000

"""IOT-A-RELATED-PARAMETERS"""

Num_IOT_category = 2

Num_IOT_A = 10

Length_IOT_area_A = 1000
Width_IOT_area_A = 1000

Height_IOT_A = 0

P_IOT_Transmit_max_A = 1
P_IOT_Transmit_min_A = 0

Transmit_rate_IOT_min_A = 10000

"""IOT-B-RELATED-PARAMETERS"""

Num_IOT_B = 10

Length_IOT_area_B = 600
Width_IOT_area_B = 600

Height_IOT_B = 0

P_IOT_Transmit_max_B = 1
P_IOT_Transmit_min_B = 0

Transmit_rate_IOT_min_B = 10000

"""LEO-RELATED-PARAMETERS"""

Num_LEO = 1

Height_LEO = 1000000

receive_rate_max = 100000000



class IOT:
    def __init__(self, id, x, y, z, power, bandwidth, rate, uav):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.power = power
        self.bandwidth = bandwidth
        self.rate = rate
        
        self.uav = uav



class UAV:
    def __init__(self, id, x, y, z, power, bandwidth, leo, iot_num):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.power = power
        self.bandwidth = bandwidth
        
        self.leo = leo
        
        self.iot_num = iot_num



class LEO:
    def __init__(self, id, z):
        self.id = id
        self.z = z



def generate_IOT(num, x_list, y_list, z_list, power_list, bandwidth_list, transmit_rate, connect_uav_list):
    
    
    
    IOT_list = []
    for i in range(num):
        id = i
        x = x_list[i]
        y = y_list[i]
        z = z_list[i]
        power = power_list[i]
        bandwidth = bandwidth_list[i]
        rate = transmit_rate[i]
        uav = connect_uav_list[i]
        IOT_list.append(IOT(id, x, y, z, power, bandwidth, rate, uav))
    return IOT_list



def generate_UAV(num, x_list, y_list, z_list, power_list, bandwidth_list, connect_leo_list, connect_iot_num_list):
    UAV_list = []
    for i in range(num):
        id = i
        x = x_list[i]
        y = y_list[i]
        z = z_list[i]
        power = power_list[i]
        bandwidth = bandwidth_list[i]
        leo = connect_leo_list[i]
        connect_iot_num = connect_iot_num_list[i]
        UAV_list.append(UAV(id, x, y, z, power, bandwidth, leo, connect_iot_num))
    return UAV_list



def generate_LEO(num, z_list):
    LEO_list = []
    for i in range(num):
        id = i
        z = z_list[i]
        LEO_list.append(LEO(id, z))
    return LEO_list



def calculate_IOT_UAV_capacity(IOT_list):
    
    IOT_UAV_capacity_list = []
    for i in range(len(IOT_list)):
        
        uav = IOT_list[i].uav
        
        distance = math.sqrt((IOT_list[i].x - uav.x) ** 2 + (IOT_list[i].y - uav.y) ** 2 + (
                IOT_list[i].z - uav.z) ** 2)
        
        channel_gain = P0_IOT_UAV / (distance ** 2)
        
        capacity = IOT_list[i].bandwidth * math.log2(
            1 + (IOT_list[i].power * channel_gain) / (IOT_list[i].bandwidth * N0))
        IOT_UAV_capacity_list.append(capacity)
    return IOT_UAV_capacity_list



def calculate_UAV_LEO_capacity(UAV_list):
    
    UAV_LEO_capacity_list = []
    for i in range(len(UAV_list)):
        capacity = UAV_list[i].bandwidth * math.log2(1 + UAV_list[i].power * Fading_UAV_LEO)
        UAV_LEO_capacity_list[i].append(capacity)
    return UAV_LEO_capacity_list



def calculate_IOT_UAV_SNR(IOT_list):
    
    IOT_UAV_SNR_list = []
    for i in range(len(IOT_list)):
        
        uav = IOT_list[i].uav
        
        distance = math.sqrt((IOT_list[i].x - uav.x) ** 2 + (IOT_list[i].y - uav.y) ** 2 + (
                IOT_list[i].z - uav.z) ** 2)
        
        channel_gain = P0_IOT_UAV / (distance ** 2)
        
        if IOT_list[i].bandwidth == 0:
            snr = 0
        else:
            snr = IOT_list[i].power * channel_gain / (IOT_list[i].bandwidth * N0)
        IOT_UAV_SNR_list.append(snr)
    return IOT_UAV_SNR_list



def calculate_UAV_LEO_SNR(UAV_list, IOT_list):
    
    UAV_LEO_SNR_list = []
    for i in range(len(IOT_list)):
        iot = IOT_list[i]
        
        uav = iot.uav
        
        leo = uav.leo
        
        distance = math.sqrt((uav.z - leo.z) ** 2)
        
        channel_gain = P0_UAV_LEO / (distance ** 2)
        
        uav_power = uav.power / uav.connect_iot_num
        
        if iot.bandwidth == 0:
            snr = 0
        else:
            snr = uav_power * channel_gain / (iot.bandwidth * N0)
        UAV_LEO_SNR_list.append(snr)
    return UAV_LEO_SNR_list



def calculate_IOT_LEO_capacity(IOT_list, UAV_list):
    
    IOT_UAV_SNR = calculate_IOT_UAV_SNR(IOT_list)
    UAV_LEO_SNR = calculate_UAV_LEO_SNR(UAV_list, IOT_list)

    IOT_LEO_capacity = []
    for i in range(len(IOT_list)):
        
        iot = IOT_list[i]
        
        snr_iot_uav = IOT_UAV_SNR[i]
        
        snr_uav_leo = UAV_LEO_SNR[i]
        
        capacity = iot.bandwidth * math.log2(
            1 + (snr_iot_uav * snr_uav_leo) / (1 + snr_iot_uav + snr_uav_leo))
        IOT_LEO_capacity.append(capacity)
    return IOT_LEO_capacity, IOT_UAV_SNR, UAV_LEO_SNR



def calculate_UAV_power(UAV_list):
    UAV_power = 0
    for i in range(len(UAV_list)):
        UAV_power += UAV_list[i].power
    return UAV_power



def calculate_IOT_power(IOT_list):
    IOT_power = 0
    for i in range(len(IOT_list)):
        IOT_power += IOT_list[i].power
    return IOT_power



def calculate_IOT_power_max(IOT_list):
    IOT_power_max = 0
    for ii in range(len(IOT_list)):
        if IOT_list[ii].power > IOT_power_max:
            IOT_power_max = IOT_list[ii].power
    return IOT_power_max



def generate_populations_uav_xy(num_populations, num_uav, x_range, y_range):
    
    
    populations_x = []
    populations_y = []
    for i in range(num_populations):
        population_x = []
        population_y = []
        for j in range(num_uav):
            x = random.uniform(0, x_range)
            y = random.uniform(0, y_range)
            population_x.append(x)
            population_y.append(y)
        populations_x.append(population_x)
        populations_y.append(population_y)
    return populations_x, populations_y



def generate_populations_iot_power(num_populations, num_iot, power_max):
    
    "输出是num_populations个种群的iot输出功率组成的list"
    populations = []
    for i in range(num_populations):
        population = []
        for j in range(num_iot):
            power = random.uniform(0, power_max)
            population.append(power)
        populations.append(population)
    return populations



def generate_populations_uav_iot_bandwidth(num_populations, num_iot, bandwidth_all):
    
    
    populations_bw_real = []
    populations_bw_cut = []
    for i in range(num_populations):
        
        per_uav_bw_real = []
        per_uav_bw_cut = []
        for u in range(Num_UAV):
            bw_allocation = [0] + sorted(random.sample(range(1, bandwidth_all), num_iot - 1)) + [bandwidth_all]
            bw_allocation_real = []
            for j in range(len(bw_allocation) - 1):
                bw_j = bw_allocation[j + 1] - bw_allocation[j]
                bw_allocation_real.append(bw_j)
            per_uav_bw_real.append(bw_allocation_real)
            per_uav_bw_cut.append(bw_allocation)
        populations_bw_real.append(per_uav_bw_real)
        populations_bw_cut.append(per_uav_bw_cut)
    return populations_bw_real, populations_bw_cut



def generate_iot_bandwidth(num_iot, bandwidth_all):
    
    bw_allocation = [0] + sorted(random.sample(range(1, bandwidth_all), num_iot - 1)) + [bandwidth_all]
    bw_allocation_real = []
    for j in range(len(bw_allocation) - 1):
        bw_j = bw_allocation[j + 1] - bw_allocation[j]
        bw_allocation_real.append(bw_j)
    return bw_allocation_real, bw_allocation



def generate_iot_bd_proportion(num_iot):
    
    prop_list = []
    prop_cut_list = [random.uniform(0, 1) for i in range(num_iot - 1)]
    prop_cut_list.append(0)
    prop_cut_list.append(1)
    prop_cut_list.sort()

    for i in range(num_iot):
        prop = prop_cut_list[i + 1] - prop_cut_list[i]
        prop_list.append(prop)

    return prop_cut_list, prop_list



def flatten_3d_list(lst_3d):
    
    result = []
    for lst_2d in lst_3d:
        flat_list = [item for sublist in lst_2d for item in sublist]
        result.append(sum(flat_list))
    return result



def generate_weight_vectors(H, m):
    
    weights = np.zeros((m, H))
    if H == 2:
        w = np.linspace(0, 1, m)
        weights[:, 0] = w
        weights[:, 1] = 1 - w
    elif H == 3:
        w = np.linspace(0, 1, int(np.sqrt(m)))
        count = 0
        for i in range(len(w)):
            for j in range(len(w)):
                if w[i] + w[j] <= 1:
                    weights[count, 0] = w[i]
                    weights[count, 1] = w[j]
                    weights[count, 2] = 1 - w[i] - w[j]
                    count += 1
    else:
        
        for i in range(m):
            weight = np.random.rand(H)
            weight = weight / np.sum(weight)
            weights[i, :] = weight
    return weights



def generate_neighbor_index(weights, T):
    
    m = weights.shape[0]
    neighbor_index = np.zeros((m, T), dtype=np.int32)
    distances = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            distances[i, j] = np.sum((weights[i, :] - weights[j, :]) ** 2)
            distances[j, i] = distances[i, j]
    for i in range(m):
        neighbor_index[i, :] = np.argsort(distances[i, :])[1:T + 1]
    return neighbor_index



def get_Z(obj_values):
    
    num_objs = len(obj_values[0])  
    min_values = [float('inf')] * num_objs  

    for obj in obj_values:
        for i, value in enumerate(obj):
            if value < min_values[i]:
                min_values[i] = value

    return min_values



def get_Znad(obj_values):
    
    num_objs = len(obj_values[0])  
    max_values = [-float('inf')] * num_objs  

    for obj in obj_values:
        for i, value in enumerate(obj):
            if value > max_values[i]:
                max_values[i] = value

    return max_values



def crossover(list_a, list_b):
    
    crossover_list_a = [random.randint(0, 1) for _ in range(len(list_a))]
    
    crossover_list_b = [(1 - a) for a in crossover_list_a]
    
    list_a_cross_1 = [a * b for a, b in zip(list_a, crossover_list_a)]
    list_b_cross_1 = [a * b for a, b in zip(list_b, crossover_list_b)]
    
    list_a_new = [a + b for a, b in zip(list_a_cross_1, list_b_cross_1)]

    
    list_a_cross_2 = [a * b for a, b in zip(list_a, crossover_list_b)]
    list_b_cross_2 = [a * b for a, b in zip(list_b, crossover_list_a)]
    
    list_b_new = [a + b for a, b in zip(list_a_cross_2, list_b_cross_2)]

    return list_a_new, list_b_new



def mutation_iot_category_bd(bd_category_list, iot_category_mutation_rate):
    bd_left = bd_category_list[0]
    bd_right = bd_category_list[1]
    if random.random() < iot_category_mutation_rate:
        if random.random() < 0.5:
            mutation_bd = random.uniform(0, 0.5 * bd_left)
            bd_category_list[0] -= mutation_bd
            bd_category_list[1] += mutation_bd
        else:
            mutation_bd = random.uniform(0, 0.5 * bd_right)
            bd_category_list[0] += mutation_bd
            bd_category_list[1] -= mutation_bd
    return bd_category_list



def mutation_UAV_xy(uav_xy_list, mutation_rate, lower_bound, upper_bound):
    
    
    for um in range(len(uav_xy_list)):
        if random.random() < mutation_rate:
            
            mutation_x = random.uniform(-0.1 * Length_UAV_area, 0.1 * Length_UAV_area)
            mutation_y = random.uniform(-0.1 * Width_UAV_area, 0.1 * Width_UAV_area)
            new_coordinate = [x + y for x, y in zip(uav_xy_list[um], [mutation_x, mutation_y])]
            
            new_coordinate[0] = min(max(new_coordinate[0], lower_bound[0]), upper_bound[0])
            new_coordinate[1] = min(max(new_coordinate[1], lower_bound[1]), upper_bound[1])
            uav_xy_list[um] = new_coordinate
    
    child = [[min(max(c[0], lower_bound[0]), upper_bound[0]), min(max(c[1], lower_bound[1]), upper_bound[1])] for c in
             uav_xy_list]
    return child



def crossover_mutation_UAV_xy(parent1, parent2, mutation_rate, lower_bound, upper_bound):
    
    
    child, _ = crossover(parent1, parent2)
    
    for um in range(Num_UAV):
        if random.random() < mutation_rate:
            
            mutation_x = random.uniform(-0.5 * Length_UAV_area, 0.5 * Length_UAV_area)
            mutation_y = random.uniform(-0.5 * Width_UAV_area, 0.5 * Width_UAV_area)
            new_coordinate = [x + y for x, y in zip(child[um], [mutation_x, mutation_y])]
            
            new_coordinate[0] = min(max(new_coordinate[0], lower_bound[0]), upper_bound[0])
            new_coordinate[1] = min(max(new_coordinate[1], lower_bound[1]), upper_bound[1])
            child[um] = new_coordinate
    
    child = [[min(max(c[0], lower_bound[0]), upper_bound[0]), min(max(c[1], lower_bound[1]), upper_bound[1])] for c in
             child]
    return child



def remove_uav(iot_x, iot_y, uav_xy_list, remove_rate):
    

    
    for u in range(len(uav_xy_list)):
        
        if random.random() < remove_rate:
            uav_x = uav_xy_list[u][0]
            uav_y = uav_xy_list[u][1]
            distance = math.sqrt(((uav_x - iot_x) ** 2) + ((uav_y - iot_y) ** 2))
            dis_x = abs(uav_x - iot_x)
            dis_y = abs(uav_y - iot_y)
            dx = (iot_x - uav_x) / dis_x
            dy = (iot_y - uav_y) / dis_y
            uav_x_new = uav_x + dx * random.uniform(0, 0.5 * adjust_rate * dis_x)
            uav_y_new = uav_y + dy * random.uniform(0, 0.5 * adjust_rate * dis_y)
            uav_xy_list[u][0] = uav_x_new
            uav_xy_list[u][1] = uav_y_new
    return uav_xy_list



def crossover_mutation_IOT_power(parent1, parent2, mutation_rate, power_min, power_max):
    
    
    child, _ = crossover(parent1, parent2)
    
    for im in range(len(parent1)):
        if random.random() < mutation_rate:
            
            mutation = random.uniform(-0.1 * Pmax_UAV_Transmit, 0.1 * Pmax_UAV_Transmit)
            new_power = child[im] + mutation
            
            new_power = min(max(new_power, power_min), power_max)
            child[im] = new_power
    
    child = [min(max(c, power_min), power_max) for c in child]
    return child



def crossover_mutation_IOT_power_new(parent1, parent2, mutation_rate, power_min, power_max):
    
    
    child, _ = crossover(parent1, parent2)
    
    my_power_mean = np.array(child)
    power_mean = np.mean(my_power_mean)
    
    
    for im in range(len(parent1)):
        if random.random() < mutation_rate:
            if child[im] >= power_mean:
                
                mutation = random.uniform(-0.1 * Pmax_UAV_Transmit, 0)
                new_power = child[im] + mutation
                
                new_power = min(max(new_power, power_min), power_max)
                child[im] = new_power
            else:
                
                mutation = random.uniform(0, 0.1 * Pmax_UAV_Transmit)
                new_power = child[im] + mutation
                
                new_power = min(max(new_power, power_min), power_max)
                child[im] = new_power
    
    child = [min(max(c, power_min), power_max) for c in child]
    return child



def exchange_IOT_bd(bd_AB_list, iot_cap_min_A_index, iot_cap_max_A_index, iot_cap_min_B_index, iot_cap_max_B_index):
    
    iot_bd_A_list = bd_AB_list[:Num_IOT_A]
    iot_bd_B_list = bd_AB_list[-Num_IOT_B:]

    
    iot_bd_A_min = iot_bd_A_list[iot_cap_min_A_index]
    iot_bd_A_max = iot_bd_A_list[iot_cap_max_A_index]
    
    iot_bd_B_min = iot_bd_B_list[iot_cap_min_B_index]
    iot_bd_B_max = iot_bd_B_list[iot_cap_max_B_index]

    
    bd_A_to_B = random.uniform(0, 0.5 * iot_bd_A_max)
    bd_B_to_A = random.uniform(0, 0.5 * iot_bd_B_max)

    
    bd_A_min_new = iot_bd_A_min + bd_A_to_B
    bd_B_min_new = iot_bd_B_min + bd_B_to_A

    
    iot_bd_A_list[iot_cap_min_A_index] = bd_A_min_new
    iot_bd_B_list[iot_cap_min_B_index] = bd_B_min_new

    
    iot_bd_AB_list = iot_bd_A_list + iot_bd_B_list

    return iot_bd_AB_list



def crossover_UAV_bd(parent1, parent2):
    
    child_sort_list = []
    for i in range(Num_UAV):
        
        idx = random.randint(0, len(parent1[i]) - 1)
        
        child = parent1[i][:idx] + parent2[i][idx:]
        
        child_sort = sorted(child)
        child_sort_list.append(child_sort)
    return child_sort_list



def crossover_IOT_bd(parent1, parent2):
    
    
    child, _ = crossover(parent1, parent2)
    
    child_sort = sorted(child)

    return child_sort



def adjust_iot_bd_prop(iot_bd_prop_list, iot_capa_min_index, iot_capa_max_index):
    
    iot_bd_prop_max = iot_bd_prop_list[iot_capa_max_index]
    
    adjust_iot_bd_prop_ = random.uniform(0, 0.5 * adjust_rate * iot_bd_prop_max)
    
    iot_bd_prop_list[iot_capa_min_index] = iot_bd_prop_list[iot_capa_min_index] + adjust_iot_bd_prop_
    iot_bd_prop_list[iot_capa_max_index] = iot_bd_prop_list[iot_capa_max_index] - adjust_iot_bd_prop_

    return iot_bd_prop_list



def get_iot_capacity_min_xy(iot_capacity_list, iot_x_list, iot_y_list):
    
    iot_capacity_min = min(iot_capacity_list)
    iot_capacity_min_index = iot_capacity_list.index(iot_capacity_min)
    iot_x = iot_x_list[iot_capacity_min_index]
    iot_y = iot_y_list[iot_capacity_min_index]

    return iot_x, iot_y, iot_capacity_min_index



def get_iot_capacity_max_xy(iot_capacity_list, iot_x_list, iot_y_list):
    
    iot_capacity_max = max(iot_capacity_list)
    iot_capacity_max_index = iot_capacity_list.index(iot_capacity_max)
    iot_x = iot_x_list[iot_capacity_max_index]
    iot_y = iot_y_list[iot_capacity_max_index]

    return iot_x, iot_y, iot_capacity_max_index



def crossover_mutation_IOT_UAV_connection(parents1, parents2, mutation_rate):
    
    child1_cross, child2_cross = crossover(parents1, parents2)
    
    for i in range(len(parents1)):
        
        if random.random() < mutation_rate:
            
            iot_connect_random_list = [0] * Num_UAV
            iot_connect_random_list[random.randint(0, Num_UAV - 1)] = 1
            child1_cross[i] = iot_connect_random_list

        
        if random.random() < mutation_rate:
            
            iot_connect_random_list = [0] * Num_UAV
            iot_connect_random_list[random.randint(0, Num_UAV - 1)] = 1
            child2_cross[i] = iot_connect_random_list
    return child1_cross, child2_cross



def get_epoch_decay_rate(epo, init_epo_rate):
    rate = (1 ** epo) * init_epo_rate
    return rate



def get_new_solution_new(i, Bi, last_solutions):
    
    uav_xy_list = last_solutions[i][0]
    
    iot_bd_prop_AB_i = last_solutions[i][1]
    
    uav_xy_list_muta = mutation_UAV_xy(uav_xy_list, UAV_mutation_rate, Coordinate_UAV_min, Coordinate_UAV_max)
    
    iot_bd_prop_AB_i_muta = mutation_iot_category_bd(iot_bd_prop_AB_i, IOT_bd_cate_mutation_rate)
    
    iot_bd_prop_a_i = last_solutions[i][2][0]
    iot_bd_prop_b_i = last_solutions[i][2][1]
    
    iot_capacity_A_list = last_IOT_capacity_A[i]
    iot_capacity_B_list = last_IOT_capacity_B[i]

    
    iot_capacity_min_x_A, iot_capacity_min_y_A, iot_min_A_index = get_iot_capacity_min_xy(
        iot_capacity_A_list, iot_x_list_A_u, iot_y_list_A_u)
    iot_capacity_min_x_B, iot_capacity_min_y_B, iot_min_B_index = get_iot_capacity_min_xy(
        iot_capacity_B_list, iot_x_list_B_u, iot_y_list_B_u)
    iot_capacity_max_x_A, iot_capacity_max_y_A, iot_max_A_index = get_iot_capacity_max_xy(
        iot_capacity_A_list, iot_x_list_A_u, iot_y_list_A_u)
    iot_capacity_max_x_B, iot_capacity_max_y_B, iot_max_B_index = get_iot_capacity_max_xy(
        iot_capacity_B_list, iot_x_list_B_u, iot_y_list_B_u)

    
    if random.random() < 0.5:
        
        uav_xy_list_muta = remove_uav(iot_capacity_min_x_A, iot_capacity_min_y_A, uav_xy_list_muta, Remove_Rate)
        uav_xy_list_muta = remove_uav(iot_capacity_min_x_B, iot_capacity_min_y_B, uav_xy_list_muta, Remove_Rate)
    else:
        
        iot_bd_prop_a_i = adjust_iot_bd_prop(iot_bd_prop_a_i, iot_min_A_index, iot_max_A_index)
        iot_bd_prop_b_i = adjust_iot_bd_prop(iot_bd_prop_b_i, iot_min_B_index, iot_max_B_index)

    return [uav_xy_list_muta, iot_bd_prop_AB_i_muta, [iot_bd_prop_a_i, iot_bd_prop_b_i]]




def update_connect_num(uav_class_list, connect_list):
    for u in range(len(uav_class_list)):
        connect_num = 0
        for i in range(len(connect_list)):
            connect_num += connect_list[i][u]
        uav_class_list[u].iot_num = connect_num
    return uav_class_list



def update_connect_uav(iot_class_list, uav_class_list, connect_list):
    for i in range(len(iot_class_list)):
        uav_id = connect_list[i].index(1)
        iot_class_list[i].uav = uav_class_list[uav_id]
    return iot_class_list



def calculate_new_solution(uav_xy_list, iot_bd_cate_list, iot_bd_A_prop_list, iot_bd_B_prop_list):
    
    UAV_class_list_temporary = last_UAV_class_list[0]
    x_list = [item[0] for item in uav_xy_list]
    y_list = [item[1] for item in uav_xy_list]

    for u in range(Num_UAV):
        
        UAV_class_list_temporary[u].x = x_list[u]
        UAV_class_list_temporary[u].y = y_list[u]

    
    IOT_class_A_list_temporary = IOT_class_list_A_o[0]
    IOT_class_B_list_temporary = IOT_class_list_B_o[0]

    
    iot_A_bd_list = [element * iot_bd_cate_list[0] * B for element in iot_bd_A_prop_list]
    iot_B_bd_list = [element * iot_bd_cate_list[1] * B for element in iot_bd_B_prop_list]

    
    for t in range(len(IOT_class_A_list_temporary)):
        IOT_class_A_list_temporary[t].bandwidth = iot_A_bd_list[t]
    for t in range(len(IOT_class_B_list_temporary)):
        IOT_class_B_list_temporary[t].bandwidth = iot_B_bd_list[t]

    
    IOT_capacity_list_A_new, _, _ = calculate_IOT_LEO_capacity(IOT_class_A_list_temporary, UAV_class_list_temporary)
    IOT_capacity_list_B_new, _, _ = calculate_IOT_LEO_capacity(IOT_class_B_list_temporary, UAV_class_list_temporary)

    
    iot_capacity_A_min = min(IOT_capacity_list_A_new)
    iot_capacity_B_min = min(IOT_capacity_list_B_new)

    
    ob_A_capacity = -iot_capacity_A_min
    ob_B_capacity = -iot_capacity_B_min

    return [ob_A_capacity, ob_B_capacity], IOT_capacity_list_A_new, IOT_capacity_list_B_new



def update_Z(uav_xy_new, uav_power_new, iot_bd_list, Z_last):
    
    objective_list_new = calculate_new_solution(uav_xy_new, uav_power_new, iot_bd_list)
    for o in range(Objective_num):
        if objective_list_new[o] < Z_last[o]:
            Z_last[o] = objective_list_new[o]
    return Z_last, objective_list_new



def update_Z_and_Znad(
        uav_xy_new, iot_bd_cate_prop_list_new, iot_a_bd_prop_list_new, iot_b_bd_prop_list_new, Z_last):
    
    objective_list_new, iot_new_cap_A, iot_new_cap_B = calculate_new_solution(
        uav_xy_new, iot_bd_cate_prop_list_new, iot_a_bd_prop_list_new, iot_b_bd_prop_list_new)
    for o in range(Objective_num):
        if objective_list_new[o] < Z_last[o]:
            Z_last[o] = objective_list_new[o]
        
        
    return Z_last, objective_list_new, iot_new_cap_A, iot_new_cap_B



def pareto_frontier(points):
    
    
    num_objectives = Objective_num
    
    pareto_front = []
    
    pareto_index = []
    
    for i in range(len(points)):
        dominated = False
        for j in range(len(points)):
            
            if all(points[j][k] <= points[i][k] for k in range(num_objectives)) and \
                    any(points[j][k] < points[i][k] for k in range(num_objectives)):
                dominated = True
                break
        
        if not dominated:
            pareto_front.append(points[i])
            pareto_index.append(i)
    
    return pareto_front, pareto_index



def plot_pareto_frontier(points, pareto_points):
    
    
    plt.scatter([p[0] for p in points], [p[1] for p in points], color='gray', label='All Points')
    
    plt.scatter([p[0] for p in pareto_points], [p[1] for p in pareto_points], color='blue', label='Pareto Frontier')
    
    plt.legend(loc='upper right')
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    
    plt.show()



def plot_pareto_frontier_save(points, pareto_points, description):
    
    
    plt.scatter([p[0] for p in points], [p[1] for p in points], color='gray', label='last epoch PF')
    
    plt.scatter([p[0] for p in pareto_points], [p[1] for p in pareto_points], color='blue', label='this epoch PF')
    
    plt.legend(loc='upper right')
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    
    path_figure = os.path.join(figures_dir, str(description) + '.png')
    plt.savefig(path_figure)
    plt.close()

def Tchebycheff_dist(w, f, z):
    return w * abs(f - z)

def Tchebycheff_dist_uni(w, f, z, z_nad):
    return w * abs((f - z) / (z_nad - z))


def cpt_tchbycheff(idx, uav_xy_list, iot_bd_cate_list, iot_bd_A_prop_list, iot_bd_B_prop_list, Z, Vector):
    
    distance_max = 0
    
    vi = Vector[idx]
    
    F_X, _, _ = calculate_new_solution(uav_xy_list, iot_bd_cate_list, iot_bd_A_prop_list, iot_bd_B_prop_list)
    for oi in range(Objective_num):
        fi = Tchebycheff_dist(vi[oi], F_X[oi], Z[oi])  
        if fi > distance_max:
            distance_max = fi  
    return distance_max



def cpt_tchbycheff_for_new(idx, obj_new_solution, Z, Vector):
    
    distance_max = 0
    
    vi = Vector[idx]
    
    F_X = obj_new_solution
    for oi in range(Objective_num):
        fi = Tchebycheff_dist(vi[oi], F_X[oi], Z[oi])  
        if fi > distance_max:
            distance_max = fi  
    return distance_max



def update_BTX(Bi, new_solution, Z, last_solutions_list, Vector, obj_new_solution):
    
    
    F_Y_set = []
    for j in Bi:
        
        uav_xy_list = last_solutions_list[j][0]
        iot_bd_cate_list = last_solutions_list[j][1]
        iot_a_bd_prop_list = last_solutions_list[j][2][0]
        iot_b_bd_prop_list = last_solutions_list[j][2][1]
        d_x = cpt_tchbycheff(j, uav_xy_list, iot_bd_cate_list, iot_a_bd_prop_list, iot_b_bd_prop_list, Z,
                             Vector)  

        d_y = cpt_tchbycheff_for_new(j, obj_new_solution, Z, Vector)  

        if d_y <= d_x:
            last_solutions_list[j] = new_solution  
            F_Y = obj_new_solution  
            F_Y_set.append(F_Y)  
            
            last_IOT_capacity_A[j] = IOT_capacity_A_new
            last_IOT_capacity_B[j] = IOT_capacity_B_new
        else:
            continue
    return last_solutions_list, F_Y_set, last_IOT_capacity_A, last_IOT_capacity_B



def is_dominate(F_X, F_Y):
    
    f = 0
    for xv, yv in zip(F_X, F_Y):
        if xv < yv:
            f = f + 1
        if xv > yv:
            return False
    if f != 0:
        return True
    return False



def get_kmeans_coordinates(data, k, max_iter=300):
    
    
    centers = random.sample(data, k)

    
    for i in range(max_iter):
        
        clusters = [[] for _ in range(k)]

        
        point_to_cluster = []
        for point in data:
            distances = [math.dist(point, center) for center in centers]
            min_distance = min(distances)
            min_index = distances.index(min_distance)
            clusters[min_index].append(point)
            point_to_cluster.append(min_index)

        
        for j in range(k):
            if clusters[j]:
                centers[j] = [sum(x) / len(clusters[j]) for x in zip(*clusters[j])]

    if k == 1:
        return centers[0], point_to_cluster

    return centers, point_to_cluster



def save_to_file(lt, filename):
    
    f = np.array(lt)
    np.save(filename, f)



def load_to_list(filename):
    f = np.load(filename)
    lt = f.tolist()
    return lt



def update_uav_connect_iot_num(uav_list, iot_list_A, iot_list_B):
    for uu in range(Num_UAV):
        connect_num = 0
        uav = uav_list[uu]
        for ii in range(len(iot_list_A)):
            iot = iot_list_A[ii]
            if iot.uav == uav:
                connect_num += 1
        for ii in range(len(iot_list_B)):
            iot = iot_list_B[ii]
            if iot.uav == uav:
                connect_num += 1
        uav_list[uu].connect_iot_num = connect_num
    return uav_list



def update_connection(uav_class_list_new, iot_class_list_new, Num_IOT):
    for ic in range(Num_IOT):
        iot_uav_dis_ = []
        for uc in range(Num_UAV):
            uav = uav_class_list_new[uc]
            iot = iot_class_list_new[ic]
            dis = math.sqrt((uav.x - iot.x) ** 2 + (uav.y - iot.y) ** 2 + (uav.z - iot.z) ** 2)
            iot_uav_dis_.append(dis)
        dis_min_value_ = min(iot_uav_dis_)
        dis_min_index_ = iot_uav_dis_.index(dis_min_value_)
        iot_class_list_new[ic].uav = uav_class_list_new[dis_min_index_]
    return iot_class_list_new



def cal_iot_capa_st(iot_capa_list):
    
    st_list = []
    for i in range(len(iot_capa_list)):
        iot_std_dev = statistics.stdev(iot_capa_list[i])
        st_list.append(iot_std_dev)
    return st_list



def calculate_HV(fy_list, ref_point):
    ref_point = np.array(ref_point)
    ind = HV(ref_point=ref_point)
    fy_list = np.array(fy_list)
    hv = ind(fy_list)
    return hv


"""———————————————————————————————————————————main function————————————————————————————————————————————"""
if __name__ == '__main__':
    """--------------------------------------初始化N个UAV class----------------------------------------------"""

    leo_z_list_o = []
    for i in range(Num_LEO):
        leo_z_list_o.append(Height_LEO)
    LEO_list_o = generate_LEO(Num_LEO, leo_z_list_o)

    """--------------------------------------初始化N个UAV class----------------------------------------------"""
    
    UAV_class_list_o = []
    
    relation_UAV_LEO_list_o = []

    for N_num in range(N):
        
        uav_x_list_o = []  
        uav_y_list_o = []
        uav_z_list_o = []
        uav_power_list_o = []  
        uav_bandwidth_list_o = []
        uav_connect_leo_list_o = []
        relation_UAV_LEO_o = []  
        uav_connect_iot_num_list = []  

        for i in range(Num_UAV):
            uav_x_list_o.append(random.uniform(0, Length_UAV_area))  
            uav_y_list_o.append(random.uniform(0, Width_UAV_area))
            uav_z_list_o.append(Height_UAV)  
            uav_power_list_o.append(Pmax_UAV_Transmit)  
            uav_bandwidth_list_o.append(B)  
            uav_connect_leo = random.choice(LEO_list_o)  
            uav_connect_leo_list_o.append(uav_connect_leo)  
            uav_connect_iot_num_list.append(None)

            relation_uav = []  
            for j in range(Num_LEO):
                
                if uav_connect_leo.id == j:
                    relation_uav.append(1)
                
                else:
                    relation_uav.append(0)
            relation_UAV_LEO_o.append(relation_uav)

        
        UAV_list_o = generate_UAV(Num_UAV, uav_x_list_o, uav_y_list_o, uav_z_list_o, uav_power_list_o,
                                  uav_bandwidth_list_o,
                                  uav_connect_leo_list_o, uav_connect_iot_num_list)
        
        UAV_class_list_o.append(UAV_list_o)
        
        relation_UAV_LEO_list_o.append(relation_UAV_LEO_o)
    """-------------------------------------初始化N个IOT A+B的bd list-----------------------------------------------"""
    
    IOT_bd_prop_list_AB_o = []
    IOT_bd_cut_prop_list_AB_o = []
    
    IOT_bd_prop_AB_o = []
    
    iot_bd_prop_by_all_A_list = []
    iot_bd_prop_by_all_B_list = []

    
    for n in range(N):
        iot_prop_bd_cut_A, iot_prop_bd_A = generate_iot_bd_proportion(Num_IOT_A)
        iot_prop_bd_cut_B, iot_prop_bd_B = generate_iot_bd_proportion(Num_IOT_B)
        IOT_bd_prop_list_AB_o.append([iot_prop_bd_A, iot_prop_bd_B])
        IOT_bd_cut_prop_list_AB_o.append([iot_prop_bd_cut_A, iot_prop_bd_cut_B])
        
        _, iot_bd_prop_AB = generate_iot_bd_proportion(Num_IOT_category)
        IOT_bd_prop_AB_o.append(iot_bd_prop_AB)

        
        iot_bd_prop_by_all_A = []
        iot_bd_prop_by_all_B = []
        for bd_prop in iot_prop_bd_A:
            iot_bd_prop_A = bd_prop * iot_bd_prop_AB[0] * B
            iot_bd_prop_by_all_A.append(iot_bd_prop_A)
        for bd_prop in iot_prop_bd_B:
            iot_bd_prop_B = bd_prop * iot_bd_prop_AB[1] * B
            iot_bd_prop_by_all_B.append(iot_bd_prop_B)
        iot_bd_prop_by_all_A_list.append(iot_bd_prop_by_all_A)
        iot_bd_prop_by_all_B_list.append(iot_bd_prop_by_all_B)

    """-------------------------------------初始化N个IOT A class list-----------------------------------------------"""
    
    IOT_class_list_A_o = []
    
    iot_x_list_A_u = load_to_list(r'D:\SAG_MOEAD_exp_fig\IOT_position_u\iot_x_list_A_u.npy')
    iot_y_list_A_u = load_to_list(r'D:\SAG_MOEAD_exp_fig\IOT_position_u\iot_y_list_A_u.npy')
    iot_z_list_A_u = load_to_list(r'D:\SAG_MOEAD_exp_fig\IOT_position_u\iot_z_list_A_u.npy')
    iot_rate_list_A_u = []
    relation_IOT_UAV_A_list_o = []  

    for n in range(Num_IOT_A):
        
        iot_rate_list_A_u.append(Transmit_rate_IOT_min_A)

    for N_num in range(N):
        
        iot_power_list_A_o = []
        iot_connect_uav_list_A_o = []
        relation_IOT_UAV_A_o = []  

        for n in range(Num_IOT_A):
            
            iot_power_list_A_o.append(P_IOT_Transmit_max_A)
            
            iot_connect_uav = random.choice(UAV_class_list_o[N_num])
            iot_connect_uav_list_A_o.append(iot_connect_uav)

            relation_iot_A = []  
            for j in range(Num_UAV):
                
                if iot_connect_uav.id == j:
                    relation_iot_A.append(1)
                
                else:
                    relation_iot_A.append(0)
            relation_IOT_UAV_A_o.append(relation_iot_A)

        
        IOT_list_o = generate_IOT(Num_IOT_A, iot_x_list_A_u, iot_y_list_A_u, iot_z_list_A_u, iot_power_list_A_o,
                                  iot_bd_prop_by_all_A_list[N_num],
                                  iot_rate_list_A_u, iot_connect_uav_list_A_o)
        
        IOT_class_list_A_o.append(IOT_list_o)
        
        relation_IOT_UAV_A_list_o.append(relation_IOT_UAV_A_o)

    """-------------------------------------Initialize N IOT B class lists-----------------------------------------------"""
    
    IOT_class_list_B_o = []
    
    iot_x_list_B_u = load_to_list(r'D:\SAG_MOEAD_exp_fig\IOT_position_u\iot_x_list_B_u.npy')
    iot_y_list_B_u = load_to_list(r'D:\SAG_MOEAD_exp_fig\IOT_position_u\iot_y_list_B_u.npy')
    iot_z_list_B_u = load_to_list(r'D:\SAG_MOEAD_exp_fig\IOT_position_u\iot_z_list_B_u.npy')
    iot_rate_list_B_u = []
    relation_IOT_UAV_B_list_o = []  

    for n in range(Num_IOT_B):
        
        iot_rate_list_B_u.append(Transmit_rate_IOT_min_B)

    for N_num in range(N):
        
        iot_power_list_B_o = []
        iot_connect_uav_list_B_o = []
        relation_IOT_UAV_B_o = []  

        for n in range(Num_IOT_B):
            
            iot_power_list_B_o.append(P_IOT_Transmit_max_B)
            
            iot_connect_uav = random.choice(UAV_class_list_o[N_num])
            iot_connect_uav_list_B_o.append(iot_connect_uav)
            relation_iot_B = []  
            for j in range(Num_UAV):
                
                if iot_connect_uav.id == j:
                    relation_iot_B.append(1)
                
                else:
                    relation_iot_B.append(0)
            relation_IOT_UAV_B_o.append(relation_iot_B)

        
        IOT_list_o = generate_IOT(Num_IOT_B, iot_x_list_B_u, iot_y_list_B_u, iot_z_list_B_u, iot_power_list_B_o,
                                  iot_bd_prop_by_all_B_list[N_num],
                                  iot_rate_list_B_u, iot_connect_uav_list_B_o)
        
        IOT_class_list_B_o.append(IOT_list_o)
        
        relation_IOT_UAV_B_list_o.append(relation_IOT_UAV_B_o)


    """-------------------------------Update the number of connected IOT based on the number of IOT connected to the UAV----------------------------------"""
    for nn in range(N):
        UAV_class_list_o[nn] = update_uav_connect_iot_num(UAV_class_list_o[nn], IOT_class_list_A_o[nn],
                                                          IOT_class_list_B_o[nn])

    """-------------------------------------------Merge A+B IOT class for N populations----------------------------------------"""
    IOT_class_list_AB_o = []
    for n in range(N):
        IOT_class_AB_o = IOT_class_list_A_o[n] + IOT_class_list_B_o[n]
        IOT_class_list_AB_o.append(IOT_class_AB_o)
    """-------------------------------------------------The solution of the initial population is obtained----------------------------------------------"""
    
    UAV_xy_population_list_o = []
    
    IOT_connection_population_list_o = []
    
    init_solution_population = []

    for nn in range(N):
        
        UAV_xy_list_o = []
        for uu in range(Num_UAV):
            UAV_xy_list_o.append([UAV_class_list_o[nn][uu].x, UAV_class_list_o[nn][uu].y])
        
        IOT_connection_AB_o = relation_IOT_UAV_A_list_o[nn] + relation_IOT_UAV_B_list_o[nn]

        UAV_xy_population_list_o.append(UAV_xy_list_o)
        
        init_solution_population.append([UAV_xy_list_o, IOT_bd_prop_AB_o[nn], IOT_bd_prop_list_AB_o[nn]])
    """-----------------------------------Calculate the smallest capacity of the two IOT's out of N initial parents--------------------------------"""
    
    IOT_capacity_min_population_A_o = []
    IOT_capacity_min_population_B_o = []
    
    IOT_capacity_population_A_o = []
    IOT_capacity_population_B_o = []

    for n in range(N):
        """The relevant parameters of the NTH individual are obtained"""
        
        IOT_list_n_A_o = IOT_class_list_A_o[n]
        IOT_list_n_B_o = IOT_class_list_B_o[n]
        
        UAV_list_n_o = UAV_class_list_o[n]

        """Initialization calculates the channel capacity of the initial IOT uplink channel in the nth individual (solved by two layers of signal-to-noise ratio)"""
        IOT_capacity_list_A_o, _, _ = calculate_IOT_LEO_capacity(IOT_list_n_A_o, UAV_list_n_o)
        IOT_capacity_list_B_o, _, _ = calculate_IOT_LEO_capacity(IOT_list_n_B_o, UAV_list_n_o)
        
        IOT_capacity_population_A_o.append(IOT_capacity_list_A_o)
        IOT_capacity_population_B_o.append(IOT_capacity_list_B_o)
        
        IOT_capacity_A_min = min(IOT_capacity_list_A_o)
        IOT_capacity_B_min = min(IOT_capacity_list_B_o)
        
        IOT_capacity_min_population_A_o.append(IOT_capacity_A_min)
        IOT_capacity_min_population_B_o.append(IOT_capacity_B_min)
        
        print("IOT_capacity_min_population_A_o", IOT_capacity_min_population_A_o)
        print("IOT_capacity_min_population_B_o", IOT_capacity_min_population_B_o)

    
    objective_capacity_min_A_o = IOT_capacity_min_population_A_o
    objective_capacity_min_B_o = IOT_capacity_min_population_B_o

    """-----------------------------------Plot data from an individual in the initial population-------------------------------------"""
    
    IOT_class_list_AB_show = IOT_class_list_AB_o[1]
    IOT_class_list_A_show = IOT_class_list_A_o[1]
    IOT_class_list_B_show = IOT_class_list_B_o[1]
    
    UAV_class_list_show = UAV_class_list_o[1]

    
    IOT_show_A_x = [obj.x for obj in IOT_class_list_A_show]
    IOT_show_A_y = [obj.y for obj in IOT_class_list_A_show]
    IOT_show_A_z = [obj.z for obj in IOT_class_list_A_show]

    IOT_show_B_x = [obj.x for obj in IOT_class_list_B_show]
    IOT_show_B_y = [obj.y for obj in IOT_class_list_B_show]
    IOT_show_B_z = [obj.z for obj in IOT_class_list_B_show]

    
    UAV_show_x = [obj.x for obj in UAV_class_list_show]
    UAV_show_y = [obj.y for obj in UAV_class_list_show]
    UAV_show_z = [obj.z for obj in UAV_class_list_show]

    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(IOT_show_A_x, IOT_show_A_y, IOT_show_A_z, c='r', marker='o')
    ax.scatter(IOT_show_B_x, IOT_show_B_y, IOT_show_B_z, c='b', marker='o')
    ax.scatter(UAV_show_x, UAV_show_y, UAV_show_z, c='g', marker='^')

    
    for iot in IOT_class_list_AB_show:
        if iot.uav is not None:
            uav = iot.uav
            ax.plot([iot.x, uav.x], [iot.y, uav.y], [iot.z, uav.z], linestyle='--', linewidth=0.5, c='k')

    
    plt.title('UAV and IOT in the first population before evolution', fontsize=15)

    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    
    plt.show()

    """---------------------------------Draws the target value of the initialized N individuals--------------------------------"""
    
    objective_capacity_min_opposite_A_o = [-capacity for capacity in objective_capacity_min_A_o]
    objective_capacity_min_opposite_B_o = [-capacity for capacity in objective_capacity_min_B_o]

    
    objective_list_combine_o = list(zip(objective_capacity_min_opposite_A_o, objective_capacity_min_opposite_B_o))

    
    plt.scatter(objective_capacity_min_opposite_A_o, objective_capacity_min_opposite_B_o)

    
    plt.title('The objective value of the initial population', fontsize=15)
    plt.xlabel("IOT-capacity-min-A-(opposite)")
    plt.ylabel("IOT-capacity-min-B-(opposite)")

    
    plt.show()

    """---------------------------------The initial pareto EP is obtained and the scatter plot is drawn--------------------------------"""
    
    Objective_PF_o, PF_index_o = pareto_frontier(objective_list_combine_o)
    Objective_PF_o = [list(t) for t in Objective_PF_o]
    
    plot_pareto_frontier(objective_list_combine_o, Objective_PF_o)
    
    save_to_file(Objective_PF_o, r'D:\SAG_MOEAD_exp_fig\SAG_MOEAD_8.2.nr\file\PF_o')
    
    IOT_capacity_population_A_PF_o = [IOT_capacity_population_A_o[index_o] for index_o in PF_index_o]
    IOT_capacity_population_B_PF_o = [IOT_capacity_population_B_o[index_o] for index_o in PF_index_o]
    save_to_file(IOT_capacity_population_A_PF_o,
                 r'D:\SAG_MOEAD_exp_fig\SAG_MOEAD_8.2.nr\file\IOT_capacity_population_A_PF_o')
    save_to_file(IOT_capacity_population_B_PF_o,
                 r'D:\SAG_MOEAD_exp_fig\SAG_MOEAD_8.2.nr\file\IOT_capacity_population_B_PF_o')

    """-------------------------------------Initializing an EP---------------------------------------------"""
    
    EP = []
    
    FY = []
    
    EP_IOT_A_capacity_list = []
    EP_IOT_B_capacity_list = []
    """---------------------------------The reference point z of the initial population is obtained---------------------------------------"""

    Z_o = get_Z(objective_list_combine_o)
    print("Z_o", Z_o)

    
    Z_nad_o = get_Znad(objective_list_combine_o)
    print("Z_nad_o", Z_nad_o)

    """-----------------------------------Generate weight vectors and weight vector fields----------------------------------"""
    
    weights = generate_weight_vectors(Objective_num, Weight_vector_num)
    
    neighbor_index = generate_neighbor_index(weights, Neighbor_num)

    print("weights", weights)
    print("neighbor_index", neighbor_index)

    """----------------------------------------Initialize the size of the adjustment step---------------------------------------------"""
    init_adjust_rate = 1
    """--------------------------------Store the standard deviation of the IOT capacity for each epoch----------------------------------"""
    IOT_capacity_avg_dev_list_A = []
    IOT_capacity_avg_dev_list_B = []
    """--------------------------------Stores the HV value of each epoch----------------------------------"""
    HV_list = []
    """----------------------------------------Update iteration---------------------------------------------"""
    for epoch in range(Evolution_epoch):
        print("---the ", epoch, " evolution---")
        
        adjust_rate = get_epoch_decay_rate(epoch, init_adjust_rate)
        print("adjust_rate:", adjust_rate)
        FY_epoch_set = []
        if epoch == 0:
            
            last_UAV_class_list = UAV_class_list_o
            
            last_IOT_class_AB_list = IOT_class_list_AB_o
            
            last_solution_list = init_solution_population
            
            last_object_list = objective_list_combine_o
            last_Z = Z_o
            last_Z_nad = Z_nad_o
            
            last_IOT_capacity_A = IOT_capacity_population_A_o
            last_IOT_capacity_B = IOT_capacity_population_B_o
            
            PF_last_epoch = []

        for i in range(N):
            New_solution = get_new_solution_new(i, neighbor_index, last_solution_list)

            Uav_xy_new = New_solution[0]
            Iot_bd_cate_prop_list_new = New_solution[1]
            IOT_bd_prop_A_new = New_solution[2][0]
            IOT_bd_prop_B_new = New_solution[2][1]
            last_Z, objective_new, IOT_capacity_A_new, IOT_capacity_B_new = update_Z_and_Znad(
                Uav_xy_new, Iot_bd_cate_prop_list_new, IOT_bd_prop_A_new, IOT_bd_prop_B_new, last_Z)

            last_solution_list, FY_set, last_IOT_capacity_A, last_IOT_capacity_B = update_BTX(
                neighbor_index[i], New_solution, last_Z, last_solution_list, weights, objective_new)

            
            EP.append(New_solution)
            FY.append(objective_new)

            
            EP_IOT_A_capacity_list.append(IOT_capacity_A_new)
            EP_IOT_B_capacity_list.append(IOT_capacity_B_new)

            
            FY, solution_index = pareto_frontier(FY)
            
            EP = [EP[index] for index in solution_index]
            
            EP_IOT_A_capacity_list = [EP_IOT_A_capacity_list[index] for index in solution_index]
            EP_IOT_B_capacity_list = [EP_IOT_B_capacity_list[index] for index in solution_index]

        "----------------------------------------The HV value corresponding to the EP obtained under each generation of epoch was calculated---------------------------------"
        Ref_point = [0, 0]
        HV_value = calculate_HV(FY, Ref_point)
        print("HV", HV_value)
        HV_list.append(HV_value)

        
        print("The number of eps in the round is", len(EP))
        "----------------------------------------Obtain the standard deviation of IOT capacity for each epoch iteration---------------------------------"
        IOT_capa_dev_list_A = [np.std(one_d_list) for one_d_list in EP_IOT_A_capacity_list]
        IOT_avg_dev_A = np.mean(IOT_capa_dev_list_A)
        IOT_capacity_avg_dev_list_A.append(IOT_avg_dev_A)
        print("Mean standard deviation of A-IOT-capa for EP in this round:", IOT_avg_dev_A)
        save_to_file(IOT_capacity_avg_dev_list_A,
                     r'D:\SAG_MOEAD_exp_fig\SAG_MOEAD_8.2.nr\file\IOT_capacity_avg_dev_list_A')
        IOT_capa_dev_list_B = [np.std(one_d_list) for one_d_list in EP_IOT_B_capacity_list]
        IOT_avg_dev_B = np.mean(IOT_capa_dev_list_B)
        IOT_capacity_avg_dev_list_B.append(IOT_avg_dev_B)
        print("Mean standard deviation of B-IOT-capa for EP in this round:", IOT_avg_dev_B)
        save_to_file(IOT_capacity_avg_dev_list_B,
                     r'D:\SAG_MOEAD_exp_fig\SAG_MOEAD_8.2.nr\file\IOT_capacity_avg_dev_list_B')

        
        plot_pareto_frontier_save(PF_last_epoch, FY, 'No.' + str(epoch + 1) + 'pareto-frontier-after-round-iteration')

        
        PF_last_epoch = FY

        
        if epoch == Evolution_epoch - 1:
            
            save_to_file(EP, r'D:\SAG_MOEAD_exp_fig\SAG_MOEAD_8.2.nr\file\EP_final')
            
            save_to_file(FY, r'D:\SAG_MOEAD_exp_fig\SAG_MOEAD_8.2.nr\file\PF_final')
            
            save_to_file(EP_IOT_A_capacity_list,
                         r'D:\SAG_MOEAD_exp_fig\SAG_MOEAD_8.2.nr\file\EP_IOT_A_capacity_list_final')
            save_to_file(EP_IOT_B_capacity_list,
                         r'D:\SAG_MOEAD_exp_fig\SAG_MOEAD_8.2.nr\file\EP_IOT_B_capacity_list_final')
            
            EP_IOT_A_capacity_list_df = pd.DataFrame(EP_IOT_A_capacity_list)
            EP_IOT_B_capacity_list_df = pd.DataFrame(EP_IOT_B_capacity_list)
            EP_IOT_A_capacity_list_df.to_excel(
                r'D:\SAG_MOEAD_exp_fig\SAG_MOEAD_8.2.nr\file\EP_IOT_A_capacity_list_final.xlsx', engine='openpyxl')
            EP_IOT_B_capacity_list_df.to_excel(
                r'D:\SAG_MOEAD_exp_fig\SAG_MOEAD_8.2.nr\file\EP_IOT_B_capacity_list_final.xlsx', engine='openpyxl')
            save_to_file(HV_list, r'D:\SAG_MOEAD_exp_fig\SAG_MOEAD_8.2.nr\file\HV_list')

