import numpy as np
import itertools
def logical_error_rate(p_phy, d):
    if d%2 == 1:
        return 0.1 * (p_phy / 0.01)**(0.5 * d + 0.5)
    else:
        return 0.05 * (d*d)/(d-1)/(d-1) * (p_phy / 0.01)**(0.5 * d)
    
def logical_error_retangle(dz, dx, p_phy):
    logical_error_z = 0.5 * (dx/dz) * logical_error_rate(p_phy, dz)
    logical_error_x = 0.5 * (dz/dx) * logical_error_rate(p_phy, dx)
    return logical_error_z, logical_error_x

def distillation_code_distance(eps, p_phy):
    for d in range(3,50,2):
        if 6.5 * d * logical_error_rate(p_phy, d) < eps:
            return d

# def logical_error_rate(code_distance, physical_error_rate=0.001):
#     return 0.1*(physical_error_rate/0.01)**((code_distance+1)/2)

# def code_distance(logical_error_rate, physical_error_rate=0.001):
#     exponent = np.log(logical_error_rate*10)/np.log(physical_error_rate/0.01)
#     return max(int(np.ceil(exponent))*2-1, 1)

def search_t_protocol(target_error_rate, p_phy):
    t_error_range = []
    initial_t_error_range = []
    if p_phy == 0.001:
        eps = [0.0004, 8e-05, 7e-06, 5e-06, 3e-06, 2e-07, 3e-08, 1e-08, 6e-09, 2e-09, 1e-09, 
             3.8850000000000016e-10, 3.304716553287983e-10, 1.2955e-12, 1.2178655692729767e-12, 4.405000000000001e-14,
             3.563400713436386e-14, 4.815000000000002e-15, 3.9081789802289295e-15, 1.2225000000000002e-15, 1.1252984389348027e-15, 
             1.3475000000000002e-16, 5.0750000000000023e-17, 4.03781224489796e-17]
        volume = [1000, 4000, 5000, 6000, 8000, 20000, 30000, 40000, 50000, 60000, 100000, 
             2199074, 2523632, 4685822, 5216768, 5827026, 6438000, 7131094, 7827712, 8609258, 9397136, 10272750, 10432750, 11317504]
    elif p_phy == 0.0005:
        eps=[1e-5, 8e-7,3e-07, 2e-07, 1e-08, 1e-09, 4e-11, 3.7260546875000005e-12, 3.2233414127423824e-12, 1.186650390625e-12, 
             1.1583162379535146e-12,2.9983642578125002e-15, 2.91156640625e-15, 3.871166992187501e-17, 3.3973240312071337e-17, 
             7.554645996093752e-20,6.170946930280958e-20]
        volume=[2000, 3500, 4000, 4500, 8000.0, 10000.0, 20000.0, 1621006, 1888000, 2185074, 2509632, 3688250.0, 4144784.0, 
                4645822.0, 5176768.0,7051094.0, 7747712.0]
    
    for i, t_error in enumerate(eps): 
        s = 8 * t_error ** 2
        if s < 1.5 * target_error_rate:
            t_error_range.append([eps[i],volume[i]])
        if len(t_error_range) >= 6:
            break
    for i, t_error in enumerate(eps): 
        s = t_error
        if s < 1.5 * target_error_rate:
            initial_t_error_range.append([eps[i],volume[i]])
        if len(initial_t_error_range) >= 6:
            break
    return t_error_range, initial_t_error_range #length = 6


def MEK(d, t_error, rl_1_error, ml_error, t_volume, rl_1_volume, ml_volume, p_phy): # Campbell-1603.04230
    # ml_error 输入态的error
    # rl_1_error R l-1 门的error（包含修正）
    pass_rate = 1 - 8 * t_error - 2 * ml_error - 0.5 * rl_1_error
    pass_rate += 96 * logical_error_rate(p_phy, d)
    
    error_rate = 8 * t_error ** 2 + ml_error ** 2 + rl_1_error / 4
    error_rate += 0.5 * logical_error_rate(p_phy, d)
    
    
    depth = 20 * d
    footprint = 2 * 16 * d ** 2
    
    volume = (footprint * depth + 8 * t_volume + 2 * ml_volume + rl_1_volume) / pass_rate / 2

    return error_rate, volume

def mek_level_one(d, t_error, initial_t_error, t_volume,initial_t_volume, ll, p_phy): # d, t_error, ll>=4
    ml_error = 7 / 15 * p_phy
    ml_volume = 0
    rl_1_error = initial_t_error
    rl_1_volume = initial_t_volume
    error_rates = [initial_t_error]
    volumes = [initial_t_volume]
    for l in range(4, ll+1):
        error_rate, volume = MEK(d, t_error, rl_1_error, ml_error, t_volume, rl_1_volume, ml_volume, p_phy)
        error_rates.append(error_rate)
        volumes.append(volume)
        rl_1_error = rl_1_error * 0.5 + error_rate
        rl_1_volume = rl_1_volume * 0.5 + volume
    return error_rates, volumes

def mek_level_two(d1,d2, t_error1, initial_t_error1, t_volume1,initial_t_volume1,t_error2, initial_t_error2, 
                  t_volume2, initial_t_volume2, ll, p_phy): 
    error_rates_one, volumes_one = mek_level_one(d1, t_error1, initial_t_error1, initial_t_volume1, t_volume1, 25, p_phy)#限定最大L=25
    error_rates = [initial_t_error2]
    volumes = [initial_t_volume2]
    rl_1_error = initial_t_error2
    rl_1_volume = initial_t_volume2
    for l in range(4, ll+1):
        ml_error = error_rates_one[l-3]
        ml_volume = volumes_one[l-3]
        error_rate, volume = MEK(d2, t_error2, rl_1_error, ml_error, t_volume2, rl_1_volume, ml_volume, p_phy)
        error_rates.append(error_rate)
        volumes.append(volume)
        rl_1_error = rl_1_error * 0.5 + error_rate
        rl_1_volume = rl_1_volume * 0.5 + volume
    return error_rates, volumes

def mek_level_three(d1,d2,d3, t_error1, initial_t_error1,t_volume1,initial_t_volume1,t_error2, initial_t_error2, 
                    t_volume2, initial_t_volume2,  t_error3, initial_t_error3, t_volume3, initial_t_volume3, ll, p_phy): 
    error_rates_two, volumes_two = mek_level_two(d1,d2, t_error1, initial_t_error1, initial_t_volume1, t_volume1,t_error2,
                                                 initial_t_error2, initial_t_volume2, t_volume2, 25, p_phy)#限定最大L=25
    error_rates = [initial_t_error3]
    volumes = [initial_t_volume3]
    rl_1_error = initial_t_error3
    rl_1_volume = initial_t_volume3
    for l in range(4, ll+1):
        ml_error = error_rates_two[l-3]
        ml_volume = volumes_two[l-3]
        error_rate, volume = MEK(d2, t_error2, rl_1_error, ml_error, t_volume2, rl_1_volume, ml_volume, p_phy)
        error_rates.append(error_rate)
        volumes.append(volume)
        rl_1_error = rl_1_error * 0.5 + error_rate
        rl_1_volume = rl_1_volume * 0.5 + volume
    return error_rates, volumes

def search_mek_level_two(ll, target_error_rate, p_phy):#搜索t_error，d用梯度下降
    parameters = ["fail",float("inf")] 
    t_error_range2, initial_t_error_range2 = search_t_protocol(target_error_rate, p_phy)
    t_error_range1, initial_t_error_range1 = search_t_protocol(target_error_rate**0.5, p_phy)
    for t_error1, t_volume1 in t_error_range1:
        for initial_t_error1, initial_t_volume1 in initial_t_error_range1:
            for t_error2, t_volume2 in t_error_range2:
                for initial_t_error2, initial_t_volume2 in initial_t_error_range2:
                    d_target = [41,41] #d1,d2
                    d_step = [1,1] #搜索步长
                    for i in range(len(d_target)):
                        d_range0 = [41,41]
                        for j in range(d_target[i]//d_step[i]):
                            d1,d2= d_range0
                            error_rates, volumes = mek_level_two(d1,d2, t_error1, initial_t_error1, t_volume1,initial_t_volume1,
                                                                 t_error2, initial_t_error2, t_volume2, initial_t_volume2, ll, p_phy)
                            if error_rates[-1] < target_error_rate:
                                d_target[i] = d_range0[i]
                                d_range0[i] -= d_step[i]
                            else:
                                break
                    d1,d2 = d_target
                    d_set = list(itertools.product([d1,d1+2],[d2,d2+2]))
                    for dd1, dd2 in d_set:
                        error_rates, volumes = mek_level_two(dd1,dd2, t_error1, initial_t_error1, t_volume1,initial_t_volume1,
                                                         t_error2, initial_t_error2, t_volume2, initial_t_volume2, ll, p_phy)
                        if error_rates[-1] < target_error_rate and volumes[-1] < parameters[-1]:
                                parameters = [dd1,dd2, t_error1, initial_t_error1, t_volume1,initial_t_volume1,
                                              t_error2, initial_t_error2, t_volume2, initial_t_volume2, error_rates[-1], volumes[-1]]
    return parameters
    
    
    
def search_mek_level_three(ll, target_error_rate, p_phy):#搜索t_error，d用梯度下降
    parameters = ["fail",float("inf")] 
    t_error_range3, initial_t_error_range3 = search_t_protocol(target_error_rate, p_phy)
    t_error_range2, initial_t_error_range2 = search_t_protocol(target_error_rate**0.5, p_phy)
    t_error_range1, initial_t_error_range1 = search_t_protocol((target_error_rate**0.5)**0.5, p_phy)
    for t_error1, t_volume1 in t_error_range1:
        for initial_t_error1, initial_t_volume1 in initial_t_error_range1:
            for t_error2, t_volume2 in t_error_range2:
                for initial_t_error2, initial_t_volume2 in initial_t_error_range2:
                    for t_error3, t_volume3 in t_error_range3:
                        for initial_t_error3, initial_t_volume3 in initial_t_error_range3:
                            d_target = [41,41,41] #d1,d2
                            d_step = [1,1,1] #搜索步长
                            for i in range(len(d_target)):
                                d_range0 = [41,41,41]
                                for j in range(d_target[i]//d_step[i]):
                                    d1,d2,d3= d_range0
                                    error_rates, volumes = mek_level_three(d1,d2,d3, t_error1, initial_t_error1,t_volume1,initial_t_volume1,
                                                                           t_error2, initial_t_error2, t_volume2, initial_t_volume2,  t_error3, 
                                                                           initial_t_error3, t_volume3, initial_t_volume3, ll, p_phy)
                                    if error_rates[-1] < target_error_rate:
                                        d_target[i] = d_range0[i]
                                        d_range0[i] -= d_step[i]
                                    else:
                                        break
                            d1,d2,d3 = d_target
                            d_set = list(itertools.product([d1,d1+2],[d2,d2+2],[d3,d3+2]))
                            for dd1, dd2, dd3 in d_set:
                                error_rates, volumes = mek_level_three(dd1,dd2,dd3, t_error1, initial_t_error1,t_volume1,initial_t_volume1,
                                                                           t_error2, initial_t_error2, t_volume2, initial_t_volume2,  t_error3, 
                                                                           initial_t_error3, t_volume3, initial_t_volume3, ll, p_phy)
                                if error_rates[-1] < target_error_rate and volumes[-1] < parameters[-1]:
                                        parameters = [dd1,dd2,dd3, t_error1, initial_t_error1,t_volume1,initial_t_volume1,
                                                                           t_error2, initial_t_error2, t_volume2, initial_t_volume2,  t_error3, 
                                                                           initial_t_error3, t_volume3, initial_t_volume3, error_rates[-1], volumes[-1]]
    return parameters

#T decomposition
def T_decomposition(ll, p_phy, infidelity):
    if p_phy == 0.001:
        eps = [0.0004, 8e-05, 7e-06, 5e-06, 3e-06, 2e-07, 3e-08, 1e-08, 6e-09, 2e-09, 1e-09, 
                 3.8850000000000016e-10, 3.304716553287983e-10, 1.2955e-12, 1.2178655692729767e-12, 4.405000000000001e-14,
                 3.563400713436386e-14, 4.815000000000002e-15, 3.9081789802289295e-15, 1.2225000000000002e-15, 1.1252984389348027e-15, 
                 1.3475000000000002e-16, 5.0750000000000023e-17, 4.03781224489796e-17]
        volume = [1000, 4000, 5000, 6000, 8000, 20000, 30000, 40000, 50000, 60000, 100000, 
                 2199074, 2523632, 4685822, 5216768, 5827026, 6438000, 7131094, 7827712, 8609258, 9397136, 10272750, 10432750, 11317504]
    elif p_phy == 0.0005:
        eps=[1e-5, 8e-7,3e-07, 2e-07, 1e-08, 1e-09, 4e-11, 3.7260546875000005e-12, 3.2233414127423824e-12, 1.186650390625e-12, 
                 1.1583162379535146e-12,2.9983642578125002e-15, 2.91156640625e-15, 3.871166992187501e-17, 3.3973240312071337e-17, 
                 7.554645996093752e-20,6.170946930280958e-20]
        volume=[2000, 3500, 4000, 4500, 8000.0, 10000.0, 20000.0, 1621006, 1888000, 2185074, 2509632, 3688250.0, 4144784.0, 
                    4645822.0, 5176768.0,7051094.0, 7747712.0]

    #infidelity = 1e-13
    output1, output2 = 0, float("inf")
    for l in range(3, 30):
        t_count = int (3 * (l - 1 - np.log(np.pi)/np.log(2)))
        for i,e in enumerate(eps):
            if e * t_count +  (np.pi/2**l)**2 <= infidelity * 2:
                if t_count * volume[i] < output2:
                    output1, output2 = e * t_count, t_count * volume[i]
    return output1, output2