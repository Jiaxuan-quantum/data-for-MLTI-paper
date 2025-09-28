import numpy as np
import itertools

def input_angle(k, beta):
    alpha = np.arctan(np.tan(beta)**(1/k))
    return alpha

def output_angle(k, alpha):
    beta = np.arctan(np.tan(alpha)**(k))
    return beta

def logical_error_rate(p_phy, d):
    if d%2 == 1:
        return 0.1 * (p_phy / 0.01)**(0.5 * d + 0.5)
    else:
        return 0.05 * (d*d)/(d-1)/(d-1) * (p_phy / 0.01)**(0.5 * d)
    
def logical_error_retangle(dz, dx, p_phy):
    logical_error_z = 0.5 * (dx/dz) * logical_error_rate(p_phy, dz)
    logical_error_x = 0.5 * (dz/dx) * logical_error_rate(p_phy, dx)
    return logical_error_z, logical_error_x
    
    
def pure_state_to_density_matrix(psi):
    """将纯态向量转换为密度矩阵"""
    # 确保纯态是归一化的
    norm = np.sum(np.abs(psi)**2)
    if not np.isclose(norm, 1.0):
        raise ValueError("纯态未归一化")
    return np.outer(psi, psi.conj())

# 定义泡利矩阵
I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
T_dag = np.array([[np.cos(np.pi/8), 1j*np.sin(np.pi/8)], [1j*np.sin(np.pi/8), np.cos(np.pi/8)]])
T = np.array([[np.cos(np.pi/8), -1j*np.sin(np.pi/8)], [-1j*np.sin(np.pi/8), np.cos(np.pi/8)]])

def single_qubit_error(rho, px, pz):
    """
    对单个量子比特的密度矩阵应用比特翻转和相位翻转错误信道
    
    参数:
        rho: 2x2 密度矩阵
        px: 比特翻转概率
        pz: 相位翻转概率
    
    返回:
        错误信道后的密度矩阵
    """
    # 计算四种情况的概率
    p_no_error = (1 - px) * (1 - pz)
    p_x_error = px * (1 - pz)
    p_z_error = (1 - px) * pz
    p_y_error = px * pz
    
    # 计算四种情况下的密度矩阵
    rho_no_error = rho
    rho_x_error = X @ rho @ X
    rho_z_error = Z @ rho @ Z
    rho_y_error = X @ Z @ rho @ Z @ X  # 等价于 Y @ rho @ Y (忽略全局相位)
    
    # 加权平均
    rho_error = (p_no_error * rho_no_error + 
                 p_x_error * rho_x_error + 
                 p_z_error * rho_z_error + 
                 p_y_error * rho_y_error)
    
    return rho_error



def transversal_injection(rho_input, k, pz, px):
    """
    对k个量子比特的密度矩阵进行测量操作
    
    参数:
        rho_list: 包含k个2x2密度矩阵的列表（初始乘积态）
        px: 每个量子比特的比特翻转概率
        pz: 每个量子比特的相位翻转概率
    
    返回:
        rho_sub: 测量后的量子态（2x2矩阵，基为|0...0>和|1...1>）
        error_rate: 错误率（1 - 测量成功概率）
    """
    
    # 对每个量子比特应用错误信道
    rho = single_qubit_error(rho_input, pz, px) #交换了X和Z基
    
    # 初始化矩阵元
    diag0 = 1.0  # <0...0|ρ|0...0>
    diag1 = 1.0  # <1...1|ρ|1...1>
    off_diag = 1.0  # <0...0|ρ|1...1>
    off_diag_conj = 1.0  # <1...1|ρ|0...0>
    
    # 计算所有量子比特的矩阵元乘积
    a = rho[0, 0]  # <0|ρ|0>
    b = rho[0, 1]  # <0|ρ|1>
    c = rho[1, 0]  # <1|ρ|0>
    d = rho[1, 1]  # <1|ρ|1>

    diag0 = a ** k
    diag1 = d ** k
    off_diag = b ** k
    off_diag_conj = c ** k
    
    # 计算测量成功概率
    success_prob = diag0 + diag1
    
    # 构建测量后的密度矩阵（在子空间{|0...0>, |1...1>}上）
    rho_output = np.array([
        [diag0, off_diag * (-1j) ** (1-k)],
        [off_diag_conj * (1j) ** (1-k), diag1]
    ]) / success_prob
    
    return rho_output, success_prob.real

import numpy as np

def fidelity(state, rho):
    """
    计算目标纯态和单比特密度矩阵之间的保真度
    
    参数:
    state: 目标纯态，是以下形式:
           1) 二维复数向量 (如 [1, 0] 表示 |0⟩)
    rho: 2x2 密度矩阵
    
    返回:
    fid: 保真度值 (0到1之间的实数)
    """
    psi = np.array(state)
    
    # 确保态向量是归一化的
    psi = psi / np.linalg.norm(psi)
    
    # 计算保真度 F(ψ, ρ) = ⟨ψ|ρ|ψ⟩
    fid = np.real(psi.conj().T @ rho @ psi)
    
    return fid

def reject_rate(k, p_phy):
    aa = [0.04565,0.10555,0.15128,0.20331,0.24488,0.29364,0.3271,0.36941,0.40219,0.4408,0.46968,0.50321,0.5298,0.55587,0.57861]
    bb = [0.0225,0.05313,0.07663,0.10789,0.13158,0.15992,0.18059,0.20646,0.22855,0.24987,0.27021,0.29296,0.31412,0.33496,0.35004]
    if p_phy == 0.001:
        return aa[k-1]
    elif p_phy == 0.0005:
        return bb[k-1]

def protocol_level_one(dx, dz, beta, p_phy):
    if dz % 3 == 0:
        k1 = dz // 3
    else:
        raise ValueError("dz should be 3k1 !")
    alpha = input_angle(k1, beta)
    time = 2
    space = 2 * dz * dx
    pass_rate = 1 - reject_rate(k1, p_phy) #不通过错误检测的概率 #粗略估计为15 k p
    
    #横向注入
    rho_input = pure_state_to_density_matrix(np.array([np.cos(alpha), 1j * np.sin(alpha)]))
    rho_output, succ_rate = transversal_injection(rho_input = rho_input, k = k1, pz = 2 / 15 * p_phy / pass_rate, px = 0)
    
    #两层逻辑错误率
    logical_z_rate, logical_x_rate = logical_error_retangle(dz, dx, p_phy)
    rho_output = single_qubit_error(rho_output, 2 * logical_z_rate, 2 * logical_x_rate)
    output_infidelity = fidelity(state = np.array([1j * np.sin(beta), np.cos(beta)]), rho = rho_output)

    spacetime_overhead = space * time / pass_rate / succ_rate
    
    return spacetime_overhead, output_infidelity, rho_output

def protocol_level_one_independent(dx, dz, beta, p_phy):
    if dz % 3 == 0:
        k1 = dz // 3
    else:
        raise ValueError("dz should be 3k1 !")
    alpha = input_angle(k1, beta)
    time = 2
    space = 2 * dz * dx
    pass_rate = 1 - reject_rate(k1, p_phy) #不通过错误检测的概率 #粗略估计为15 k p
    
    #横向注入
    rho_input = pure_state_to_density_matrix(np.array([np.cos(alpha), 1j * np.sin(alpha)]))
    rho_output, succ_rate = transversal_injection(rho_input = rho_input, k = k1, pz = 2 / 15 * p_phy / pass_rate, px = 0)
    
    #两层逻辑错误率
    logical_z_rate, logical_x_rate = logical_error_retangle(dz, dx, p_phy)
    rho_output = single_qubit_error(rho_output, 2 * logical_z_rate, 2 * logical_x_rate)
    
    #S门
    if k1 % 2 == 0:
        logical_z_rate, logical_x_rate = logical_error_retangle(dz, dx + dz, p_phy)
        logical_meas = dz * (logical_z_rate + logical_x_rate)
        rho_output = single_qubit_error(rho_output, dz * logical_z_rate + logical_meas, dz * logical_x_rate)
        
    output_infidelity = fidelity(state = np.array([1j * np.sin(beta), np.cos(beta)]), rho = rho_output)
    spacetime_overhead = space * time / pass_rate / succ_rate
    
    return spacetime_overhead, output_infidelity, rho_output

def optimized_spacetime_volume_level_one(target_error_rate, beta, p_phy):
    parameters = [3, 4, 1, float("inf")] #[dx, dz, succ_rate, output_infidelity, spacetime_overhead]
    for dx in range(3, 50):
        for dz in range(3, 48, 3):
            spacetime_overhead, output_infidelity, rho_output = protocol_level_one_independent(dx, dz, beta, p_phy)
            if output_infidelity < target_error_rate:
                if spacetime_overhead < parameters[-1]:
                    parameters = [dx, dz, output_infidelity, spacetime_overhead]
    #限制条件 e1 < target_error_rate
    #最小化时空开销（搜索dz,dx）
    dx, dz, output_infidelity, spacetime_overhead = parameters
    return dx, dz, output_infidelity, spacetime_overhead

def optimized_infidelity_level_one(beta, p_phy):
    infidelity = 1
    dx = 99
    for dz in range(3,48,3):
        spacetime_overhead, output_infidelity, rho_output = protocol_level_one(dx, dz, beta, p_phy)
        if output_infidelity < infidelity:
            infidelity = output_infidelity
    return dz, infidelity

def distillation_code_distance(eps, p_phy):
    for d in range(3,48,2):
        if 6.5 * d * logical_error_rate(p_phy, d) < eps:
            return d

# def t_protocol(t_error_approx, p_phy): # Gidney-2409.17595
#     if p_phy >= 0.001:
#         #cultivation
#         eps = [0.0004, 8e-05, 7e-06, 5e-06, 3e-06, 2e-07, 3e-08, 1e-08, 6e-09, 2e-09, 1e-09, 
#          3.8850000000000016e-10, 3.304716553287983e-10, 1.2955e-12, 1.2178655692729767e-12, 4.405000000000001e-14,
#          3.563400713436386e-14, 4.815000000000002e-15, 3.9081789802289295e-15, 1.2225000000000002e-15, 1.1252984389348027e-15, 
#          1.3475000000000002e-16, 5.0750000000000023e-17, 4.03781224489796e-17]
#         volume = [1000, 4000, 5000, 6000, 8000, 20000, 30000, 40000, 50000, 60000, 100000, 
#          2199074, 2523632, 4685822, 5216768, 5827026, 6438000, 7131094, 7827712, 8609258, 9397136, 10272750, 10432750, 11317504]
#         # 以下二级蒸馏 234d^3
# #         for j in range(4, 11):
# #             e1 = 28 * eps[j]**2
# #             d = distillation_code_distance(eps = e1, p_phy = p_phy)
# #             for d_test in [d, d+1]:
# #                 eps.append(e1 + 6.5 * d_test * logical_error_rate(p_phy, d_test))
# #                 volume.append(234 * d_test ** 3+4*)
#         #print(eps)
#         #print(volume)
#         #查找t_error_approx
#         for i, e in enumerate(eps):
#             if e <= t_error_approx:
#                 return e, volume[i]
#     #elif p_phy >= 0.0005:
def search_t_protocol(target_error_rate, k, gamma, p_phy):
    search_range = []
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
        s = t_error * k * abs(gamma)**(2*(1-1/k))
        if s < 1.5 * target_error_rate:
            search_range.append([eps[i],volume[i]])
        if len(search_range) >= 6:
            break
    return search_range #length = 5

def protocol_level_two(dz2,dx2,dz1,dx1,k2, gamma, p_phy, t_error, t_volume):
    beta = input_angle(k2, gamma) - np.pi/8
    spacetime_overhead1, output_infidelity1, rho_output1 = protocol_level_one(dx1, dz1, abs(beta), p_phy) 
    rho_input = rho_output1
    
    #T是方形 dz1* dz1 
        # 0.5概率需要S修正(T+S)
    logical_z_rate1 = 2 * (dz1) * logical_error_retangle(dz1, dx1+dz1+1, p_phy)[0] #两次测量的Z error
    logical_x_rate1 = 2 * logical_error_retangle(dz1, dx1, p_phy)[1]  + 2 * logical_error_retangle(dz1, dz1, p_phy)[1]
    #两次测量仅考虑第一轮的X #后面的XX（可忽略）
    
    #d/2 Y测量中，第一个比特idle
    logical_z_rate1 += (dz1//2) * logical_error_retangle(dz1, dx1, p_phy)[0]
    logical_x_rate1 += (dz1//2) * logical_error_retangle(dz1, dx1, p_phy)[1]
    
    rho_input1 = single_qubit_error(rho_input, logical_z_rate1, logical_x_rate1)
    
    logical_meas = (dz1) * logical_error_retangle(dz1, dx1+dz1+1, p_phy)[0] + (dz1) * logical_error_retangle(dz1, dx1+dz1+1, p_phy)[1] 
    #测量error，不大于memory error，一个等价于T之前的xERROR，另一个等价于S之前（也即T之前）的Zerror 
    logical_meas2 = (dz1//2) * logical_error_retangle(dz1, dz1, p_phy)[0] + (dz1//2) * logical_error_retangle(dz1, dz1, p_phy)[1]
    #d/2 Y测量受XZ影响，产生Z error
    
    rho_input1 = single_qubit_error(rho_input1, logical_meas + logical_meas2, logical_meas)
    
        # 0.5概率不需要S(T+idle)
    
    logical_z_rate1 = dz1 * (logical_error_retangle(dz1, dx1+dz1+1, p_phy)[0] + 1.5 * dz1 * logical_error_retangle(dz1, dx1, p_phy)[0]) #所有Z error
    logical_x_rate1 = 1 * logical_error_retangle(dz1, dx1, p_phy)[1]  + logical_error_retangle(dz1, dz1, p_phy)[1]
    logical_x_rate1 += 1.5 * dz1 * logical_error_retangle(dz1, dx1, p_phy)[1]
    #第一轮的X(计算态+T态) #后面的XX（可忽略） + 1.5 dz1轮idle
    
    rho_input2 = single_qubit_error(rho_input, logical_z_rate1, logical_x_rate1)
    
    logical_meas = (dz1) * logical_error_retangle(dz1, dx1+dz1+1, p_phy)[0] + (dz1) * logical_error_retangle(dz1, dx1+dz1+1, p_phy)[1] 
    #测量error，等价于T之前的xERROR
    
    rho_input2 = single_qubit_error(rho_input2, 0, logical_meas)
    
    #叠加
    rho_input = rho_input1 * 0.5 + rho_input2 * 0.5
    
    # T(dag)门
    if beta > 0:
        rho_input = T_dag@rho_input@T
    else:
        rho_input = T_dag@Z@rho_input@Z@T
    # T态 error
    rho_input = single_qubit_error(rho_input, t_error, 0)

    # detection error 
    logical_z_rate2 = 1 * logical_error_retangle(dz2, dx2, p_phy)[0] #只有第一轮会影响测量结果
    logical_x_rate2 = (dz2 + 2) * logical_error_retangle(dz2, dx2, p_phy)[1]
    #transversal_injection
    rho_output, succ_rate = transversal_injection(rho_input = rho_input, k = k2, 
                                                  pz = logical_z_rate2, px = logical_x_rate2)
    
    logical_z_rate = (dz2 + 1) * logical_error_retangle(k2 * dz2 + k2 -1, dx2, p_phy)[0] #合并后ZL扩大为 k2*dz2 + k2 -1，后dz2+1轮
    rho_output = single_qubit_error(rho_output, logical_z_rate, 0)
    output_infidelity = fidelity(state = np.array([1j * np.sin(gamma), np.cos(gamma)]), rho = rho_output)
    
    spacetime_overhead = 2 * dz1**2 * (1.75*dz1+2.5*dx1+1.75) * k2  + (dz2 + 2) * 2 * dx2 * (k2 * dz2 + k2 -1) #两层门 + 检测
    spacetime_overhead += k2 * t_volume + k2 * spacetime_overhead1
    spacetime_overhead = spacetime_overhead / succ_rate
    
    return spacetime_overhead, output_infidelity, rho_output


def optimized_infidelity_level_two(gamma, p_phy):
    dx1,dz2,dx2 = 99,99,99
    t_error = 0 
    t_volume = 0 
    infidelity = 1
    k2_output = 0
    dz1_output = 0
    for k2 in range(2,40):
        beta = input_angle(k2, gamma)-np.pi/8
        if abs(beta) > np.pi/16:
            continue
        for dz1 in range(3,48,3):
            spacetime_overhead, output_infidelity, rho_output = protocol_level_two(dz2,dx2,dz1,dx1,k2,gamma,p_phy,t_error,t_volume)
            if output_infidelity < infidelity:
                infidelity = output_infidelity
                k2_output = k2
                dz1_output = dz1
    return infidelity, k2_output, dz1_output

def protocol_level_three(k3, dz3, dx3, dz2,dx2,dz1,dx1,k2,delta, p_phy, t_error1, t_volume1, t_error2, t_volume2):
    gamma = input_angle(k3, delta) - np.pi/8
    spacetime_overhead2, output_infidelity2, rho_output2 = protocol_level_two(dz2,dx2,dz1,dx1,k2, abs(gamma), p_phy, t_error1, t_volume1)
    rho_input = rho_output2
    #T是方形 dz2* dz2 
        # 0.5概率需要S修正(T+S)
    logical_z_rate1 = 2 * (dz2) * logical_error_retangle(dz2, dx2+dz2+1, p_phy)[0] #两次测量的Z error
    logical_x_rate1 = 2 * logical_error_retangle(dz2, dx2, p_phy)[1]  + 2 * logical_error_retangle(dz2, dz2, p_phy)[1]
    #两次测量仅考虑第一轮的X #后面的XX（可忽略）
    
    #d/2 Y测量中，第一个比特idle
    logical_z_rate1 += (dz2//2) * logical_error_retangle(dz2, dx2, p_phy)[0]
    logical_x_rate1 += (dz2//2) * logical_error_retangle(dz2, dx2, p_phy)[1]
    
    rho_input1 = single_qubit_error(rho_input, logical_z_rate1, logical_x_rate1)
    
    logical_meas = (dz2) * logical_error_retangle(dz2, dx2+dz2+1, p_phy)[0] + (dz2) * logical_error_retangle(dz2, dx2+dz2+1, p_phy)[1] 
    #测量error，不大于memory error，一个等价于T之前的xERROR，另一个等价于S之前（也即T之前）的Zerror 
    logical_meas2 = (dz2//2) * logical_error_retangle(dz2, dz2, p_phy)[0] + (dz2//2) * logical_error_retangle(dz2, dz2, p_phy)[1]
    #d/2 Y测量受XZ影响，产生Z error
    
    rho_input1 = single_qubit_error(rho_input1, logical_meas + logical_meas2, logical_meas)
    
        # 0.5概率不需要S(T+idle)
    
    logical_z_rate1 = dz2 * (logical_error_retangle(dz2, dx2+dz2+1, p_phy)[0] + 1.5 * dz2 * logical_error_retangle(dz2, dx2, p_phy)[0]) #所有Z error
    logical_x_rate1 = 1 * logical_error_retangle(dz2, dx2, p_phy)[1]  + logical_error_retangle(dz2, dz2, p_phy)[1]
    logical_x_rate1 += 1.5 * dz2 * logical_error_retangle(dz2, dx2, p_phy)[1]
    #第一轮的X(计算态+T态) #后面的XX（可忽略） + 1.5 dz2轮idle
    
    rho_input2 = single_qubit_error(rho_input, logical_z_rate1, logical_x_rate1)
    
    logical_meas = (dz2) * logical_error_retangle(dz2, dx2+dz2+1, p_phy)[0] + (dz2) * logical_error_retangle(dz2, dx2+dz2+1, p_phy)[1] 
    #测量error，等价于T之前的xERROR
    
    rho_input2 = single_qubit_error(rho_input2, 0, logical_meas)
    
    #叠加
    rho_input = rho_input1 * 0.5 + rho_input2 * 0.5
    
    # T(dag)门
    if gamma > 0:
        rho_input = T_dag@rho_input@T
    else:
        rho_input = T_dag@Z@rho_input@Z@T
    # T态 error
    rho_input = single_qubit_error(rho_input, t_error2, 0)
    
    # detection error 
    logical_z_rate2 = 1/2 * (dz3 + 2 - 1) * logical_error_retangle(dz3, dx3, p_phy)[0] #前一半会影响测量结果, 后一半可被纠正
    logical_x_rate2 = (dz3 + 2) * logical_error_retangle(dz3, dx3, p_phy)[1]

    #transversal_injection
    rho_output, succ_rate = transversal_injection(rho_input = rho_input, k = k3, 
                                                  pz = logical_z_rate2, px = logical_x_rate2)
    logical_z_rate = (dz3 + 1) * logical_error_retangle(k3 * dz3 + k3 -1, dx3, p_phy)[0] #合并后ZL扩大为 k3*dz3 + k3 -1
    rho_output = single_qubit_error(rho_output, logical_z_rate, 0)
    output_infidelity = fidelity(state = np.array([1j * np.sin(delta), np.cos(delta)]), rho = rho_output)
    
    spacetime_overhead = 2 * dz2**2 * (1.75*dz2+2.5*dx2+1.75) * k3  + (dz3 + 2) * 2 * dx3 * (k3 * dz3 + k3 -1) #两层门 + 检测
    spacetime_overhead += k3 * t_volume2 + k3 * spacetime_overhead2
    spacetime_overhead = spacetime_overhead / succ_rate
    
    return spacetime_overhead, output_infidelity, rho_output



def protocol_level_four(k4,dz4,dx4,k3,dz3,dx3,dz2,dx2,dz1,dx1,k2,theta,p_phy,t_error1,t_volume1,t_error2,t_volume2,t_error3,t_volume3):
    delta = input_angle(k4, theta) - np.pi/8
    spacetime_overhead3, output_infidelity3, rho_output3 = protocol_level_three(
        k3, dz3, dx3, dz2,dx2,dz1,dx1,k2,abs(delta), p_phy, t_error1, t_volume1, t_error2, t_volume2)
    rho_input = rho_output3
    #T是方形 dz3* dz3 
        # 0.5概率需要S修正(T+S)
    logical_z_rate1 = 2 * (dz3) * logical_error_retangle(dz3, dx3+dz3+1, p_phy)[0] #两次测量的Z error
    logical_x_rate1 = 2 * logical_error_retangle(dz3, dx3, p_phy)[1]  + 2 * logical_error_retangle(dz3, dz3, p_phy)[1]
    #两次测量仅考虑第一轮的X #后面的XX（可忽略）
    
    #d/2 Y测量中，第一个比特idle
    logical_z_rate1 += (dz3//2) * logical_error_retangle(dz3, dx3, p_phy)[0]
    logical_x_rate1 += (dz3//2) * logical_error_retangle(dz3, dx3, p_phy)[1]
    
    rho_input1 = single_qubit_error(rho_input, logical_z_rate1, logical_x_rate1)
    
    logical_meas = (dz3) * logical_error_retangle(dz3, dx3+dz3+1, p_phy)[0] + (dz3) * logical_error_retangle(dz3, dx3+dz3+1, p_phy)[1] 
    #测量error，不大于memory error，一个等价于T之前的xERROR，另一个等价于S之前（也即T之前）的Zerror 
    logical_meas2 = (dz3//2) * logical_error_retangle(dz3, dz3, p_phy)[0] + (dz3//2) * logical_error_retangle(dz3, dz3, p_phy)[1]
    #d/2 Y测量受XZ影响，产生Z error
    
    rho_input1 = single_qubit_error(rho_input1, logical_meas + logical_meas2, logical_meas)
    
        # 0.5概率不需要S(T+idle)
    
    logical_z_rate1 = dz3 * (logical_error_retangle(dz3, dx3+dz3+1, p_phy)[0] + 1.5 * dz3 * logical_error_retangle(dz3, dx3, p_phy)[0]) #所有Z error
    logical_x_rate1 = 1 * logical_error_retangle(dz3, dx3, p_phy)[1]  + logical_error_retangle(dz3, dz3, p_phy)[1]
    logical_x_rate1 += 1.5 * dz3 * logical_error_retangle(dz3, dx3, p_phy)[1]
    #第一轮的X(计算态+T态) #后面的XX（可忽略） + 1.5 dz3轮idle
    
    rho_input2 = single_qubit_error(rho_input, logical_z_rate1, logical_x_rate1)
    
    logical_meas = (dz3) * logical_error_retangle(dz3, dx3+dz3+1, p_phy)[0] + (dz3) * logical_error_retangle(dz3, dx3+dz3+1, p_phy)[1] 
    #测量error，等价于T之前的xERROR
    
    rho_input2 = single_qubit_error(rho_input2, 0, logical_meas)
    
    #叠加
    rho_input = rho_input1 * 0.5 + rho_input2 * 0.5
    
    # T(dag)门
    if delta > 0:
        rho_input = T_dag@rho_input@T
    else:
        rho_input = T_dag@Z@rho_input@Z@T
    # T态 error
    rho_input = single_qubit_error(rho_input, t_error3, 0)
    
    # detection error 
    logical_z_rate2 = 1/2 * (dz4 + 2 - 1) * logical_error_retangle(dz4, dx4, p_phy)[0] #前一半会影响测量结果, 后一半可被纠正
    logical_x_rate2 = (dz4 + 2) * logical_error_retangle(dz4, dx4, p_phy)[1]

    #transversal_injection
    rho_output, succ_rate = transversal_injection(rho_input = rho_input, k = k4, 
                                                  pz = logical_z_rate2, px = logical_x_rate2)
    logical_z_rate = (dz4 + 1) * logical_error_retangle(k4 * dz4 + k4 -1, dx4, p_phy)[0] #合并后ZL扩大为 k3*dz3 + k3 -1
    rho_output = single_qubit_error(rho_output, logical_z_rate, 0)
    output_infidelity = fidelity(state = np.array([1j * np.sin(theta), np.cos(theta)]), rho = rho_output)
    
    spacetime_overhead = 2 * dz3**2 * (1.75*dz3+2.5*dx3+1.75)* k4  + (dz4 + 2) * 2 * dx4 * (k4 * dz4 + k4 -1) #两层门 + 检测
    spacetime_overhead += k4 * t_volume3 + k4 * spacetime_overhead3
    spacetime_overhead = spacetime_overhead / succ_rate
    
    return spacetime_overhead, output_infidelity, rho_output


def optimized_spacetime_volume_level_two(target_error_rate, gamma, p_phy):
    #[k2,dz2,dx2,dz1,dx1, t_error, t_volume, output_infidelity, spacetime_overhead]
    parameters = [2,2,2,1,1,0,0,1,float("inf")] 
    for k2 in range(2,30):
        beta = input_angle(k2, gamma)-np.pi/8
        if abs(beta) > np.pi/16:
            continue
        for dz1 in range(3,41,3):
            t_range = search_t_protocol(target_error_rate= target_error_rate, k=k2, gamma=gamma, p_phy=p_phy)
            for t_error1, t_volume1 in t_range:
                d_target = [41,41,41] #dx1,dz2,dx2
                d_step = [1,1,1] #搜索步长 也可设为 d_step = [2,2,2]
                for i in range(len(d_target)):
                    d_range0 = [41,41,41]
                    for j in range(d_target[i]//d_step[i]):
                        dx1,dz2,dx2 = d_range0
                        spacetime_overhead, output_infidelity, rho_output = protocol_level_two(
                            dz2,dx2,dz1,dx1,k2, gamma, p_phy, t_error1, t_volume1)
                        if output_infidelity < target_error_rate:
                            d_target[i] = d_range0[i]
                            d_range0[i] -= d_step[i]
                        else:
                            break
                dx1,dz2,dx2 = d_target
                d_set = list(itertools.product([dx1,dx1+1],
                                               [dz2,dz2+1],[dx2,dx2+1]))
                for ddx1,ddz2,ddx2 in d_set:
                    spacetime_overhead, output_infidelity, rho_output = protocol_level_two(
                            ddz2,ddx2,dz1,ddx1,k2,gamma, p_phy, t_error1, t_volume1)
                    if output_infidelity < target_error_rate and spacetime_overhead < parameters[-1]:
                        parameters = [k2,dz1,ddx1,ddz2,ddx2, t_error1, t_volume1, output_infidelity, spacetime_overhead]
    return parameters

def optimized_infidelity_level_three(delta, p_phy):
    dx1,dz2,dx2, dz3, dx3 = 99,99,99,99,99
    parameters = [3,3,3,2,2,2,1,1,0,0,1,float("inf")] 
    for k3, k2 in list(itertools.product(range(2,30),range(2,30))):
        gamma = input_angle(k3, delta) - np.pi/8
        beta = input_angle(k2, abs(gamma))-np.pi/8
        if abs(beta) > np.pi/16 or abs(gamma) > np.pi/16:
            continue
        for dz1 in range(3,48,3):
            t_error1, t_volume1 = 0, 0
            t_error2, t_volume2 = 0, 0
            spacetime_overhead, output_infidelity, rho_output = protocol_level_three(
                                k3, dz3, dx3, dz2,dx2,dz1,dx1,k2,delta, p_phy, t_error1, t_volume1, t_error2, t_volume2)
            if output_infidelity < parameters[-2]:
                parameters = [k3, dz3, dx3, k2,dz1,dx1,dz2,dx2, t_error1, t_volume1, t_error2, 
                                t_volume2, output_infidelity, spacetime_overhead]
    return parameters[-2]

def optimized_spacetime_volume_level_three(target_error_rate, delta, p_phy):
    #[k3,dz3,dx3,k2,dz2,dx2,dz1,dx1, t_error, t_volume, output_infidelity, spacetime_overhead]
    parameters = [3,3,3,2,2,2,1,1,0,0,1,float("inf")] 
    for k3, k2 in list(itertools.product(range(2,30),range(2,30))):
        gamma = input_angle(k3, delta) - np.pi/8
        beta = input_angle(k2, abs(gamma))-np.pi/8
        if abs(beta) > np.pi/16 or abs(gamma) > np.pi/16:
            continue
        t_range2 = search_t_protocol(target_error_rate= target_error_rate, k=k3, gamma=delta, p_phy=p_phy)
        t_range1 = search_t_protocol(target_error_rate= target_error_rate/k3/abs(delta)**(2-2/k3), k=k2, gamma=abs(gamma), p_phy=p_phy)
        for dz1 in range(3,41,3):
            for t_error1, t_volume1 in t_range1:
                for t_error2, t_volume2 in t_range2:
                    d_target = [41,41,41,41,41] #dx1,dz2,dx2,dz3,dx3
                    d_step = [1,1,1,1,1] #搜索步长 也可设为 d_step = [2,2,2,2,2]
                    for i in range(len(d_target)):
                        d_range0 = [41,41,41,41,41]
                        for j in range(d_target[i]//d_step[i]):
                            dx1,dz2,dx2,dz3,dx3 = d_range0
                            spacetime_overhead, output_infidelity, rho_output = protocol_level_three(
                                k3, dz3, dx3, dz2,dx2,dz1,dx1,k2,delta, p_phy, t_error1, t_volume1, t_error2, t_volume2)
                            if output_infidelity < target_error_rate:
                                d_target[i] = d_range0[i]
                                d_range0[i] -= d_step[i]
                            else:
                                break
                    dx1,dz2,dx2,dz3,dx3 = d_target
                    d_set = list(itertools.product([dx1,dx1+1],
                                                   [dz2,dz2+1],[dx2,dx2+1],
                                                   [dz3,dz3+1],[dx3,dx3+1]))
                    for ddx1,ddz2,ddx2,ddz3,ddx3 in d_set:
                        spacetime_overhead, output_infidelity, rho_output = protocol_level_three(
                                k3, ddz3, ddx3, ddz2,ddx2,dz1,ddx1,k2,delta, p_phy, t_error1, t_volume1, t_error2, t_volume2)
                        if output_infidelity < target_error_rate and spacetime_overhead < parameters[-1]:
                            parameters = [k3, ddz3, ddx3, k2,dz1,ddx1,ddz2,ddx2, t_error1, t_volume1, t_error2, 
                                          t_volume2, output_infidelity, spacetime_overhead]
    return parameters

def optimized_spacetime_volume_level_four(target_error_rate, theta, p_phy):
    #[k4,dz4,dx4,k3,dz3,dx3,k2,dz2,dx2,dz1,dx1, t_error, t_volume, output_infidelity, spacetime_overhead]
    parameters = [float("inf")] 
    for k4, k3, k2 in list(itertools.product(range(2,25),range(2,25),range(2,25))):
        delta = input_angle(k4, theta) - np.pi/8
        gamma = input_angle(k3, abs(delta)) - np.pi/8
        beta = input_angle(k2, abs(gamma))-np.pi/8
        if abs(beta) > np.pi/16 or abs(gamma) > np.pi/16 or abs(delta) > np.pi/16:
            continue
        t_range3 = search_t_protocol(target_error_rate= target_error_rate, k=k4, gamma=theta, p_phy=p_phy)
        t_range2 = search_t_protocol(target_error_rate= target_error_rate/k4/abs(theta)**(2-2/k4), k=k3, gamma=abs(delta), p_phy=p_phy)
        t_range1 = search_t_protocol(target_error_rate= target_error_rate/k3/abs(delta)**(2-2/k3)/k4/abs(theta)**(2-2/k4), k=k2, gamma=abs(gamma), p_phy=p_phy)
        for dz1 in range(3,37,3):
            for t_error3, t_volume3 in t_range3:
                for t_error1, t_volume1 in t_range1:
                    for t_error2, t_volume2 in t_range2:
                        d_target = [37,37,37,37,37,37,37] #dx1,dz2,dx2,dz3,dx3
                        d_step = [1,1,1,1,1,1,1] #搜索步长 也可设为 d_step = [2,2,2,2,2,2,2]
                        for i in range(len(d_target)):
                            d_range0 = [37,37,37,37,37,37,37]
                            for j in range(d_target[i]//d_step[i]):
                                dx1,dz2,dx2,dz3,dx3,dz4,dx4 = d_range0
                                spacetime_overhead, output_infidelity, rho_output = protocol_level_four(
                                    k4,dz4,dx4,k3,dz3,dx3,dz2,dx2,dz1,dx1,k2,theta,p_phy,t_error1,t_volume1,
                                    t_error2,t_volume2,t_error3,t_volume3)
                                if output_infidelity < target_error_rate:
                                    d_target[i] = d_range0[i]
                                    d_range0[i] -= d_step[i]
                                else:
                                    break
                        dx1,dz2,dx2,dz3,dx3,dz4,dx4 = d_target
                        d_set = list(itertools.product([dx1,dx1+1],
                                                       [dz2,dz2+1],[dx2,dx2+1],
                                                       [dz3,dz3+1],[dx3,dx3+1],
                                                      [dz4,dz4+1],[dx4,dx4+1]))
                        for ddx1,ddz2,ddx2,ddz3,ddx3,ddz4,ddx4 in d_set:
                            spacetime_overhead, output_infidelity, rho_output = protocol_level_four(
                                    k4,ddz4,ddx4,k3,ddz3,ddx3,ddz2,ddx2,dz1,ddx1,k2,theta,p_phy,t_error1,t_volume1,
                                    t_error2,t_volume2,t_error3,t_volume3)
                            if output_infidelity < target_error_rate and spacetime_overhead < parameters[-1]:
                                parameters = [k4,ddz4,ddx4,k3,ddz3,ddx3,ddz2,ddx2,dz1,ddx1,k2,theta,p_phy,t_error1,t_volume1,
                                    t_error2,t_volume2,t_error3,t_volume3, output_infidelity, spacetime_overhead]
    return parameters

def optimized_infidelity_level_four(theta, p_phy):
    #[k4,dz4,dx4,k3,dz3,dx3,k2,dz2,dx2,dz1,dx1, t_error, t_volume, output_infidelity, spacetime_overhead]
    inf = 1
    dx1,dz2,dx2, dz3, dx3,dz4,dx4 = 99,99,99,99,99,99,99
    for k4, k3, k2 in list(itertools.product(range(2,20),range(2,20),range(2,20))):
        delta = input_angle(k4, theta) - np.pi/8
        gamma = input_angle(k3, abs(delta)) - np.pi/8
        beta = input_angle(k2, abs(gamma))-np.pi/8
        if abs(beta) > np.pi/16 or abs(gamma) > np.pi/16 or abs(delta) > np.pi/16:
            continue
        for dz1 in range(3,48,3):
            t_error1, t_volume1 = 0, 0
            t_error2, t_volume2 = 0, 0
            t_error3, t_volume3 = 0, 0
            spacetime_overhead, output_infidelity, rho_output = protocol_level_four(
                                    k4,dz4,dx4,k3,dz3,dx3,dz2,dx2,dz1,dx1,k2,theta,p_phy,t_error1,t_volume1,
                                    t_error2,t_volume2,t_error3,t_volume3)
            if output_infidelity < inf:
                inf = output_infidelity
    return inf
