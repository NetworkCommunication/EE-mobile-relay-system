import numpy as np
import cvxpy as cp

H = 100.0
q_I = np.array([-500.0, 0])
q_F = np.array([500.0, 0])
K = 4
gamma_0 = 1e+7
B = 1.0
P_max = 0.01
w = np.array([ [ [-450.0, 450], [-50, 50] ], [ [50, -450], [450, 50] ],
                 [ [-400, -400], [450, -450] ], [ [-250, -250], [400, 450] ] ])
delta_t = 1.0
num_of_slots = np.ceil(260.0 / delta_t)
num_of_slots = num_of_slots.astype(np.int16)
W = 40.0
A = 0.18
rho = 1.225
C_D0 = 0.08
V_h = (W / (2 * rho * A) ) ** 0.5
C_1 = W / ( (2 ** 0.5 ) * rho * A)
C_2 = (1 / 8) * C_D0 * rho * A
v_max = 50.0
epsilon_1 = 1e-3


def path_init(time, N):
    time_of_fly = 1000 / v_max
    N_of_fly = np.ceil(N * (time_of_fly / time) )
    N_of_fly = N_of_fly.astype(np.int16)
    N_of_fly_half = np.ceil(N_of_fly / 2)
    N_of_fly_half = N_of_fly_half.astype(np.int16)
    init_q = np.zeros([N + 1, 2])
    init_q[ : N_of_fly_half + 1, 0] = np.linspace(-500, 0, N_of_fly_half + 1)
    init_q[-N_of_fly_half - 1: , 0] = np.linspace(0, 500, N_of_fly_half + 1)
    return init_q


def u_sche_and_assoc_o(fixed_p_d, fixed_p_u, fixed_q, delta, N):
    constraints = []
    R_min = cp.Variable(nonneg = True)
    alpha_u = cp.Variable(shape = (K, N), nonneg = True)
    alpha_d = cp.Variable(shape = (K, N), nonneg = True)
    C_u = -np.ones([K, 2, N])
    C_d = -np.ones([K, 2, N])
    for k in range(0, K):
        for j in range(0, 2):
            C_u[k][j][ : N - 1] = np.log1p(fixed_p_u[k][j][ : N - 1] * gamma_0 / ( ( (fixed_q[1: N] - np.ones([N - 1, 2]) @ np.diag(w[k][j]) ) ** 2) @ np.ones([2]) + H ** 2) )
        C_u[k][..., : N - 1] = B * C_u[k][..., : N - 1] / np.log(2)
    for k in range(0, K):
        for j in range(0, 2):
            C_d[k][j][1: ] = np.log1p(fixed_p_d[1: ] * gamma_0 / ( ( (fixed_q[2: ] - np.ones([N - 1, 2]) @ np.diag(w[k][j]) ) ** 2) @ np.ones([2]) + H ** 2) )
        C_d[k][..., 1: ] = B * C_d[k][..., 1: ] / np.log(2)
    constraints.append(sum(alpha_u[..., 0]) <= 1)
    constraints.append(sum(alpha_d[..., N - 1]) <= 1)
    constraints.append(np.ones([K]) @ (alpha_u[..., 1: N - 1] + alpha_d[..., 1: N - 1]) <= 1)
    constraints.append(alpha_u[..., : N - 1] <= 1)
    constraints.append(alpha_d[..., 1: ] <= 1)
    for k in range(0, K):
        constraints.append(R_min * np.ones([2]) <= delta * (C_d[k][..., 1: ] @ alpha_d[k][1: ]) )
    for k in range(0, K):
        for n in range(1, N):
            constraints.append(delta * (C_d[k][..., 1: n + 1] @ alpha_d[k][1: n + 1]) <= delta * (np.array([ [0, 1], [1, 0] ]) @ C_u[k][..., : n] @ alpha_u[k][ : n]) )
    prob = cp.Problem(cp.Maximize(R_min), constraints)
    prob.solve()
    throughput = prob.value
    val_of_alpha_u = -np.ones([K, N])
    val_of_alpha_u[..., : N - 1] = alpha_u[..., : N - 1].value
    val_of_alpha_d = -np.ones([K, N])
    val_of_alpha_d[..., 1: ] = alpha_d[..., 1: ].value
    return val_of_alpha_u, val_of_alpha_d, throughput


def tx_pow_ctrl(fixed_alpha_u, fixed_alpha_d, fixed_q, delta, N):
    constraints = []
    R_min = cp.Variable(nonneg = True)
    p_d = cp.Variable(shape = (N), nonneg = True)
    p_u = []
    for k in range(0, K):
        p_u.append(cp.Variable(shape = (2, N), nonneg = True) )
    R_u = -10 * np.ones([K, 2, N])
    R_d = -10 * np.ones([K, 2, N])
    for k in range(0, K):
        for j in range(0, 2):
            R_u[k][j][ : N - 1] = 1 / ( ( (fixed_q[1: N] - np.ones([N - 1, 2]) @ np.diag(w[k][j]) ) ** 2) @ np.ones([2]) + H ** 2)
        R_u[k][..., : N - 1] = gamma_0 * R_u[k][..., : N - 1]
    for k in range(0, K):
        for j in range(0, 2):
            R_d[k][j][1: ] = 1 / ( ( (fixed_q[2: ] - np.ones([N - 1, 2]) @ np.diag(w[k][j]) ) ** 2) @ np.ones([2]) + H ** 2)
        R_d[k][..., 1: ] = gamma_0 * R_d[k][..., 1: ]
    S_1 = []
    for k in range(0, K):
        S_1.append(cp.Variable(shape = (2, N), nonneg = True) )
    for k in range(0, K):
        constraints.append(R_min <= delta * (S_1[k][..., 1: ] @ np.ones([N - 1]) ) )
    for k in range(0, K):
        for n in range(1, N):
            constraints.append(delta * cp.sum(S_1[k][..., 1: n + 1], axis = 1) <= delta * B * (np.array([ [0, 1], [1, 0] ]) @ cp.log1p(cp.multiply(R_u[k][..., : n], p_u[k][..., : n]) ) @ fixed_alpha_u[k][ : n]) / np.log(2) )
    for k in range(0, K):
        constraints.append(S_1[k][..., 1: ] <= B * (cp.log1p(R_d[k][..., 1: ] @ cp.diag(p_d[1: ]) ) @ np.diag(fixed_alpha_d[k][1: ]) ) / np.log(2) )
    constraints.append(p_d[1: ] <= P_max)
    for k in range(0, K):
        constraints.append(p_u[k][..., : N - 1] <= P_max)
    prob = cp.Problem(cp.Maximize(R_min), constraints)
    prob.solve(qcp=True)
    throughput = prob.value
    val_of_p_d = -10 * np.ones([N])
    val_of_p_d[1: ] = p_d[1: ].value
    val_of_p_u = -10 * np.ones([K, 2, N])
    for k in range(0, K):
        val_of_p_u[k][..., : N - 1] = p_u[k][..., : N - 1].value
    return val_of_p_d, val_of_p_u, throughput


def UAV_traj_o_of_ee_max(fixed_alpha_u, fixed_alpha_d, fixed_p_d, fixed_p_u, throughput_r, rth_q, delta, N):
    throughput_k = throughput_r
    kth_q = rth_q
    epsilon_2 = 1e-4
    v_r = ( ( ( (kth_q[1:] - kth_q[: N]) ** 2) @ np.ones([2]) ) ** 0.5) / delta
    kth_S_2 = (v_r ** 2 + ((v_r ** 4 + 4 * (V_h ** 4) ) ** 0.5) ) ** 0.5
    kth_E_prop = delta * sum(C_1 / kth_S_2 + C_2 * (v_r ** 3) )
    for i in range(2, 31):
        ratio_k = throughput_k / kth_E_prop
        constraints = []
        R_min = cp.Variable(nonneg = True)
        q_n = cp.Variable([N + 1, 2])
        S_1 = []
        S_1.append(cp.Variable(shape = (K, N), nonneg = True) )
        S_1.append(cp.Variable(shape = (K, N), nonneg = True) )
        S_2 = cp.Variable(shape = (N), nonneg = True)
        E_prop_3 = delta * sum(C_1 * cp.inv_pos(S_2) + C_2 * ( ( ( (q_n[1: ] - q_n[ : N]) ** 2) @ np.ones([2]) ) ** 1.5) / (delta ** 3) )
        kth_A = -np.ones([K, 2, N])
        kth_B = np.zeros([K, 2, N])
        for k in range(0, K):
            for j in range(0, 2):
                A_3 = 1 / ( ( (kth_q[1: N] - np.ones([N - 1, 2]) @ np.diag(w[k][j]) ) ** 2) @ np.ones([2]) + H ** 2)
                A_4 = fixed_p_u[k][j][ : N - 1] * gamma_0 * A_3
                kth_A[k][j][ : N - 1] = 1 / (1 + A_4) * A_4 * A_3
                kth_B[k][j][ : N - 1] = np.log1p(A_4)
            kth_A[k][..., : N - 1] = B * np.log2(np.e) * kth_A[k][..., : N - 1]
            kth_B[k][..., : N - 1] = B * kth_B[k][..., : N - 1] / np.log(2)
        kth_D = -np.ones([K, 2, N])
        kth_F = np.zeros([K, 2, N])
        for k in range(0, K):
            for j in range(0, 2):
                A_6 = 1 / ( ( (kth_q[2: ] - np.ones([N - 1, 2]) @ np.diag(w[k][j]) ) ** 2) @ np.ones([2]) + H ** 2)
                A_7 = fixed_p_d[1: ] * gamma_0 * A_6
                kth_D[k][j][1: ] = 1 / (1 + A_7) * A_7 * A_6
                kth_F[k][j][1: ] = np.log1p(A_7)
            kth_D[k][..., 1: ] = B * np.log2(np.e) * kth_D[k][..., 1: ]
            kth_F[k][..., 1: ] = B * kth_F[k][..., 1: ] / np.log(2)
        for j in range(0, 2):
            constraints.append(R_min <= delta * (S_1[j][..., 1: ] @ np.ones([N - 1]) ) )
        for k in range(0, K):
            for j in range(0, 2):
                for n in range(1, N):
                    constraints.append(delta * sum(S_1[j][k][1: n + 1]) <= delta * ( (cp.multiply(-kth_A[k][1 - j][ : n], ( ( ( (q_n[1: n + 1] - (np.ones([n, 2]) @ np.diag(w[k][1 - j]) ) ) ** 2) @ np.ones([2]) ) - ( ( (kth_q[1: n + 1] - (np.ones([n, 2]) @ np.diag(w[k][1 - j]) ) ) ** 2) @ np.ones([2]) ) ) ) + kth_B[k][1 - j][ : n]) @ fixed_alpha_u[k][ : n]) )
        for k in range(0, K):
            for j in range(0, 2):
                constraints.append(S_1[j][k][1: ] <= cp.multiply( (cp.multiply(-kth_D[k][j][1: ], ( ( ( (q_n[2: ] - (np.ones([N - 1, 2]) @ np.diag(w[k][j]) ) ) ** 2) @ np.ones([2]) ) - ( ( (kth_q[2: ] - (np.ones([N - 1, 2]) @ np.diag(w[k][j]) ) ) ** 2) @ np.ones([2]) ) ) ) + kth_F[k][j][1: ]), fixed_alpha_d[k][1: ]) )
        constraints.append(S_2 ** 2 <= (4 * (V_h ** 4) / (kth_S_2 ** 2) ) - cp.multiply(8 * (V_h ** 4) / (kth_S_2 ** 3), S_2 - kth_S_2) + (4 / (delta ** 2) * (cp.multiply( (kth_q[1: ] - kth_q[ : N])[..., 0], (q_n[1: ] - q_n[ : N])[..., 0]) + cp.multiply( (kth_q[1: ] - kth_q[ : N])[..., 1], (q_n[1: ] - q_n[ : N])[..., 1]) ) ) - (2 / (delta ** 2) * ( ( (kth_q[1: ] - kth_q[ : N]) ** 2) @ np.ones([2]) ) ) )
        constraints.append( ( (q_n[1: ] - q_n[ : N]) ** 2) @ np.ones([2]) <= (v_max * delta) ** 2)
        constraints.append(q_n[0] == q_I)
        constraints.append(q_n[N] == q_F)
        prob = cp.Problem(cp.Maximize(R_min + (-E_prop_3) * ratio_k), constraints)
        prob.solve(qcp=True)
        throughput_k = R_min.value
        kth_E_prop = E_prop_3.value
        kth_q = q_n.value
        if prob.value < epsilon_2:
            break
    val_of_q_n = q_n.value
    ratio_now = throughput_k / kth_E_prop
    return val_of_q_n, throughput_k, kth_E_prop, ratio_now


def BCD_of_ee_max(init_p_d, init_p_u, init_q, delta, N):
    ret_of_u_sche_and_assoc_o = u_sche_and_assoc_o(init_p_d, init_p_u, init_q, delta, N)
    ret_of_tx_pow_ctrl = tx_pow_ctrl(ret_of_u_sche_and_assoc_o[0], ret_of_u_sche_and_assoc_o[1], init_q, delta, N)
    ret_of_UAV_traj_o_of_ee_max = UAV_traj_o_of_ee_max(ret_of_u_sche_and_assoc_o[0], ret_of_u_sche_and_assoc_o[1], ret_of_tx_pow_ctrl[0], ret_of_tx_pow_ctrl[1], ret_of_tx_pow_ctrl[2], init_q, delta, N)
    for i in range(2, 31):
        ratio = ret_of_UAV_traj_o_of_ee_max[3]
        ret_of_u_sche_and_assoc_o = u_sche_and_assoc_o(ret_of_tx_pow_ctrl[0], ret_of_tx_pow_ctrl[1], ret_of_UAV_traj_o_of_ee_max[0], delta, N)
        ret_of_tx_pow_ctrl = tx_pow_ctrl(ret_of_u_sche_and_assoc_o[0], ret_of_u_sche_and_assoc_o[1], ret_of_UAV_traj_o_of_ee_max[0], delta, N)
        ret_of_UAV_traj_o_of_ee_max = UAV_traj_o_of_ee_max(ret_of_u_sche_and_assoc_o[0], ret_of_u_sche_and_assoc_o[1], ret_of_tx_pow_ctrl[0], ret_of_tx_pow_ctrl[1], ret_of_tx_pow_ctrl[2], ret_of_UAV_traj_o_of_ee_max[0], delta, N)
        if abs( (ratio - ret_of_UAV_traj_o_of_ee_max[3]) / ratio) < epsilon_1:
            break
    return ret_of_UAV_traj_o_of_ee_max[0], ret_of_UAV_traj_o_of_ee_max[1], ret_of_UAV_traj_o_of_ee_max[2], ret_of_UAV_traj_o_of_ee_max[3]


def main():
    init_q = path_init(260.0, num_of_slots)
    init_p_d = P_max * np.ones([num_of_slots]) / 2
    init_p_d[0] = -10.0
    init_p_u = P_max * np.ones([K, 2, num_of_slots]) / 2
    for k in range(0, K):
        init_p_u[k][..., num_of_slots - 1] = -10.0
    init_p_u_in_trad_scheme = P_max * np.ones([2, K, num_of_slots]) / 2
    for j in range(0, 2):
        init_p_u_in_trad_scheme[j][..., num_of_slots - 1] = -10.0
    ret_of_BCD_of_ee_max = BCD_of_ee_max(init_p_d, init_p_u, init_q, delta_t, num_of_slots)
    result_of_val_of_q_n = ret_of_BCD_of_ee_max[0]
    result_of_throughput = ret_of_BCD_of_ee_max[1]
    result_of_nrg = ret_of_BCD_of_ee_max[2]
    result_of_ee = ret_of_BCD_of_ee_max[3]
    print(f'result_of_throughput is')
    print(f'{result_of_throughput}')
    print(f'result_of_nrg is')
    print(f'{result_of_nrg}')
    print(f'result_of_ee is')
    print(f'{result_of_ee}')


if __name__ == '__main__':
    main()
