import math

def idm_acceleration(v, s, delta_v):
    # IDM parameters (classical values)
    v0 = 15.0      # desired speed (m/s)
    T = 1.5        # headway time
    a_max = 1.0    # maximum acceleration
    b = 1.5        # comfortable deceleration
    s0 = 2.0       # minimum gap
    delta = 4      # exponent

    # dynamic desired gap
    s_star = s0 + v * T + (v * delta_v) / (2 * math.sqrt(a_max * b) + 1e-9)

    if s <= 0:
        s = 0.1

    # IDM acceleration formula
    acc = a_max * (1 - (v / v0)**delta - (s_star / s)**2)
    return acc


# Small test
v = 10.0      # current speed
s = 50.0      # distance to leader
delta_v = 0.0 # same speed as leader

a = idm_acceleration(v, s, delta_v)
print("Acceleratia IDM =", a)
