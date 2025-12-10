import math

def idm_acceleration(v, s, delta_v):
    """
    Classical IDM acceleration.
    v        - current speed of ego vehicle [m/s]
    s        - gap to leader [m] (if no leader, use large number)
    delta_v  - v - v_leader [m/s] (positive if ego faster than leader)
    """
    # IDM parameters (classical values)
    v0 = 15.0      # desired speed (m/s)
    T = 1.5        # desired time headway
    a_max = 1.0    # maximum acceleration
    b = 1.5        # comfortable deceleration
    s0 = 2.0       # minimum gap
    delta = 4      # exponent

    # Desired dynamic gap s*
    s_star = s0 + v * T + (v * delta_v) / (2 * math.sqrt(a_max * b) + 1e-9)

    # Avoid division by zero
    if s <= 0:
        s = 0.1

    # IDM acceleration formula
    acc = a_max * (1 - (v / v0) ** delta - (s_star / s) ** 2)
    return acc

# add traffic lights
def signal_phase(t, cycle_length, green_duration):
    """
    Simple fixed-time traffic signal:
    - GREEN for [0, green_duration)
    - RED for [green_duration, cycle_length]
    Repeats every cycle_length seconds.

    Returns:
    phase: 'green' or 'red'
    time_to_change: seconds until next phase change
    """

    t_mod = t % cycle_length
    if t_mod< green_duration:
        phase = 'green'
        time_to_change = green_duration - t_mod
    else:
        phase = 'red'
        time_to_change = cycle_length - t_mod
    return phase, time_to_change

def simulate_idm():
    # Simulation settings
    t_end = 10.0   # total time [s]
    dt = 0.1       # time step [s]
    n_steps = int(t_end / dt)

    # Initial conditions
    x = 0.0        # position [m]
    v = 10.0       # initial speed [m/s]
    a = 0.0        # initial acceleration [m/s^2]

    # No leader: we simulate free road (very large gap)
    big_gap = 1e6
    delta_v = 0.0  # no relative speed

    print("t [s]\t x [m]\t v [m/s]\t a [m/s^2]")
    for i in range(n_steps + 1):
        t = i * dt

        # Print every 0.5 s (la fiecare 5 pași)
        if i % 5 == 0:
            print(f"{t:5.1f}\t {x:6.1f}\t {v:6.2f}\t {a:7.3f}")

        # 1) calculăm accelerația după IDM pentru starea curentă
        a = idm_acceleration(v=v, s=big_gap, delta_v=delta_v)

        # 2) actualizăm viteza (nu o lăsăm să devină negativă)
        v = max(0.0, v + a * dt)

        # 3) actualizăm poziția
        x = x + v * dt

def simulate_idm_with_signal():
    #Simulation settings
    t_end = 40.0
    dt = 0.5
    n_steps = int(t_end / dt)

    #Initial conditions
    x = 0.0       # position [m]
    v = 10.0      # initial speed [m/s]
    a = 0.0       # initial acceleration [m/s^2]

    #No leader (free way)
    big_gap = 1e6
    delta_v = 0.0 # no relative speed

    # Traffic light within 200m, with cycle: 30s green, 30s red

    x_signal = 200.0
    cycle_length = 60.0
    green_duration = 30.0

    print("t [s]\t x [m]\t v [m/s]\t phase\t t_to_change [s]")
    for i in range(n_steps + 1):
        t = i * dt

        # Determinăm faza semaforului
        phase, time_to_change = signal_phase(t, cycle_length, green_duration)

        # Afișăm doar fiecare pas (dt e deja 0.5)
        print(f"{t:5.1f}\t {x:6.1f}\t {v:6.2f}\t {phase:5s}\t {time_to_change:6.1f}")

        # Deocamdată IDM NU știe de semafor; doar îl observăm.
        a = idm_acceleration(v=v, s=big_gap, delta_v=delta_v)
        v = max(0.0, v + a * dt)
        x = x + v * dt

if __name__ == "__main__":
    #simulate_idm()
    simulate_idm_with_signal()
