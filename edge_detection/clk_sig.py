import numpy as np
import os
import matplotlib.pyplot as plt

# Define the output directory for the files
output_dir = "clk_signals"
os.makedirs(output_dir, exist_ok=True)

# Define time points and signal values
discrete_time_points = np.arange(0, 1596e-9, 5e-9)
extra_points = np.arange(1e-12, 1596e-9 + 1e-12, 5e-9)
time = sorted(np.concatenate((discrete_time_points, extra_points)))

test_discrete_time_points = np.arange(0, 96e-9, 5e-9)
test_extra_points = np.arange(1e-12, 96e-9 + 1e-12, 5e-9)
test_time = sorted(np.concatenate((test_discrete_time_points, test_extra_points)))

RST =          [0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
RST_bar =      [1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
phi1 =         [0,1,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,0]
phi1_bar =     [1,0,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,1]
phi21 =        [0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,1]
phi21_bar =    [1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,0]
phi22 =        [1,0,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,1]
phi22_bar =    [0,1,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,0]

phi1c =        [0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,0]
phi1c_bar =    [1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,1]
phi2c =        [0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
phi2c_bar =    [1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
phi1_pos =     [0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0]
phi1_pos_bar = [1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1]
phi1_neg =     [0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0]
phi1_neg_bar = [1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1]

vin1 =         [0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0]
vin1 = [600e-3 * x for x in vin1]
vin2 =         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
vin3 =         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
vin4 =         [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,0]
vin4 = [600e-3 * x for x in vin4]

vin_row = vin1 + vin2 + vin3 + vin4

RST_ext = np.tile(RST, 16)
RST_bar_ext = np.tile(RST_bar, 16)
phi1_ext = np.tile(phi1, 16)
phi1_bar_ext = np.tile(phi1_bar, 16)
phi21_ext = np.tile(phi21, 16)
phi21_bar_ext = np.tile(phi21_bar, 16)
phi22_ext = np.tile(phi22, 16)
phi22_bar_ext = np.tile(phi22_bar, 16)
phi1c_ext = np.tile(phi1c, 16)
phi1c_bar_ext = np.tile(phi1c_bar, 16)
phi2c_ext = np.tile(phi2c, 16)
phi2c_bar_ext = np.tile(phi2c_bar, 16)
phi1_pos_ext = np.tile(phi1_pos, 16)
phi1_pos_bar_ext = np.tile(phi1_pos_bar, 16)
phi1_neg_ext = np.tile(phi1_neg, 16)
phi1_neg_bar_ext = np.tile(phi1_neg_bar, 16)
vin = np.tile(vin_row, 4)

RST_path = "/home/happilab/clock_signals/RST.txt"
with open(RST_path, "w") as file:
    for t, v in zip(time, RST_ext):
        file.write(f"{t:.12e}\t{v}\n")

RST_bar_path = "/home/happilab/clock_signals/RST_bar.txt"
with open(RST_bar_path, "w") as file:
    for t, v in zip(time, RST_bar_ext):
        file.write(f"{t:.12e}\t{v}\n")

phi1_path = "/home/happilab/clock_signals/phi1.txt"
with open(phi1_path, "w") as file:
    for t, v in zip(time, phi1_ext):
        file.write(f"{t:.12e}\t{v}\n")

phi1_bar_path = "/home/happilab/clock_signals/phi1_bar.txt"
with open(phi1_bar_path, "w") as file:
    for t, v in zip(time, phi1_bar_ext):
        file.write(f"{t:.12e}\t{v}\n")

phi21_path = "/home/happilab/clock_signals/phi21.txt"
with open(phi21_path, "w") as file:
    for t, v in zip(time, phi21_ext):
        file.write(f"{t:.12e}\t{v}\n")

phi21_bar_path = "/home/happilab/clock_signals/phi21_bar.txt"
with open(phi21_bar_path, "w") as file:
    for t, v in zip(time, phi21_bar_ext):
        file.write(f"{t:.12e}\t{v}\n")

phi22_path = "/home/happilab/clock_signals/phi22.txt"
with open(phi22_path, "w") as file:
    for t, v in zip(time, phi22_ext):
        file.write(f"{t:.12e}\t{v}\n")

phi22_bar_path = "/home/happilab/clock_signals/phi22_bar.txt"
with open(phi22_bar_path, "w") as file:
    for t, v in zip(time, phi22_bar_ext):
        file.write(f"{t:.12e}\t{v}\n")

phi1c_path = "/home/happilab/clock_signals/phi1c.txt"
with open(phi1c_path, "w") as file:
    for t, v in zip(time, phi1c_ext):
        file.write(f"{t:.12e}\t{v}\n")

phi1c_bar_path = "/home/happilab/clock_signals/phi1c_bar.txt"
with open(phi1c_bar_path, "w") as file:
    for t, v in zip(time, phi1c_bar_ext):
        file.write(f"{t:.12e}\t{v}\n")

phi2c_path = "/home/happilab/clock_signals/phi2c.txt"
with open(phi2c_path, "w") as file:
    for t, v in zip(time, phi2c_ext):
        file.write(f"{t:.12e}\t{v}\n")

phi2c_bar_path = "/home/happilab/clock_signals/phi2c_bar.txt"
with open(phi2c_bar_path, "w") as file:
    for t, v in zip(time, phi2c_bar_ext):
        file.write(f"{t:.12e}\t{v}\n")

phi1_pos_path = "/home/happilab/clock_signals/phi1_pos.txt"
with open(phi1_pos_path, "w") as file:
    for t, v in zip(time, phi1_pos_ext):
        file.write(f"{t:.12e}\t{v}\n")

phi1_pos_bar_path = "/home/happilab/clock_signals/phi1_pos_bar.txt"
with open(phi1_pos_bar_path, "w") as file:
    for t, v in zip(time, phi1_pos_bar_ext):
        file.write(f"{t:.12e}\t{v}\n")

phi1_neg_path = "/home/happilab/clock_signals/phi1_neg.txt"
with open(phi1_neg_path, "w") as file:
    for t, v in zip(time, phi1_neg_ext):
        file.write(f"{t:.12e}\t{v}\n")

phi1_neg_bar_path = "/home/happilab/clock_signals/phi1_neg_bar.txt"
with open(phi1_neg_bar_path, "w") as file:
    for t, v in zip(time, phi1_neg_bar_ext):
        file.write(f"{t:.12e}\t{v}\n")

vin_path = "/home/happilab/clock_signals/vin.txt"
with open(vin_path, "w") as file:
    for t, v in zip(time, vin):
        file.write(f"{t:.12e}\t{v}\n")

plt.plot(test_time, vin4, linestyle='-')
# plt.plot(vin_row)
plt.show()