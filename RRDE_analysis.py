import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

df = pd.read_csv("examplefolder.csv")

# Options
baseline_correction = True     # If true, the initial current will be subtracted from the entire current set
show_fit = False               # If true it will show the I_ring / I_disc linear fit

area = 0.2 * 0.2 * 3.14

# Dataframe column names
time = "time/s"
ring_curr = "<I>/mA_ring"
disc_curr = "<I>/mA_disc"
pot = "Ewe/V"
cycle = "cycle number"
rot = "Rotation/rpm"
water = "H2O"
branch = "IsCathodic"
id = "ID"

# Figure settings
fig, ax1 = plt.subplots(figsize = (12/2.54, 12/2.54))

# data filter
df = df[df[branch]]

# Colormap
cmap = plt.get_cmap("viridis")
n = len(df[id].unique())
color_list = cmap(np.linspace(0.0, 0.65, n))

# Conversion from LF8 to RHE
def pH_shift(pH):
    return 0.222 + 0.047 + (0.05916 * pH)

shift_dict = {
    50: pH_shift(8.00),
    20: pH_shift(8.55),
    10: pH_shift(9.05),
    5: pH_shift(9.69)
}

# Linear function
def linear(x, a, b):
    return ((a * x) + b)

if baseline_correction == True:
    for cycle_id in df[id].unique():
        sub_df = df[df[id] == cycle_id]

        disc_baseline = sub_df.iloc[:50][disc_curr].mean()
        ring_baseline = sub_df.iloc[:50][ring_curr].mean()

        # df.loc[df[id] == cycle_id, disc_curr] = df.loc[df[id] == cycle_id, disc_curr] - disc_baseline
        df.loc[df[id] == cycle_id, ring_curr] = df.loc[df[id] == cycle_id, ring_curr] - ring_baseline

n_list = []

for i, cycle_id in enumerate(df[id].unique()):
    sub_df = df[df[id] == cycle_id].copy()

    # Find water content
    w = sub_df.iloc[0][water]
    id_water = df[water].unique().tolist().index(w)

    # Shift potential
    her_pot_shift = shift_dict[w]
    sub_df[pot] = sub_df[pot] + her_pot_shift

    cat_branch = sub_df[sub_df[branch]]

    # Range used for linear fitting
    coll_eff_interval = cat_branch.loc[(cat_branch[disc_curr] > -0.01) & (cat_branch[disc_curr] < -0.005)]

    x_data = -coll_eff_interval[disc_curr]
    y_data = coll_eff_interval[ring_curr]

    popt, pcov = curve_fit(linear, x_data, y_data)

    # Calculation for R^2
    residuals = y_data - linear(x_data, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    r_squared = 1 - (ss_res/ss_tot)

    x_fit_min = -cat_branch[disc_curr].min()
    x_fit_max = -cat_branch[disc_curr].max()

    y_fit_min = linear(x_fit_min, *popt)
    y_fit_max = linear(x_fit_max, *popt)

    # label = f"N = {popt[0]:1.4f}, R$^2$ = {r_squared:1.4f}"
    # label = "KOAc " + str(w) + " H2O"
    # label = str(sub_df.iloc[0][rot]) + " rpm, KOAc 5H$_2$O"

    label = "+" + str(sub_df.iloc[0]["file"]).split("_")[12][:3] + f" mV RHE (N = {popt[0]:1.3f})"

    if show_fit:
        ax1.plot(-cat_branch[disc_curr], cat_branch[ring_curr], color = color_list[i], lw = 1.5, label = label)
        ax1.plot(-coll_eff_interval[disc_curr], coll_eff_interval[ring_curr], color = "crimson", lw = 2.5)
        ax1.plot([x_fit_min, x_fit_max], [y_fit_min, y_fit_max], ls = "dashed", color = "black", lw = 0.5)
    else:
        x = cat_branch[pot]
        y1 = cat_branch[disc_curr] / area
        y2 = -cat_branch[ring_curr] / popt[0] / area

        ax1.plot(x, y1, color = color_list[i], label = label)
        ax1.plot(x, y2, color = color_list[i], alpha = 0.8, ls = "dashed")

    fit_dict = {
        water: w,
        "N": popt[0],
        "R2": r_squared
    }
    print(popt[0], sub_df.iloc[0]["file"])
    n_list.append(fit_dict)

fit_df = pd.DataFrame(n_list)
# print(fit_df.groupby(by = water, as_index = False).mean())
# print(fit_df.groupby(by = water, as_index = False).std())

ax1.legend(frameon = False)

# ax1.set_ylim(-0.08, 0.01)
# ax1.set_xlim(-0.75, -0.25)
# ax2.set_ylim(-0.32, 0.02)

ax1.tick_params(axis = "both", direction = "in")

if show_fit:
    ax1.set_xlabel("-I$_{disc}$ (mA)", fontsize = 12)
    ax1.set_ylabel("I$_{ring}$ (mA)", fontsize = 12)
else:
    ax1.set_xlabel("E (V vs RHE)", fontsize = 12)
    ax1.set_ylabel("I (mA cm$^{-2}$)", fontsize = 12)

fig.tight_layout(pad = 0.1)
plt.show()
if show_fit:
    fig.savefig("lsv.png", dpi = 1200, bbox_inches = "tight", pad_inches = 0.05)
else:
    fig.savefig("lsv_fit.png", dpi = 1200, bbox_inches = "tight", pad_inches = 0.05)