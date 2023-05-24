import os
import warnings
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# Define constants
FS_TO_NS = 1e-6 # Our LAMMPS inputs sample in real units (timesteps in femtoseconds, conv. to nanoseconds)

# Define model functions for fitting with curve_fit (nonlinear least squares)
def I_bulk_q_t(x, tau_fast=None, tau_kww=None, beta=None, Af=None):
    tau_fast = tau_fast or 1500
    tau_kww = tau_kww or 25000
    beta = beta or 0.9
    Af = Af or 0.8
    return (Af + (1.0 - Af) * np.exp(-x/tau_fast)) * np.exp(-(x/tau_kww)**beta)

def I_bound_q_t(x, tau_fast=None, tau_kww=None, beta=None, Af=None, Am=None):
    tau_fast = tau_fast or 1500
    tau_kww = tau_kww or 25000
    beta = beta or 0.9
    Af = Af or 0.8
    Am = Am or 0.8
    return (Af + (1.0 - Af) * np.exp(-x/tau_fast)) * np.exp(-(x/tau_kww)**beta)*(1.0 - Am) + Am

#     exp_factor = np.exp(-x/tau_fast)
#     power_factor = np.exp(-(x/tau_kww)**beta)
# 
#     # Check for invalid values
#     if np.isnan(power_factor).any() or np.isinf(power_factor).any():
#         raise ValueError("Invalid value encountered in power operation.")
# 
#     return (Af + (1.0 - Af) * exp_factor) * power_factor * (1.0 - Am) + Am

# Define input directory and file name to store all data
src_bulk_data = 'path/to/sample_data_bulk/fsqt_txt/'
src_bound_data = 'path/to/sample_data_bound/fsqt_txt/'
data_filename = 'data.txt'

main_directory = 'path/to/main/directory/'
os.chdir(main_directory)

# Create new directories for all data and plots
os.mkdir("data_and_plots_bulk")
os.mkdir("data_and_plots_bulk/all_fits")
os.mkdir("data_and_plots_bound")
os.mkdir("data_and_plots_bound/all_fits")

# Loop over the data files and fit the model functions to each one
for fit_type, source_dir, output_dir in [("bulk", src_bulk_data, "data_and_plots_bulk"),
                                         ("bound", src_bound_data, "data_and_plots_bound")]:
    os.chdir(os.path.join(main_directory, output_dir))

# Create a header for the output data file containing all Q-values and estimated fit parameters
    with open(data_filename, "a") as f:
        if fit_type == "bulk":
            f.write("# Q tau_fast tau_fast_err tau_kww tau_kww_err beta beta_err Af Af_err\n\n")
        else:
            f.write("# Q tau_fast tau_fast_err tau_kww tau_kww_err beta beta_err Af Af_err Am Am_err\n\n")

# Loop over the data files and fit the model function to each one
    for i in range(15):
        tt = (i + 1) * 0.1 # calculate value for the filename
        tt_filename = "fsqt_q{:.3f}.txt".format(tt) # construct filename
        tdir = "Q{:.3f}".format(tt) # construct directory name
        os.mkdir(tdir) # create directory
        os.chdir(tdir) # enter directory
    
        if fit_type == "bulk":
            fitting_function = I_bulk_q_t
            p0 = [1500, 500000, 0.5, 0.8]
        else:
            fitting_function = I_bound_q_t
            p0 = [1500, 250000, 0.7, 0.6, 0.5]
    
        with open(data_filename, "w") as f:
        
            with open(os.path.join(source_dir, tt_filename), "r") as tt_file:
            
                for line in tt_file:
                    x, y, *_ = line.strip().split() # split line into x and y
                    f.write("{} {}\n".format(x, y)) # write x and y to data.txt
    
        # Fit the function to the data and save the parameters to a file
        data = np.loadtxt(data_filename)
        x = data[:, 0]
        y = data[:, 1]
    
        fitting_successful = False;
    
        while not fitting_successful:
        
            try:
                with warnings.catch_warnings():
                
                    warnings.simplefilter("ignore", category=RuntimeWarning)

                    popt, pcov = curve_fit(fitting_function, x, y, p0=p0)
                    perr = np.sqrt(np.diag(pcov))
                    fitting_successful = True;
        
            except RuntimeWarning:
                pass

#         except(RuntimeWarning, ValueError):
#             if i >= 9:  # When the code reaches the 10th fit
#                 p0[0] -= 1
#                 p0[1] -= 500
#                 p0[2] = round(p0[2] - 0.005, 2)
#                 p0[3] = round(p0[3] - 0.005, 2)
#                 p0[4] = round(p0[4] + 0.005, 2)
#             else:
#                 p0[1] -= 2500
#                 p0[2] = round(p0[2] - 0.005, 2)
#                 p0[3] = round(p0[3] - 0.005, 2)
#                 p0[4] = round(p0[4] + 0.005, 2)
#                 
#                 print("Fit failed to converge. Adjusted params:", p0)

        with open("params.dat", "w") as f:
            if fit_type == "bulk":
                param_names = ["tau_fast", "tau_kww", "beta", "Af"]
                param_errs = ["tau_fast_err", "tau_kww_err", "beta_err", "Af_err"]
            else:
                param_names = ["tau_fast", "tau_kww", "beta", "Af", "Am"]
                param_errs = ["tau_fast_err", "tau_kww_err", "beta_err", "Af_err", "Am_err"]

            for i, p in enumerate(popt):
                f.write("{} = {:.6f} ± {:.6f}\n".format(param_names[i], p, perr[i]))

# Save the fitted parameters to the output file
        with open(os.path.join("..", data_filename), "a") as f:
            if fit_type == "bulk":
                f.write("{:.3f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                    tt, popt[0]*(FS_TO_NS), perr[0]*(FS_TO_NS),
                    popt[1]*(FS_TO_NS), perr[1]*(FS_TO_NS),
                    popt[2], perr[2], popt[3], perr[3]))
            else:
                f.write("{:.3f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                    tt, popt[0]*(FS_TO_NS), perr[0]*(FS_TO_NS),
                    popt[1]*(FS_TO_NS), perr[1]*(FS_TO_NS),
                    popt[2], perr[2], popt[3], perr[3], popt[4], perr[4]))
    
# Plot the data and the fitted function and save the plot as a PNG file
            fig, ax = plt.subplots()
    
            ax.set_xscale('log')
            ax.plot(x, y, 'bo', label='Data')
            ax.plot(x, fitting_function(x, *popt), 'r-', label='Fit')
    
            ax.legend()
            ax.set_xlabel('time (ps)')
            ax.set_ylabel('I(q,t)')
    
            xtick_locs = np.logspace(1, 6, num=6, dtype=int)
            xtick_labels = ['0.01', '0.1', '1', '10', '100', '1000']

            ax.set_xticks(xtick_locs)
            ax.set_xticklabels(xtick_labels)
            ax.set_yticks(np.arange(0, 1.1, 0.1))
    
            plt.savefig(f"fit_q{tt:.3f}.png")
            plt.savefig(f"../all_fits/fit_q{tt:.3f}.png")
            plt.close(fig)
    
            os.chdir('..') # go back to the parent directory
    
os.chdir(main_directory) # exit the parent directory after all data and plots have been generated
