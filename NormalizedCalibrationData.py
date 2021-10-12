# Niko sent me calibration data (2020-11-30, 4:35 p.m.) based on Ec=0.7373. This procedure creates another file and turns all the temperatures into Ec-normalized temperatures.
def NormalizeCalibrationData()
    Ec = 0.7373*1e-3
    with h5py.File("./data/Calibration.h5", "r") as f:
        me_g0gT = np.array(f['master_equation/normalized zero bias conductance'])[:,0]
        me_T = np.array(f['master_equation/temperature'])[:,0]
        mc_g0gT = np.array(f['monte_carlo/mean normalized zero bias conductance'])[:,0]
        mc_g0gTstd = np.array(f['monte_carlo/std normalized zero bias conductance'])[:,0]
        mc_T = np.array(f['monte_carlo/temperature'])[:,0]
        mcht_g0gT = np.array(f['monte_carlo_high_temperature/normalized zero bias conductance'])[:,0]
        mcht_T = np.array(f['monte_carlo_high_temperature/temperature'])[:,0]
    with h5py.File("./data/Calibration_NormalizedTemperature.h5", "w") as f:
        f.create_dataset('master_equation/normalized zero bias conductance', data=me_g0gT)
        f.create_dataset('master_equation/normalized temperature', data=me_T/Ec)
        f.create_dataset('monte_carlo/mean normalized zero bias conductance', data=mc_g0gT)
        f.create_dataset('monte_carlo/std normalized zero bias conductance', data=mc_g0gTstd)
        f.create_dataset('monte_carlo/normalized temperature', data=mc_T/Ec)
        f.create_dataset('monte_carlo_high_temperature/normalized zero bias conductance', data=mcht_g0gT)
        f.create_dataset('monte_carlo_high_temperature/normalized temperature', data=mcht_T/Ec)