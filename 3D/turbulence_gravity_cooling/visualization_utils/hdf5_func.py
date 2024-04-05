import numpy as np
import pandas as pd
import h5py

class UnitValue:
    def __init__(self, value, unit):
        self.value = value
        self.unit = unit

class Units:
    def __init__(self, density):
        self.mu = UnitValue(0.6, "")
        self.H_mass = UnitValue(1/(6.02e23), "g")  # [g]
        self.kBoltz = UnitValue(1.380649e-16, "cm^2 g/s^2/K")  # Boltzmann constant
        self.pc = UnitValue(3.0857e18, "cm")  # [cm]
        self.yr = UnitValue(3.15576e7, "s")
        self.Msolar = UnitValue(1.989e33, "g")

        if density == "dense":
            self.density = UnitValue(3.13415110082341e-22, "g/cm^3")
            self.Hdensity = UnitValue(187.250603676598, "cm^-3")
            self.time = UnitValue(219132547200359, "s")
            self.length = UnitValue(1.85417801326507e+20, "cm")
            self.velocity = UnitValue(8.46420276462703, "km/s")
            self.mass = UnitValue(1.989e+39, "g")  # [g]
        elif density == "moderate":
            self.density = UnitValue(3.13415110082341e-23, "g/cm^3")
            self.Hdensity = UnitValue(18.7250603676598, "cm^-3")
            self.time = UnitValue(691412414174066, "s")
            self.length = UnitValue(3.98876347381883e+20, "cm")
            self.velocity = UnitValue(5.76900760247941, "km/s")
            self.mass = UnitValue(1.989e+39, "g")  # [g]
        elif density == "sparse":
            self.density = UnitValue(3.13415110082341e-24, "g/cm^3")
            self.Hdensity = UnitValue(1.87250603676598, "cm^-3")
            self.time = UnitValue(2.18643803130574e+15, "s")
            self.length = UnitValue(8.59353039832737e+20, "cm")
            self.velocity = UnitValue(3.93037912590431, "km/s")
            self.mass = UnitValue(1.989e+39, "g")  # [g]
        elif density == "GADGET":
            self.density = UnitValue(6.76991117829454e-22, "g/cm^3")
            self.Hdensity = UnitValue(404.469955082753, "cm^-3")
            self.time = UnitValue(3.085678e+16, "s")
            self.length = UnitValue(3.085678e+21, "cm")
            self.velocity = UnitValue(1, "km/s")
            self.mass = UnitValue(1.989e+43, "g")  # [g]
        
        # u_pres
        self.pressure = UnitValue(self.mass.value / (self.length.value * self.time.value ** 2), "g/(cm⋅s^2)")







def readBinary(path, offset=0.4):
    # fast read
    with open(path, 'rb') as f:
        data = f.read(-1)

        np_data = np.frombuffer(data, dtype=np.int8)
        time = np.frombuffer(np_data[:8], dtype=np.float64)[0]
        Nptcl = np.frombuffer(np_data[8:16], dtype=np.int64)[0]
        header=8*2
        np_data = np_data[header:].reshape(Nptcl,-1)

        idx = np_data[:,0:8]
        pos = np_data[:,8:8*4]
        others = np_data[:,8*4:]

        idx = idx.reshape(1,-1)
        pos = pos.reshape(3,-1)
        others = others.reshape(9,-1)
        #time_Myr=6.9284040*(time-0.4)
        time_Myr=6.9284040*(time-offset)
        #print(time_Myr, "Myr")
        # np.arrayのまま変換できる。
        idx = np.frombuffer(idx, dtype=np.uint64) #8
        pos = np.frombuffer(pos, dtype=np.float64) #24
        others = np.frombuffer(others, dtype=np.float32) #36

        pos = pos.reshape(-1,3)
        others = others.reshape(-1,9)

    col=["id", "x", "y", "z", "vx", "vy", "vz", "eng", "dens", "h", "dt", "T", "pres"]
    df=pd.DataFrame(idx, columns=[col[0]])
    s=pd.DataFrame(pos, columns=col[1:4])
    df=pd.concat([df,s], axis=1)
    s=pd.DataFrame(others, columns=col[4:])
    df=pd.concat([df,s], axis=1)
    return df, time_Myr

def save_all_conditions_to_hdf5(outfh):
    conditions = ["dense", "moderate", "sparse", "GADGET"]
    for cond in conditions:
        units_instance = Units(cond)
        cond_group = outfh.create_group(cond)
        for attr_name, unit_value in units_instance.__dict__.items():
            if isinstance(unit_value, UnitValue):
                grp = cond_group.create_group(attr_name)
                grp.create_dataset("value", data=unit_value.value)
                grp.attrs['unit'] = unit_value.unit


def load_all_conditions_from_hdf5(filename):
    with h5py.File(filename, 'r') as f:
        all_units = {}
        for cond in f:
            if cond in ["dense", "moderate", "sparse", "GADGET"]:
                cond_group = f[cond]
                units_instance = Units(cond)
                for attr_name in cond_group:
                    grp = cond_group[attr_name]
                    value = grp['value'][()]
                    unit = grp.attrs['unit']
                    setattr(units_instance, attr_name, UnitValue(value, unit))
                all_units[cond] = units_instance
    return all_units


class UnitsLoader:
    def __init__(self, density="dense"):
        self.mu = UnitValue(0.6, "")
        self.H_mass = UnitValue(1/(6.02e23), "g")  # [g]
        self.u_mass = UnitValue(1.989e+39, "g")  # [g]
        self.kBoltz = UnitValue(1.380649e-16, "cm^2 g/s^2/K")  # Boltzmann constant
        self.pc = UnitValue(3.0857e18, "cm")  # [cm]
        self.yr = UnitValue(3.15576e7, "s")
        self.Msolar = UnitValue(1.989e33, "g")
        #self.velocity = UnitValue(0.0, "km/s")

    def save_to_hdf5(self, filename):
        with h5py.File(filename, 'w') as f:
            for attr_name in self.__dict__:
                unit_value = self.__dict__[attr_name]
                grp = f.create_group(attr_name)
                grp.create_dataset("value", data=unit_value.value)
                grp.attrs['unit'] = unit_value.unit

    @classmethod
    def load_from_hdf5(cls, filename, density):
        new_instance = cls(density)
        with h5py.File(filename, 'r') as f:
            cond_group = f[density]
            for attr_name in cond_group:
                grp = cond_group[attr_name]
                value = grp["value"][()]
                unit = grp.attrs['unit']
                # setattrを使って、new_instanceに属性を設定
                setattr(new_instance, attr_name, UnitValue(value, unit))
        return new_instance
