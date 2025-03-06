import numpy as np
import sys
sys.path.append('modules/')

from ChromDynamics import MiChroM
from CndbTools import cndbTools
cndbT=cndbTools()
import logging
import os
import shutil
from scipy.optimize import curve_fit


def initialize_simulation(name='sim', N=1000, platform='None', collapse=True, nblocks_collapse=10, blocksize_collapse=10000, save_folder='output', chi=-0.1, ka=1.0, Ecut=4.0):
    #simulate
    sim = MiChroM(name=name,temperature=1.0, time_step=0.01, collision_rate=0.1)

    if platform=="None":
        platform_setup = False
        for platform in ["CUDA", "OPENCL","HIP", "CPU"]:
            try:
                sim.setup(platform=platform)
                platform_setup = True
                print(f"Selected platform: {platform}")
                break
            except: pass
            
        assert platform_setup, "No platform could be found!"
    else:
        sim.setup(platform=platform)
        print(f"Selected platform: {platform}")
    
    sim.setup(platform=platform)
    sim.saveFolder(save_folder)
    gen_seq(N)
    # mychro = sim.createRandomWalk(ChromSeq=os.path.join(args.home,f"input/exp_seq_{args.rep_frac}.txt"))#, isRing=True)
    init_struct = sim.createRandomGas(ChromSeq='input/seq.txt')#, isRing=True)
    sim.loadStructure(init_struct, center=True)

    #add potentials
    # sim.addFENEBonds(kfb=30.0)
    sim.addHarmonicBonds(kfb=100.0,r0=1.0)
    # sim.addAngles(ka=ka)
    sim.addHarmonicRestraintAngles(k_angle=ka, )
    # sim.addNextNearestNeighborHarmonicBonds(kfb=ka, r0=2.5)
    sim.addSelfAvoidance(Ecut=Ecut, k_rep=5.0, r0=0.5)
    
    gen_types_table(chi)
    sim.addCustomTypes(mu=5.0, rc = 1.5, TypesTable='input/types_table.csv')
    
    # sim.addFlatBottomHarmonic(kr=0.1, n_rad=30)
    # sim.addCylindricalConfinement(r_conf=args.rconf, z_conf=args.zconf, kr=5.0)
    
    if collapse==True:
        print('Running collapse simulation')
        for _ in range(nblocks_collapse): 
            sim.runSimBlock(blocksize_collapse, increment=False)
        # sim.saveStructure(mode = 'pdb')
    
    return sim

def gen_types_table(chi):
    with open(f'./input/types_table.csv', 'w') as f:
        f.write(f'A,B\n')
        f.write(f'{chi},{chi}\n')
        f.write(f'{chi},{chi}')
        
    
def gen_seq(N):
    with open(f'./input/seq.txt', 'w') as fseq:
        for ii in range(N):
            if ii>2: bead_type='A'
            else: bead_type='B'
            fseq.write(f'{ii+1} {bead_type}\n')
            
def get_meanRG(sim, n_blocks=5000, blocksize=1000):
    sim.initStorage(filename=sim.name)
    Rg = []
    for jj in range(n_blocks):
        sim.runSimBlock(blocksize, increment=True) 
        sim.saveStructure()
        Rg.append(sim.chromRG())
    return (np.mean(Rg), np.std(Rg))
        
    # sim.storage[0].close()
    
def load_traj(traj_file,dt=2):
    print('Loading trajectory ...')
    trajec = cndbT.load(traj_file)
    xyz=cndbT.xyz(frames=range(1,trajec.Nframes,dt))
    print('Trajectory shape:', xyz.shape)
    return xyz


ka = sys.argv[1]
chi = sys.argv[2]
ecut = sys.argv[3]
meanvals=[]
for rep in range(5):
    sim = initialize_simulation(name=f"{ka:.2f}-{chi:.2f}-{ecut:.2f}", N=1000, nblocks_collapse=1, blocksize_collapse=10000, chi=chi,ka=ka, Ecut=ecut)
    mean, std = get_meanRG(sim,n_blocks=2, blocksize=500)
    meanvals.append(mean)
    
np.savetxt(os.path.join(sim.folder, f'RG-{ka:.2f}-{ecut:.2f}-{chi:.2f}.txt'),[np.mean(meanvals), np.std(meanvals)])
            
