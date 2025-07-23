import numpy as np
import copy
from wilson_loop import cal_wilson_loop, one_flux_plane
import matplotlib.tri as tri

def moire_chern_one_band(dmesh, M1, M2):

    print(f"wavefunction shape: {dmesh.shape}")
    (nk2, ng2, num_band) = dmesh.shape
    print(f"nk2: {nk2}, ng2: {ng2}, numband: {num_band}")

    # choose flat band wavefunction
    num_band_chern = 1
    dmesh_flat = dmesh[:, :, 0:1]
    nk = int(np.sqrt(nk2))
    ng = int(ng2/2)
    print(f"nk: {nk}, num of Gvecs: {ng}")
    # dmesh reshape
    dmesh_old = copy.deepcopy(dmesh_flat)
    dmesh_flat = dmesh_flat.reshape((nk, nk, ng2, num_band_chern))
    print(dmesh_flat.shape, dmesh_old.shape)
    # check
    assert np.allclose(dmesh_flat[1, 3, :, :], dmesh_old[nk*1+3, :,:]) == True 
    # impose PBC
    dmesh_pbc = np.zeros((nk+1, nk+1, ng2, num_band_chern), dtype=complex)
    dmesh_pbc[:-1, :-1, :, :] = dmesh_flat
    # check before add PBC
    assert np.allclose(dmesh_pbc[1, 3, :, :], dmesh_old[nk*1+3, :,:]) == True 
    # manually impose PBC
    for i in range(nk+1):
        dmesh_pbc[-1, i, :, :] = np.dot(M1, dmesh_pbc[0, i, :, :])
    for i in range(nk+1):
        dmesh_pbc[i, -1, :, :] = np.dot(M2, dmesh_pbc[i, 0, :, :])


    print("dmesh pbc shape", dmesh_pbc.shape)
    phi = cal_wilson_loop(dmesh_pbc[:, :, :, 0:1], berry_evals=True)
    chern = one_flux_plane(dmesh_pbc[:, :, :, 0:1]).sum()/np.pi/2
    bc = one_flux_plane(dmesh_pbc[:, :, :, 0:1])

    return chern, phi, bc

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dmesh = np.load("uk_mesh.npy")
    dmesh_band = dmesh[:,:,91:92]
    print(dmesh.shape)
    Mmat = np.load("Mmat.npy")
    print(Mmat.shape)
    chern, phi, bc = moire_chern_one_band(dmesh_band, Mmat[0].T, Mmat[1].T)
    print(chern)

    # fig, ax = plt.subplots(figsize=(4.5, 4.5))
    # kx = np.linspace(0, 1, len(phi))
    # phi = np.array(phi)
    # ax.plot(kx, phi[:, 0], 'ko', alpha=0.5)
    # ax.set_xlabel("$k_x$")
    # ax.set_ylabel("WCC")
    # ax.set_xlim(0, 1)
    # ax.set_ylim(-3.5, 3.5)
    # ax.yaxis.set_ticks([-np.pi, 0, np.pi])
    # ax.set_yticklabels((r'$-\pi$',r'$0$',r'$\pi$'))
    # ax.set_title(f"$C=${chern:.2f}")
    # plt.tight_layout()
    # plt.grid()
    # plt.show()


    kmesh = np.load("kmesh.npy")
    enlarged_kmesh = np.load("large_kmesh.npy")
    enlarged_berry_curv = np.tile(bc, 4)
    print(kmesh.shape, enlarged_kmesh.shape, bc.shape, enlarged_berry_curv.shape)


    fig, ax = plt.subplots(1,1)
    bc = bc.reshape((1296,))
    tpc = ax.tricontourf(tri.Triangulation(kmesh[:,0], kmesh[:,1]), bc, levels=50, cmap="jet")
    ax.set_aspect(1)
    cb = plt.colorbar(tpc, ax=ax)
    plt.savefig("figure/BZ_berry_curv.png", dpi=600, bbox_inches="tight")
    plt.close()