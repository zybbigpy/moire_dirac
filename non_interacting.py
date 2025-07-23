import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from numba import njit
from pathlib import Path
from scipy.linalg import block_diag

Pi = np.pi
const_hbar2_m0 = 7.62
const_e2_epsilon0 = 180.94

def get_Glist(Gvec, nG):
    G1 = Gvec[0]
    G2 = np.array([[-1/2, -np.sqrt(3)/2], [np.sqrt(3)/2, -1/2]]) @ G1
    Glist = []
    for i in range(nG):
        for j in range(nG):
            Glist.append(i * G1 + j * G2)
    for i in range(nG):
        for j in range(1, nG):
            Glist.append(i * G2 + j * (-G1-G2))
    for i in range(1, nG):
        for j in range(1, nG):
            Glist.append(i * (-G1-G2) + j * G1)
    return np.array(Glist)

@njit
def get_G1_G2Q(Glist, Qlist):
    num_G = len(Glist)
    num_Q = len(Qlist)
    delta_tensor = np.zeros((num_Q, num_G, num_G), dtype=np.complex128)
    tol = 0.0001
    for iQ in range(num_Q):
        for i in range(num_G):
            for j in range(num_G):
                if np.linalg.norm(Glist[j] + Qlist[iQ] - Glist[i]) < tol:
                    delta_tensor[iQ, i, j] = 1
    return delta_tensor

def set_kloop(Gvec):
    num_ksec = 6
    num_onesec = 400
    num_kloop = num_onesec * (num_ksec-1)
    kline = np.zeros((num_kloop))
    kloop = np.zeros((num_kloop,2))
    ksec = np.array([[0,0],[2/3,1/3],[1/2,1/2],[1/3,2/3],[0,0],[1/2,1/2]]) @ Gvec
    #ksec_name = ['$\Gamma$', 'M', 'K', '$\Gamma$']
    #ksec = np.array([Mgamma, Mk1, Mm, Mk2, Mgamma])
    ksec_name = ["$\Gamma$", "K","M","K'","$\Gamma$",'M']

    # Set k-loop
    for i in range(num_ksec-1):
        vec = ksec[i+1] - ksec[i]
        kstep = np.linalg.norm(vec)/(num_onesec-1)
        for ik in range(num_onesec):
            kline[i*num_onesec+ik] = kline[i*num_onesec-1] + ik * kstep
            kloop[i*num_onesec+ik] = vec * ik / (num_onesec-1) + ksec[i]
    return num_ksec, num_onesec, num_kloop, kline, kloop, ksec, ksec_name

def get_quadratic_kinetic(klist, Glist, me):
    num_G = len(Glist)
    kinetic = np.zeros((len(klist), num_G, num_G), dtype=np.complex128)
    for i in range(num_G):
        k_plus_G = klist + Glist[i]
        kinetic[:, i, i] = const_hbar2_m0 / me * (k_plus_G[:,0] **2 + k_plus_G[:,1]**2)
    return kinetic

def get_massive_dirac(klist, Glist, a0, t_hopping, Delta):
    num_G = len(Glist)
    kinetic = np.zeros((len(klist), 2*num_G, 2*num_G), dtype=np.complex128)
    for i in range(num_G):
        k_plus_G = klist + Glist[i]
        kinetic[:, i, i] = 0
        kinetic[:, num_G+i, num_G+i] = -Delta
        kinetic[:, i, num_G+i] = a0 * t_hopping * (k_plus_G[:,0] - 1j*k_plus_G[:,1])
        kinetic[:, num_G+i, i] = np.conj(kinetic[:, i, num_G+i])
    return kinetic

def get_velocity(klist, Glist, a0, t_hopping):
    num_G = len(Glist)
    vx = np.zeros((len(klist), 2*num_G, 2*num_G), dtype=np.complex128)
    vy = np.zeros((len(klist), 2*num_G, 2*num_G), dtype=np.complex128)
    for i in range(num_G):
        vx[:, i, num_G+i] = a0 * t_hopping
        vx[:, num_G+i, i] = np.conj(vx[:, i, num_G+i])
        vy[:, i, num_G+i] = a0 * t_hopping * (-1j)
        vy[:, num_G+i, i] = np.conj(vy[:, i, num_G+i])
    return vx, vy

def plot_band(Eband, kline, ksec, ksec_name, figure_name, fermi=None, E_limit = None):
    aspect_ratio = 1
    num_ksec = len(ksec)
    ksec_len = np.zeros((num_ksec))
    ymin = np.amin(Eband); ymax = np.amax(Eband)
    for i in range(1,num_ksec):
        ksec_len[i] = ksec_len[i-1] + np.linalg.norm(ksec[i] - ksec[i-1])
    #for ib in range(band_select):
    fig, ax = plt.subplots(1,1)
    ax.plot(kline, Eband, 'b-', lw=1)
    #plt.plot(kline, Eband, 'b-', lw=1)
    if num_ksec > 2:
        for i in range(1, len(ksec_len)-1):
            ax.axvline(ksec_len[i],  color='black',lw=1, linestyle='--')
    #plt.ylim(ymin-0.05*(ymax-ymin), ymax+0.05*(ymax-ymin))
    if E_limit == None:
        ax.set_ylim(ymin-0.05*(ymax-ymin), ymax+0.05*(ymax-ymin))
    else:
        ax.set_ylim(E_limit[0], E_limit[1])
    if fermi !=None:
         ax.axhline(fermi,  color='black', lw=1, linestyle='--')
    #plt.ylim(-80,10)
    #plt.legend(loc='upper left')
    ax.set_xlim(kline[0], kline[-1])
    ax.set_xticks(ksec_len, ksec_name)
    ax.tick_params(axis='x', width=0, direction='in')
    ax.set_ylabel('Energy (eV)')
    ax.set_box_aspect(aspect_ratio)
    #plt.legend(loc='upper right')
    plt.savefig('figure/'+figure_name+'.png',dpi=500, bbox_inches='tight')
    plt.close()
    #plt.show()

def get_VeX_q(q, d_layer, d_gate, epsilon):
    tol = 1e-8
    if np.abs(q) < tol:
        Vq = const_e2_epsilon0 * d_layer / (2*epsilon) * (1-d_layer/(2*d_gate))
    else:
        Vq = const_e2_epsilon0 / (4*epsilon * q) \
            * (1 - np.exp(-q*d_layer)) / (1 - np.exp(-4*q*d_gate)) \
            * (2*(1-np.exp(-q*(4*d_gate-d_layer))) - np.exp(-2*q*(d_gate-d_layer)) \
                                                         *(1+np.exp(-2*q*d_layer))*(1-np.exp(-q*d_layer)))
    return Vq*np.exp(-q/0.1)

def get_Ve_Q(Qlist, core_pos, Area, d_layer, d_gate, epsilon):
    num_Q = len(Qlist)
    num_core = len(core_pos)
    Ve_Q = np.zeros((num_Q), dtype=np.complex128)
    for i in range(num_Q):
        Ve_Q[i] = 1/Area * get_VeX_q(np.linalg.norm(Qlist[i]), d_layer, d_gate, epsilon) \
            * np.sum(np.exp(1j * core_pos @ Qlist[i]))
    return Ve_Q

def get_potential(klist, Ve_Q, G1_G2Q):
    V_G1G2 = np.einsum("q,qij->ij", Ve_Q, G1_G2Q, optimize="optimal")
    return np.tile(V_G1G2, (len(klist), 1, 1))

def get_berry_curv(klist, wk, uk, vx, vy):
    num_k = len(klist)
    num_band = len(wk[0])
    vx_band = np.einsum("kim,kij,kjn->kmn", np.conj(uk), vx, uk, optimize="optimal")
    vy_band = np.einsum("kim,kij,kjn->kmn", np.conj(uk), vy, uk, optimize="optimal")
    berry_curv = np.zeros((num_k, num_band), dtype=np.float64)
    tol = 1e-6
    for n in range(num_band):
        for m in range(num_band):
            if m == n:
                continue
            else:
                berry_curv[:,n] += -2 * np.imag(vx_band[:,n,m] * vy_band[:,m,n] / np.maximum((wk[:,n]-wk[:,m])**2, tol**2))
    return berry_curv

def get_berry_curv_wilson(kmesh, uk_mesh, shift_BZ_mat):
    num_k = len(kmesh)
    Nk = int(np.sqrt(num_k)+0.1)
    num_band = uk_mesh.shape[-1]
    num_dim = uk_mesh.shape[1]
    berry_curv = np.zeros((num_k), dtype=np.float64)
    enlarged_uk_mesh = np.zeros(((Nk+1)**2, num_dim, num_band), dtype=np.complex128)
    uk_loop = np.zeros((4, num_dim, num_band), dtype=np.complex128)
    d2k = np.abs(kmesh[Nk,0] * kmesh[1,1] - kmesh[Nk,1] * kmesh[1,0])
    for i in range(Nk+1):
        for j in range(Nk+1):
            if i<Nk and j<Nk:
                enlarged_uk_mesh[i*(Nk+1)+j] = uk_mesh[i*Nk+j]
            elif i == Nk and j<Nk:
                enlarged_uk_mesh[i*(Nk+1)+j] = shift_BZ_mat[0].T @ uk_mesh[j]
            elif i<Nk and j == Nk:
                enlarged_uk_mesh[i*(Nk+1)+j] = shift_BZ_mat[1].T @ uk_mesh[i*Nk]
            elif i == Nk and j == Nk:
                enlarged_uk_mesh[i*(Nk+1)+j] = shift_BZ_mat[1].T @ shift_BZ_mat[0].T @ uk_mesh[0]
            #plot_Ck_G(np.sqrt(np.abs(enlarged_uk_mesh[i*(Nk+1)+j, :num_dim//2, 0])**2 + np.abs(enlarged_uk_mesh[i*(Nk+1)+j, num_dim//2:,0])**2), Glist, savepath=f"figure/C_kG{i}-{j}.png")
    for i in range(Nk):
        for j in range(Nk):
            loop_overlap = np.identity(num_band, dtype=np.complex128)
            uk_loop[0] = enlarged_uk_mesh[i*(Nk+1)+j]
            uk_loop[1] = enlarged_uk_mesh[(i+1)*(Nk+1)+j]
            uk_loop[2] = enlarged_uk_mesh[(i+1)*(Nk+1)+j+1]
            uk_loop[3] = enlarged_uk_mesh[i*(Nk+1)+j+1]
            for l in range(4):
                loop_overlap = loop_overlap @ np.conj(uk_loop[l%4].T) @ uk_loop[(l+1)%4]
            # Fix patch
            berry_curv[i*Nk+j] = -np.angle(np.linalg.det(loop_overlap)) / d2k
            #print(i,j, berry_curv[i*Nk+j])
    return berry_curv

def plot_Ck_G(Ck_G, Glist, title="|C_k(G)|^2 distribution", savepath=None):
    """
    Plot |C_k(G)|^2 in grayscale over reciprocal space Glist.

    Parameters:
        Ck_G:     (num_G,) complex or real array — plane wave coefficients at one k
        Glist:    (num_G, 2) array — G vectors
        title:    str — plot title
        savepath: str or None — optional path to save the figure
    """
    Ck_G = np.asarray(Ck_G).flatten()
    weights = np.abs(Ck_G)**2

    fig, ax = plt.subplots()
    sc = ax.scatter(
        Glist[:, 0], Glist[:, 1],
        c=weights,
        s=900,                        # make circles larger
        cmap='gray_r',               # grayscale: 0 → black, large → white
        edgecolors='black',
        linewidths=0.4
    )
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel(r"$G_x\ (\AA^{-1})$")
    ax.set_ylabel(r"$G_y\ (\AA^{-1})$")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label(r"$|C_k(G)|^2$ (grayscale)")
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close()

if __name__ == "__main__":
    a0 = 3.1839
    t_hopping = 1.107
    Delta = 1.602
    L = 20 * a0
    nG = 6
    Nk = 36

    d_layer = 6
    d_gate = 50
    epsilon = 10

    avec = a0 * np.array([[1,0], [1/2, np.sqrt(3)/2]])
    bvec = 2 * Pi * np.linalg.inv(avec).T
    Lvec = L * np.array([[1,0], [1/2, np.sqrt(3)/2]])
    Gvec = 2 * Pi * np.linalg.inv(Lvec).T
    Gm_center = 2/3 * bvec[0] + 1/3 * bvec[1]
    Area = np.abs(np.linalg.det(Lvec))

    core_pos = np.array([[0.0, 0.0]])
    num_core = len(core_pos)

    Glist = get_Glist(Gvec, nG)
    num_G = len(Glist)
    Qlist = get_Glist(Gvec, 2*nG-1)
    # path_G1_G2Q = f"data/G1_G2Q-nG{nG}.npy"
    # if Path(path_G1_G2Q).is_file():
    #     G1_G2Q = np.load(path_G1_G2Q)
    # else:
    G1_G2Q = get_G1_G2Q(Glist, Qlist)
    #np.save(path_G1_G2Q, G1_G2Q)
    print("G1_G2Q completed")
    # Get two special G1_G2Q where Q=G1, G2
    shift_BZ_mat = np.zeros((2, 2*num_G, 2*num_G), dtype=np.complex128)
    for i in range(len(Qlist)):
        if np.linalg.norm(Qlist[i] - Gvec[0]) < 0.01 * np.linalg.norm(Gvec[0]):
            shift_BZ_mat[0] = block_diag(G1_G2Q[i], G1_G2Q[i])
        elif np.linalg.norm(Qlist[i] - Gvec[1]) < 0.01 * np.linalg.norm(Gvec[0]):
            shift_BZ_mat[1] = block_diag(G1_G2Q[i], G1_G2Q[i])
    # test plot Glist
    fig, ax = plt.subplots(1,1)
    ax.scatter(Gm_center[0]+Glist[:,0], Gm_center[1]+Glist[:,1])
    atomic_BZ = np.array([[2/3,1/3],[1/3,2/3],[-1/3,1/3],[-2/3,-1/3],[-1/3,-2/3],[1/3,-1/3],[2/3,1/3],[0,0],[1/2,1/2]]) @ bvec
    ax.plot(atomic_BZ[:,0], atomic_BZ[:,1], color="black")
    ax.set_aspect(1)
    ax.set_xlabel("$k_x(\AA^{-1})$")
    ax.set_ylabel("$k_y(\AA^{-1})$")
    plt.savefig("figure/Glist.png", dpi=600, bbox_inches="tight")
    plt.close()

    num_k = Nk **2 
    kmesh = np.zeros((num_k, 2))
    for i in range(Nk):
        for j in range(Nk):
            kmesh[i*Nk+j] = i/Nk * Gvec[0] + j/Nk * Gvec[1]
    num_ksec, num_onesec, num_kloop, kline, kloop, ksec, ksec_name = set_kloop(Gvec)

    #kinetic_loop = get_quadratic_kinetic(kloop, Glist, me)
    kinetic_loop = get_massive_dirac(kloop, Glist, a0, t_hopping, Delta)
    print("Kinetic completed")
    Ve_Q = - get_Ve_Q(Qlist, core_pos, Area, d_layer, d_gate, epsilon)
    potential_loop = get_potential(kloop, Ve_Q, G1_G2Q)

    #qlist = np.linspace(0, 0.3, 1000)
    #vexq = np.zeros((len(qlist)))
    #for i in range(len(qlist)):
    #    vexq[i] = get_VeX_q(qlist[i], d_layer, d_gate, epsilon)
    #plt.plot(qlist, vexq/const_e2_epsilon0 * epsilon)
    #plt.show()

    print("Potential completed")

    Hk_loop = np.array(kinetic_loop)
    Hk_loop[:, :num_G, :num_G] += potential_loop
    Hk_loop[:, num_G:, num_G:] += potential_loop
    print("Hk_loop Hermitian broken: ", np.linalg.norm(Hk_loop - np.conj(np.transpose(Hk_loop,(0,2,1)))))
    wk_loop, uk_loop = np.linalg.eigh(Hk_loop)
    print("Diagonalization completed")

    plot_band(wk_loop, kline, ksec, ksec_name, "H0_bands", E_limit=[-0.1, 0.2])

    Hk_mesh = get_massive_dirac(kmesh, Glist, a0, t_hopping, Delta)
    for i in range(2):
        Hk_mesh[:, i*num_G: (i+1)*num_G, i*num_G: (i+1)*num_G] += get_potential(kmesh, Ve_Q, G1_G2Q)
    #vx_mesh, vy_mesh = get_velocity(kmesh, Glist, a0, t_hopping)
    wk_mesh, uk_mesh = np.linalg.eigh(Hk_mesh)
    print(wk_mesh.shape, uk_mesh.shape)
    np.save("wk_mesh", wk_mesh)
    np.save("uk_mesh", uk_mesh)
    np.save("Mmat", shift_BZ_mat)


    #berry_curv = get_berry_curv(kmesh, wk_mesh, uk_mesh, vx_mesh, vy_mesh)
    berry_curv = get_berry_curv_wilson(kmesh, uk_mesh[:,:,num_G:num_G+1], shift_BZ_mat)
    #print(berry_curv[:,num_G])
    #print(np.mean(berry_curv) * np.abs(np.linalg.det(Gvec)) / (2*Pi))
    enlarged_kmesh = np.concatenate((kmesh, kmesh+Gvec[0], kmesh+Gvec[1], kmesh+Gvec[0]+Gvec[1]), axis=0)
    print(berry_curv.shape)
    enlarged_berry_curv = np.tile(berry_curv, 4)
    print("enlarge bc", enlarged_berry_curv.shape)
    #fig, ax = plt.subplots(1,1)
    #tpc = ax.tricontourf(tri.Triangulation(enlarged_kmesh[:,0], enlarged_kmesh[:,1]), enlarged_berry_curv, levels=50, cmap="jet")
    #ax.set_aspect(1)
    #cb = plt.colorbar(tpc, ax=ax)
    #plt.savefig("figure/BZ_berry_curv.png", dpi=600, bbox_inches="tight")
    #plt.close()
    np.save("kmesh", kmesh)
    np.save("large_kmesh", enlarged_kmesh)

