import matplotlib

import plots
import HF

import hamiltonian

def generate_artifacts(ham, ham_infinitesimal_u, ham_large_u):
    """
    Generate artifacts to verify the code.
    Args:
        ham: Hamiltonian with default parameters.
        ham_infinitesimal_u: Hamiltonian in the infinitesimal interaction limit.
        ham_large_u: Hamiltonian in the large interaction limit.
    """
    # Artifact 1: Symmetry of the Brillouin Zone
    try:
        fig1=plots.plot_kspace(kspace=ham.k_space)
        fig1.suptitle('K space symmetry')
        fig1.savefig('results/test1.png', dpi=200)
    except Exception as e:
        print('Exec Error in `test1.png`: K space symmetry\n', str(e))

    # Artifact 2: Check energy dispersion
    try: 
        h_nonint = ham.generate_non_interacting()
        wf,en= HF.diagonalize(h_nonint) # or HF.diagonalize
        fig2 = plots.plot_2d_bandstructure(ham,en)
        fig2.suptitle('2D Bandstructure')
        fig2.savefig('results/test2a.png', dpi=200)
    except Exception as e:
        print('Exec Error in `test2a.png`: 2D Bandstructure\n', str(e))

    try:
        fig2 = plots.plot_2d_false_color_map(ham, en) # If an error was triggered before `en` was defined in 2a, this won't execute.
        fig2.suptitle('Bands')
        fig2.savefig('results/test2b.png', dpi=200)
    except Exception as e: 
        print('Exec Error in `test2b.png`: Bands\n', str(e))

    # Artifact 3: Weak Coupling Limit
    try:
        wf,en= HF.diagonalize(ham_infinitesimal_u.generate_non_interacting())
        exp_val= HF.get_exp_val(wf,en,1/2,0)
        exp_val=HF.unflatten(exp_val, ham_infinitesimal_u.D, ham_infinitesimal_u.N_k)
        wf_int, en_int, exp_val_int= HF.solve(ham_infinitesimal_u, exp_val, 99)
        fig3 = plots.plot_2d_false_color_map(ham_infinitesimal_u, en_int)
        fig3.suptitle('Weak Coupling: Bands')
        fig3.savefig('results/test3.png', dpi=200) 
    except Exception as e: 
        print('Exec Error in `test3.png`: Weak Coupling Bands\n', str(e))

    # Artifact 4: Strong Coupling Limit
    try:
        h_nonint=ham_large_u.generate_non_interacting()
        wf,en=HF.diagonalize(h_nonint)
        exp_val=HF.get_exp_val(wf,en,1/2,0)
        exp_val=HF.unflatten(exp_val, ham_large_u.D, ham_large_u.N_k)
        _, en_int, _ = HF.solve(ham_large_u, exp_val, 99)
        fig4= plots.plot_2d_false_color_map(ham_large_u, en_int)
        fig4.suptitle('Strong Coupling: Bands')
        fig4.savefig('results/test4.png', dpi=200)
    except Exception as e:
        print('Exec Error in `test4.png`: Strong Coupling Bands\n', str(e))

    try:
        plots.print_gap(ham_large_u, exp_val, en_int, 2) # If an error was triggered before `en` was defined in 4 above, this won't execute.
    except Exception as e:
        print('Exec Error in `print_gap`: Strong Coupling\n', str(e))
    return
