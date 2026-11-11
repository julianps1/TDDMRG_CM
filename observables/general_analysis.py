"""Analysis routines shared by stationary one-particle density matrices."""

from numbers import Integral

from TDDMRG_CM.observables import bond_order, mpole, pcharge
from TDDMRG_CM.utils.util_print import (
    _print,
    print_bond_order,
    print_mpole,
    print_pcharge,
    print_section,
)


def _print_selected_bond_orders(title, atom_pairs, values):
    """Print bond orders for a sequence of one-based atom pairs."""
    print_section(title, 2)
    _print('    %6s    %6s    %10s' % ('Atom A', 'Atom B', 'Bond order'))
    for (atom_a, atom_b), value in zip(atom_pairs, values):
        _print('    %6d    %6d    %10.6f' %
               (atom_a, atom_b, value.real))


def analyze_1pdm(mol, pdm, mo, ovl_ao, dpole_ao, qpole_ao,
                  bo_pairs=None):
    """Calculate and print charge, bond-order, and multipole analyses.

    Parameters
    ----------
    mol
        PySCF molecule associated with the AO basis.
    pdm
        Complete core-plus-active 1PDM in the MO basis.
    mo
        Alpha and beta MO coefficients in the AO basis.
    ovl_ao
        AO overlap matrix.
    dpole_ao, qpole_ao
        AO dipole and quadrupole integral arrays.
    bo_pairs
        Bond-order selection using one-based atom indices, matching the time-
        evolution input. ``None`` skips bond-order analysis, ``True`` analyzes
        all atom pairs, and a sequence such as ``((1, 2), (2, 3))`` analyzes
        only those pairs.

    Returns
    -------
    dict
        Named results from each analysis.
    """
    qmul, qlow = pcharge.calc(mol, pdm, mo, ovl_ao)
    print_pcharge(mol, qmul, qlow)

    if bo_pairs is None:
        bo_mul, bo_low = None, None
    elif bo_pairs is True:
        bo_mul, bo_low = bond_order.calc(mol, pdm, mo, ovl_ao)
        print_section('Mulliken bond orders', 2)
        print_bond_order(bo_mul)
        print_section('Lowdin bond orders', 2)
        print_bond_order(bo_low)
    else:
        if isinstance(bo_pairs, bool):
            raise ValueError('bo_pairs must be None, True, or a sequence of atom pairs.')
        bo_pairs = tuple(bo_pairs)
        if (len(bo_pairs) == 2 and
            all(isinstance(atom, Integral) for atom in bo_pairs)):
            bo_pairs = (bo_pairs,)
        else:
            try:
                bo_pairs = tuple(tuple(pair) for pair in bo_pairs)
            except TypeError as exc:
                raise ValueError(
                    'bo_pairs must be an atom pair or a sequence of atom pairs.') from exc
        for pair in bo_pairs:
            if len(pair) != 2:
                raise ValueError('Each entry in bo_pairs must contain two atom indices.')
            if pair[0] == pair[1]:
                raise ValueError('The two atom indices in a bond-order pair must differ.')
            if not all(isinstance(atom, Integral) and 1 <= atom <= mol.natm
                       for atom in pair):
                raise ValueError(
                    f'Atom indices in bo_pairs must be integers between 1 and {mol.natm}.')
        zero_based_pairs = tuple((pair[0] - 1, pair[1] - 1)
                                 for pair in bo_pairs)
        bo_mul, bo_low = bond_order.calc_pair(
            mol, pdm, mo, zero_based_pairs, ovl_ao)
        _print_selected_bond_orders('Mulliken bond orders', bo_pairs, bo_mul)
        _print_selected_bond_orders('Lowdin bond orders', bo_pairs, bo_low)

    e_dpole, n_dpole, e_qpole, n_qpole = \
        mpole.calc(mol, dpole_ao, qpole_ao, pdm, mo)
    print_mpole(e_dpole, n_dpole, e_qpole, n_qpole)

    return {
        'mulliken': qmul,
        'lowdin': qlow,
        'mulliken_bond_order': bo_mul,
        'lowdin_bond_order': bo_low,
        'bond_order_pairs': bo_pairs,
        'electronic_dipole': e_dpole,
        'nuclear_dipole': n_dpole,
        'electronic_quadrupole': e_qpole,
        'nuclear_quadrupole': n_qpole,
    }
