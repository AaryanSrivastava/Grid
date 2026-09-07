/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./lib/qcd/action/scalar/CovariantLaplacian.h

Copyright (C) 2016

Author: Azusa Yamaguchi

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

See the full license in the file "LICENSE" in the top level distribution
directory
*************************************************************************************/
#pragma once

NAMESPACE_BEGIN(Grid);

////////////////////////////////////////////////////////////////////////
////// Gauge Covariant Smearing
////////////////////////////////////////////////////////////////////////////
/*! @brief Covariant Smearing.
 *
 * This class provides an implementation of Gaussian (Jacobi) and Wuppertal
 * smearing. Both methods use gauge-covariant forward and backward shifts of
 * the field to be smeared. The forward and backward gauge terms are implemented
 * using CovShiftForward and CovShiftBackward. Take a look at
 * GaugeImplementations.h for more info.
 *
 * @tparam Gimpl Gauge implementation providing the gauge-covariant shift
 * operations.
 */

template <class Gimpl> class CovariantSmearing : public Gimpl {
public:
  INHERIT_GIMPL_TYPES(Gimpl);

  typedef typename Gimpl::GaugeLinkField GaugeMat;
  typedef typename Gimpl::GaugeField GaugeLorentz;

  template <typename T>
  static void GaussianSmear(const std::vector<LatticeColourMatrix> &U, T &chi,
                            const Real &width, int Iterations, int orthog) {
    GridBase *grid = chi.Grid();
    T psi(grid);

    ////////////////////////////////////////////////////////////////////////////////////
    // Follow Chroma conventions for width to keep compatibility with previous
    // data Free field iterates
    //   chi = (1 - w^2/4N p^2)^N chi
    //
    //       ~ (e^(-w^2/4N p^2)^N chi
    //       ~ (e^(-w^2/4 p^2) chi
    //       ~ (e^(-w'^2/2 p^2) chi          [ w' = w/sqrt(2) ]
    //
    // Which in coordinate space is proportional to
    //
    //   e^(-x^2/w^2) = e^(-x^2/2w'^2)
    //
    // The 4 is a bit unconventional from Gaussian width perspective, but...
    // it's Chroma convention. 2nd derivative approx d^2/dx^2  =  x+mu + x-mu -
    // 2x
    //
    // d^2/dx^2 = - p^2
    //
    // chi = ( 1 + w^2/4N d^2/dx^2 )^N chi
    //
    ////////////////////////////////////////////////////////////////////////////////////
    Real coeff = (width * width) / Real(4 * Iterations);

    int dims = Nd;
    if (orthog < Nd)
      dims = Nd - 1;

    for (int n = 0; n < Iterations; ++n) {
      psi = (-2.0 * dims) * chi;
      for (int mu = 0; mu < Nd; mu++) {
        if (mu != orthog) {
          psi = psi + Gimpl::CovShiftForward(U[mu], mu, chi);
          psi = psi + Gimpl::CovShiftBackward(U[mu], mu, chi);
        }
      }
      chi = chi + coeff * psi;
    }
  }

  /*! @brief Wuppertal Smearing.
   *
   * The implementation is based on the definition given in
   * E. Bennett et al., "Meson spectroscopy from spectral densities in lattice
   * gauge theories", Phys. Rev. D 110, 074509. See Eq. (25) in section IIIA:
   * APE and Wuppertal smearing algorithms.
   *
   * For one iteration of the smearing process,
   * \f[\chi^{(n)}(x) = \frac{1}{1 + 2d\alpha}
   * \left[\chi^{(n-1)}(x) + \alpha \sum_{\mu \neq \mu_{\rm orth}}
   * \left(U_\mu(x)\chi^{(n-1)}(x+\hat\mu) +
   * U^\dagger_\mu(x-\hat\mu)\chi^{(n-1)}(x-\hat\mu)\right)\right],\f]
   * where \f$\alpha\f$ is the smearing parameter, \f$d\f$ is the
   * number of directions in which smearing is performed, and
   * \f$\mu_{\rm orth}\f$ is the direction excluded from smearing.
   *
   * @param[in] U The gauge field.
   * @param[in, out] chi The field to be smeared. The field is replaced by its
   * smeared version.
   * @param[in] step Wuppertal smearing step size (\f$\alpha\f$).
   * @param[in] Iterations Number of smearing steps.
   * @param[in] orthog Direction excluded from the smearing process.
   *                   If \f$0 \leq \texttt{orthog} < Nd\f$, smearing is
   *                   performed in remaining \f$(Nd - 1)\f$ directions. Use
   * \f$\texttt{orthog} \geq Nd\f$ to include all directions. Temporal component
   * corresponds to 3 and spatial components correspond to 0-2. The
   * functionality is useful for performing spatial smearing for example.
   */
  template <typename T>
  static void WuppertalSmear(const std::vector<LatticeColourMatrix> &U, T &chi,
                             const Real &step, int Iterations, int orthog) {
    GridBase *grid = chi.Grid();
    T psi(grid);

    Real coeff = step;

    int dims = Nd;
    if (orthog < Nd)
      dims = Nd - 1;
    double norm = 1 / (1 + 2.0 * dims * coeff);

    for (int n = 0; n < Iterations; ++n) {
      psi = chi;
      for (int mu = 0; mu < Nd; mu++) {
        if (mu != orthog) {
          psi = psi + coeff * (Gimpl::CovShiftForward(U[mu], mu, chi));
          psi = psi + coeff * (Gimpl::CovShiftBackward(U[mu], mu, chi));
        }
      }
      chi = norm * psi;
    }
  }
};

NAMESPACE_END(Grid);
