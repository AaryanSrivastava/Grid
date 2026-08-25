NAMESPACE_BEGIN(Grid);

////////////////////////////////////////////////////////////////////////
//// Wuppertal Smearing
//////////////////////////////////////////////////////////////////////////
/*! @brief Wuppertal Smearing.
 *
 * The implementation is based on the definition given in
 * E. Bennett et al., "Meson spectroscopy from spectral densities in lattice
 * gauge theories", Phys. Rev. D 110, 074509. See Eq. (25) in section IIIA: APE
 * and Wuppertal smearing algorithms.
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
 * The forward and backward gauge terms are implemented using
 * CovShiftForward and CovShiftBackward. Take a look at GaugeImplementations.h
 * for more info.
 *
 * @tparam Gimpl Gauge implementation providing the gauge-covariant shift
 * operations.
 */
template <class Gimpl> class WuppertalSmearing : public Gimpl {
public:
  INHERIT_GIMPL_TYPES(Gimpl);

  typedef typename Gimpl::GaugeLinkField GaugeMat;
  typedef typename Gimpl::GaugeField GaugeLorentz;

  template <typename T>
  /*! @brief Apply Wuppertal Smearing to a field.
   *
   * @param[in] U The gauge field.
   * @param[in, out] chi The field to be smeared. The field is replaced by its
   * smeared version.
   * @param[in] step Wuppertal smearing step size (\f$\alpha\f$).
   * @param[in] Iterations Number of smearing steps.
   * @param[in] orthog Direction excluded from the smearing process.
   *                    If \f$0 \leq \texttt{orthog} < Nd\f$, smearing is
   * performed in remaining \f$(Nd - 1)\f$ directions. Use orthog >= Nd to
   * include all directions. Temporal component corresponds to 3 and spatial
   * components correspond to 0-2. The functionality is useful for performing
   * spatial smearing for example.
   */
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
