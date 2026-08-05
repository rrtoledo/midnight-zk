//! TODO
//!
//! # Known limitation: no zero-knowledge in the one-shot prover
//!
//! [prover_oneshot::ProtogalaxyProverOneShot] does not currently implement
//! zero-knowledge blinding for its final, self-contained proof's vanishing
//! argument. The `vanishing.random_poly` field carried through
//! [FoldingProverTrace] is always a zero placeholder (it is never populated
//! by [crate::plonk::vanishing::Argument::commit], which is what generates
//! real blinding randomness for a standalone PLONK proof) — it flows
//! unchanged from folding into the final proof's
//! `vanishing::prover::Committed`, whose `.construct()` call only builds and
//! commits to `h(X)` pieces without re-randomizing it. This does not affect
//! soundness (verification still correctly rejects invalid/tampered proofs),
//! but it means the one-shot folded proof's `h(X)` commitment carries no
//! blinding, so it may leak information about the underlying witness that a
//! properly blinded proof would not. (The non-one-shot
//! [prover::ProtogalaxyProver] does not reach this stage at all — it returns
//! an intermediate object rather than a finalized proof, so this limitation
//! is specific to the one-shot path.)

use ff::{PrimeField, WithSmallOrderMulGroup};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    plonk::{
        permutation, traces::FoldingProverTrace, ConstraintSystem, Evaluator, ProvingKey,
        VerifyingKey,
    },
    poly::{
        commitment::PolynomialCommitmentScheme, Coeff, EvaluationDomain, ExtendedLagrangeCoeff,
        LagrangeCoeff, Polynomial,
    },
    utils::arithmetic::parallelize,
};

pub mod prover;
pub mod prover_oneshot;
pub(crate) mod utils;
pub mod verifier;
pub mod verifier_oneshot;

/// Protogalaxy proving key. For the moment we support folding only for the same
/// circuit.
#[derive(Clone, Debug)]
pub struct FoldingPk<F: PrimeField> {
    /// TODO
    pub domain: EvaluationDomain<F>,
    /// TODO
    pub cs: ConstraintSystem<F>,
    /// TODO
    pub l0: Polynomial<F, LagrangeCoeff>,
    /// TODO
    pub l_last: Polynomial<F, LagrangeCoeff>,
    /// TODO
    pub l_active_row: Polynomial<F, LagrangeCoeff>,
    // The following three were grouped in a type called permutation::ProverKey.
    // We prefix them here to avoid creating a new type.
    /// TODO
    pub permutation_pk_permutations: Vec<Polynomial<F, LagrangeCoeff>>,
    /// TODO
    pub permutation_pk_polys: Vec<Polynomial<F, Coeff>>,
    /// TODO
    pub permutation_pk_cosets: Vec<Polynomial<F, LagrangeCoeff>>,
    /// TODO
    pub ev: Evaluator<F>,
}

impl<F: PrimeField + WithSmallOrderMulGroup<3>> FoldingPk<F> {
    /// Evaluates the (folded) constraint system on a folded trace, producing
    /// the `h(X)` polynomial (in Lagrange form) that the folding prover and
    /// verifier both need to reconstruct/check.
    pub fn compute_h(&self, folded_trace: &FoldingProverTrace<F>) -> Polynomial<F, LagrangeCoeff> {
        let FoldingProverTrace {
            fixed_polys,
            advice_polys,
            instance_values,
            lookups,
            permutations,
            trashcans,
            challenges,
            beta,
            gamma,
            theta,
            trash_challenge,
            y,
            ..
        } = folded_trace;

        self.ev.evaluate_h::<LagrangeCoeff>(
            &self.domain,
            &self.cs,
            &advice_polys.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            &instance_values.iter().map(|i| i.as_slice()).collect::<Vec<_>>(),
            fixed_polys,
            challenges,
            y,
            *beta,
            *gamma,
            theta,
            *trash_challenge,
            lookups,
            trashcans,
            permutations,
            &self.l0,
            &self.l_last,
            &self.l_active_row,
            &self.permutation_pk_cosets,
        )
    }

    /// Given a FoldingPk, it takes the folded trace and returns a proving key.
    /// Concretely, it uses the folded fixed polys as the fixed polys for
    /// the proving key.
    /// Unsure what the verifier needs to do.
    pub fn into_proving_key<CS: PolynomialCommitmentScheme<F>>(
        self,
        folded_trace: &FoldingProverTrace<F>,
        vk: &VerifyingKey<F, CS>,
    ) -> ProvingKey<F, CS> {
        let FoldingPk {
            domain,
            l0,
            l_last,
            l_active_row,
            permutation_pk_permutations,
            permutation_pk_polys,
            permutation_pk_cosets,
            ev,
            ..
        } = self;
        let lagrange_to_extended =
            |poly: Polynomial<F, LagrangeCoeff>| -> Polynomial<F, ExtendedLagrangeCoeff> {
                let tmp = domain.lagrange_to_coeff(poly);
                domain.coeff_to_extended(tmp)
            };

        let fixed_values = folded_trace.fixed_polys.clone();
        let fixed_polys = fixed_values
            .iter()
            .cloned()
            .map(|p| domain.lagrange_to_coeff(p))
            .collect::<Vec<_>>();
        let fixed_cosets = fixed_polys
            .iter()
            .cloned()
            .map(|p| domain.coeff_to_extended(p))
            .collect::<Vec<_>>();

        ProvingKey {
            vk: vk.clone(),
            l0: lagrange_to_extended(l0),
            l_last: lagrange_to_extended(l_last),
            l_active_row: lagrange_to_extended(l_active_row),
            fixed_values,
            fixed_polys,
            fixed_cosets,
            permutation: permutation::ProvingKey {
                permutations: permutation_pk_permutations,
                polys: permutation_pk_polys,
                cosets: permutation_pk_cosets
                    .into_iter()
                    .map(|poly| {
                        let poly = vk.get_domain().lagrange_to_coeff(poly);
                        vk.get_domain().coeff_to_extended(poly)
                    })
                    .collect(),
            },
            ev,
        }
    }
}

impl<F: PrimeField + WithSmallOrderMulGroup<3>, CS: PolynomialCommitmentScheme<F>>
    From<&ProvingKey<F, CS>> for FoldingPk<F>
{
    fn from(pk: &ProvingKey<F, CS>) -> Self {
        let domain = pk.vk.get_domain().clone();
        let cs = pk.vk.cs();

        let mut l0 = domain.empty_lagrange();
        l0[0] = F::ONE;

        // Compute l_blind(X) which evaluates to 1 for each blinding factor row
        // and 0 otherwise over the domain.
        let mut l_blind = domain.empty_lagrange();
        for evaluation in l_blind[..].iter_mut().rev().take(cs.blinding_factors()) {
            *evaluation = F::ONE;
        }

        // Compute l_last(X) which evaluates to 1 on the first inactive row (just
        // before the blinding factors) and 0 otherwise over the domain
        let mut l_last = domain.empty_lagrange();
        let n = domain.n as usize;
        l_last[n - cs.blinding_factors() - 1] = F::ONE;

        let mut l_active_row = domain.empty_lagrange();
        parallelize(&mut l_active_row, |values, start| {
            for (i, value) in values.iter_mut().enumerate() {
                let idx = i + start;
                *value = F::ONE - (l_last[idx] + l_blind[idx]);
            }
        });

        const COSET_CONVERSION_MEMORY_BUDGET_BYTES: usize = 512 * 1024 * 1024;
        let per_buffer_bytes = domain.extended_len() * std::mem::size_of::<F>();
        let chunk_size = (COSET_CONVERSION_MEMORY_BUDGET_BYTES / per_buffer_bytes.max(1))
            .clamp(1, pk.permutation.cosets.len().max(1));
        let permutation_pk_cosets: Vec<_> = pk
            .permutation
            .cosets
            .chunks(chunk_size)
            .flat_map(|chunk| {
                chunk.par_iter().map(|poly| domain.extended_to_lagrange(poly.clone())).collect::<Vec<_>>()
            })
            .collect();

        Self {
            cs: cs.clone(),
            l0,
            l_last,
            l_active_row,
            permutation_pk_permutations: pk.permutation.permutations.clone(),
            permutation_pk_polys: pk.permutation.polys.clone(),
            permutation_pk_cosets,
            ev: pk.ev.clone(),
            domain,
        }
    }
}
