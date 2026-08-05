//! IVC-flavoured folding example.
//!
//! We generate `K_PROOFS` inner proofs of a small circuit (a Poseidon
//! preimage). For each inner proof we build an instance of a "verifier
//! circuit" that verifies that one inner proof in-circuit (à la `ivc.rs`,
//! using [VerifierGadget]). We then fold the `K_PROOFS` verifier-circuit
//! instances together using Protogalaxy, exactly as `folding.rs` folds
//! copies of `ShaPreImageCircuit`.
//!
//! DO NOT add this example to the CI as it is slow.

use std::{collections::BTreeMap, time::Instant};

use ff::Field;
use group::Group;
use midnight_circuits::{
    ecc::{
        curves::CircuitCurve,
        foreign::{nb_foreign_ecc_chip_columns, ForeignEccChip, ForeignEccConfig},
    },
    field::{
        decomposition::{
            chip::{P2RDecompositionChip, P2RDecompositionConfig},
            pow2range::Pow2RangeChip,
        },
        foreign::FieldChip,
        native::NB_ARITH_COLS,
        NativeChip, NativeConfig, NativeGadget,
    },
    hash::poseidon::{
        PoseidonChip, PoseidonState, NB_POSEIDON_ADVICE_COLS, NB_POSEIDON_FIXED_COLS,
    },
    instructions::{
        hash::{HashCPU, HashInstructions},
        AssignmentInstructions, PublicInputInstructions,
    },
    testing_utils::plonk_api::filecoin_srs,
    types::{ComposableChip, Instantiable},
    verifier::{self, Accumulator, AssignedAccumulator, AssignedVk, BlstrsEmulation, SelfEmulation, VerifierGadget},
};
use midnight_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{
        create_proof, keygen_pk, keygen_vk_with_k, prepare, Circuit, ConstraintSystem, Error,
    },
    poly::{kzg::KZGCommitmentScheme, EvaluationDomain},
    protogalaxy::{prover::ProtogalaxyProver, verifier::ProtogalaxyVerifier},
    transcript::{CircuitTranscript, Transcript},
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

type S = BlstrsEmulation;

type F = <S as SelfEmulation>::F;
type C = <S as SelfEmulation>::C;

type E = <S as SelfEmulation>::Engine;
type CBase = <C as CircuitCurve>::Base;

type NG = NativeGadget<F, P2RDecompositionChip<F>, NativeChip<F>>;

/// Number of inner proofs we generate, verify in-circuit and fold together.
const K_PROOFS: usize = 4;

/// Log-size of the (small) inner circuit.
const INNER_K: u32 = 10;

/// Log-size of the verifier circuit (it embeds a BLS12-381 verifier, hence
/// the much larger domain).
#[cfg(feature = "truncated-challenges")]
const K: u32 = 18;
#[cfg(not(feature = "truncated-challenges"))]
const K: u32 = 19;

/// The small circuit whose proofs we will fold: a Poseidon preimage circuit
/// (hand-wired, matching the shape [VerifierGadget] expects: a committed
/// instance column followed by a normal instance column).
#[derive(Clone, Debug, Default)]
pub struct InnerCircuit {
    preimage: Value<[F; 2]>,
}

impl InnerCircuit {
    fn from_witness(preimage: [F; 2]) -> Self {
        Self {
            preimage: Value::known(preimage),
        }
    }
}

impl Circuit<F> for InnerCircuit {
    type Config = (NativeConfig, midnight_circuits::hash::poseidon::PoseidonConfig<F>);

    type FloorPlanner = SimpleFloorPlanner;

    type Params = ();

    fn without_witnesses(&self) -> Self {
        unreachable!()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        let nb_advice_cols = NB_ARITH_COLS.max(NB_POSEIDON_ADVICE_COLS);
        let nb_fixed_cols = (NB_ARITH_COLS + 4).max(NB_POSEIDON_FIXED_COLS);

        let advice_columns: Vec<_> = (0..nb_advice_cols).map(|_| meta.advice_column()).collect();
        let fixed_columns: Vec<_> = (0..nb_fixed_cols).map(|_| meta.fixed_column()).collect();
        let committed_instance_column = meta.instance_column();
        let instance_column = meta.instance_column();

        let native_config = NativeChip::configure(
            meta,
            &(
                advice_columns[..NB_ARITH_COLS].try_into().unwrap(),
                fixed_columns[..NB_ARITH_COLS + 4].try_into().unwrap(),
                [committed_instance_column, instance_column],
            ),
        );
        let poseidon_config = PoseidonChip::configure(
            meta,
            &(
                advice_columns[..NB_POSEIDON_ADVICE_COLS].try_into().unwrap(),
                fixed_columns[..NB_POSEIDON_FIXED_COLS].try_into().unwrap(),
            ),
        );

        (native_config, poseidon_config)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let native_chip = <NativeChip<F> as ComposableChip<F>>::new(&config.0, &());
        let poseidon_chip = PoseidonChip::new(&config.1, &native_chip);

        let inputs = native_chip.assign_many(&mut layouter, &self.preimage.transpose_array())?;
        let output = poseidon_chip.hash(&mut layouter, &inputs)?;

        native_chip.constrain_as_public_input(&mut layouter, &output)
    }
}

/// A circuit that verifies (in-circuit) a single proof of [InnerCircuit].
/// `K_PROOFS` instances of this circuit (one per inner proof) are what get
/// folded together with Protogalaxy.
#[derive(Clone, Debug)]
pub struct VerifierCircuit {
    inner_vk: (EvaluationDomain<F>, ConstraintSystem<F>, Value<F>), // (domain, cs, vk_repr)
    inner_committed_instance: Value<C>,
    inner_instance: Value<F>,
    inner_proof: Value<Vec<u8>>,
}

fn configure_verifier_circuit(
    meta: &mut ConstraintSystem<F>,
) -> (
    NativeConfig,
    P2RDecompositionConfig,
    ForeignEccConfig<C>,
    midnight_circuits::hash::poseidon::PoseidonConfig<F>,
) {
    let nb_advice_cols = nb_foreign_ecc_chip_columns::<F, C, C, NG>();
    let nb_fixed_cols = NB_ARITH_COLS + 4;

    let advice_columns: Vec<_> = (0..nb_advice_cols).map(|_| meta.advice_column()).collect();
    let fixed_columns: Vec<_> = (0..nb_fixed_cols).map(|_| meta.fixed_column()).collect();
    let committed_instance_column = meta.instance_column();
    let instance_column = meta.instance_column();

    let native_config = NativeChip::configure(
        meta,
        &(
            advice_columns[..NB_ARITH_COLS].try_into().unwrap(),
            fixed_columns[..NB_ARITH_COLS + 4].try_into().unwrap(),
            [committed_instance_column, instance_column],
        ),
    );
    let core_decomp_config = {
        let pow2_config = Pow2RangeChip::configure(meta, &advice_columns[1..NB_ARITH_COLS]);
        P2RDecompositionChip::configure(meta, &(native_config.clone(), pow2_config))
    };

    let base_config = FieldChip::<F, CBase, C, NG>::configure(meta, &advice_columns);
    let curve_config =
        ForeignEccChip::<F, C, C, NG, NG>::configure(meta, &base_config, &advice_columns);

    let poseidon_config = PoseidonChip::configure(
        meta,
        &(
            advice_columns[..midnight_circuits::hash::poseidon::NB_POSEIDON_ADVICE_COLS]
                .try_into()
                .unwrap(),
            fixed_columns[..midnight_circuits::hash::poseidon::NB_POSEIDON_FIXED_COLS]
                .try_into()
                .unwrap(),
        ),
    );

    (
        native_config,
        core_decomp_config,
        curve_config,
        poseidon_config,
    )
}

impl Circuit<F> for VerifierCircuit {
    type Config = (
        NativeConfig,
        P2RDecompositionConfig,
        ForeignEccConfig<C>,
        midnight_circuits::hash::poseidon::PoseidonConfig<F>,
    );
    type FloorPlanner = SimpleFloorPlanner;
    type Params = ();

    fn without_witnesses(&self) -> Self {
        unreachable!()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        configure_verifier_circuit(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let native_chip = <NativeChip<F> as ComposableChip<F>>::new(&config.0, &());
        let core_decomp_chip = P2RDecompositionChip::new(&config.1, &(K as usize - 1));
        let scalar_chip = NativeGadget::new(core_decomp_chip.clone(), native_chip.clone());
        let curve_chip = ForeignEccChip::new(&config.2, &scalar_chip, &scalar_chip);
        let poseidon_chip = PoseidonChip::new(&config.3, &native_chip);

        let verifier_chip = VerifierGadget::new(&curve_chip, &scalar_chip, &poseidon_chip);

        let (inner_domain, inner_cs, inner_vk_repr) = &self.inner_vk;
        let assigned_inner_vk: AssignedVk<S> = verifier_chip.assign_vk_as_public_input(
            &mut layouter,
            "inner_vk",
            inner_domain,
            inner_cs,
            *inner_vk_repr,
        )?;

        let assigned_committed_instance =
            curve_chip.assign(&mut layouter, self.inner_committed_instance)?;

        let assigned_inner_pi = native_chip.assign(&mut layouter, self.inner_instance)?;

        let mut inner_proof_acc = verifier_chip.prepare(
            &mut layouter,
            &assigned_inner_vk,
            &[("com_instance", assigned_committed_instance)],
            &[&[assigned_inner_pi]],
            self.inner_proof.clone(),
        )?;

        inner_proof_acc.collapse(&mut layouter, &curve_chip, &scalar_chip)?;

        verifier_chip.constrain_as_public_input(&mut layouter, &inner_proof_acc)?;

        core_decomp_chip.load(&mut layouter)
    }
}

fn main() {
    // --- 1. Prove K_PROOFS instances of the small inner circuit. ---
    let inner_srs = filecoin_srs(INNER_K);
    let inner_vk =
        keygen_vk_with_k::<F, KZGCommitmentScheme<E>, InnerCircuit>(&inner_srs, &InnerCircuit::default(), INNER_K)
            .unwrap();
    let inner_pk = keygen_pk(inner_vk.clone(), &InnerCircuit::default()).unwrap();

    let mut rng = ChaCha8Rng::from_entropy();

    let mut fixed_bases = BTreeMap::new();
    fixed_bases.insert(String::from("com_instance"), C::identity());
    fixed_bases.extend(verifier::fixed_bases::<S>("inner_vk", &inner_vk));

    let inner_proving = Instant::now();
    let inner_proofs: Vec<(F, Vec<u8>)> = (0..K_PROOFS)
        .map(|_| {
            let preimage: [F; 2] = core::array::from_fn(|_| F::random(&mut rng));
            let output = <PoseidonChip<F> as HashCPU<F, F>>::hash(&preimage);

            // The inner proof is generated with a Poseidon-based (in-circuit
            // friendly) transcript, since it must be parsed by [VerifierGadget].
            let mut transcript = CircuitTranscript::<PoseidonState<F>>::init();
            create_proof::<F, KZGCommitmentScheme<E>, CircuitTranscript<PoseidonState<F>>, InnerCircuit>(
                &inner_srs,
                &inner_pk,
                &[InnerCircuit::from_witness(preimage)],
                1,
                &[&[&[], &vec![output]]],
                &mut rng,
                &mut transcript,
            )
            .expect("Inner proof generation should not fail");

            (output, transcript.finalize())
        })
        .collect();
    println!(
        "Time to generate {} inner proofs: {:?}",
        K_PROOFS,
        inner_proving.elapsed()
    );

    let tampered_inner_proof = {
        let output = inner_proofs[0].0.clone();
        let mut bytes = inner_proofs[0].1.clone();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;
        (output,bytes)
    };

    // --- 2. For each inner proof, precompute the accumulator attesting to its
    // validity (this is what each verifier circuit will witness) and build the
    // corresponding verifier-circuit instance + public input. ---
    let accumulating = Instant::now();
    let mut public_inputs = Vec::with_capacity(K_PROOFS);
    let mut verifier_circuits = Vec::with_capacity(K_PROOFS);
    for (output, inner_proof) in inner_proofs {
        let mut inner_acc: Accumulator<S> = {
            let mut transcript = CircuitTranscript::<PoseidonState<F>>::init_from_bytes(&inner_proof);
            let dual_msm = prepare::<F, KZGCommitmentScheme<E>, CircuitTranscript<PoseidonState<F>>>(
                &inner_vk,
                &[&[C::identity()]],
                &[&[&[output]]],
                &mut transcript,
            )
            .expect("Failed to prepare the inner proof");
            let mut acc: Accumulator<S> = dual_msm.into();
            acc.extract_fixed_bases(&fixed_bases);
            acc
        };
        inner_acc.collapse();

        let mut pi = AssignedVk::<S>::as_public_input(&inner_vk);
        pi.extend(AssignedAccumulator::as_public_input(&inner_acc));
        public_inputs.push(pi);

        verifier_circuits.push(VerifierCircuit {
            inner_vk: (
                inner_vk.get_domain().clone(),
                inner_vk.cs().clone(),
                Value::known(inner_vk.transcript_repr()),
            ),
            inner_committed_instance: Value::known(C::identity()),
            inner_instance: Value::known(output),
            inner_proof: Value::known(inner_proof),
        });
    }
    println!(
        "Time to prepare {} accumulators: {:?}",
        K_PROOFS,
        accumulating.elapsed()
    );

    // --- 3. Setup the verifier circuit. ---
    let default_verifier_circuit = VerifierCircuit {
        inner_vk: (
            inner_vk.get_domain().clone(),
            inner_vk.cs().clone(),
            Value::unknown(),
        ),
        inner_committed_instance: Value::unknown(),
        inner_instance: Value::unknown(),
        inner_proof: Value::unknown(),
    };

    let outer_srs = filecoin_srs(K);

    let keygen = Instant::now();
    let vk = keygen_vk_with_k::<F, KZGCommitmentScheme<E>, VerifierCircuit>(
        &outer_srs,
        &default_verifier_circuit,
        K,
    )
    .unwrap();
    let pk = keygen_pk(vk.clone(), &default_verifier_circuit).unwrap();
    println!("Computed verifier-circuit vk/pk in {:?}", keygen.elapsed());

    {
        use midnight_proofs::dev::MockProver;
        let prover = MockProver::run(K, &verifier_circuits[0], vec![vec![], public_inputs[0].clone()])
            .expect("MockProver failed to run");
        match prover.verify() {
            Ok(()) => println!("MockProver: verifier circuit 0 is SATISFIED"),
            Err(e) => println!("MockProver: verifier circuit 0 FAILED: {:#?}", e),
        }
    }

    // --- 4. Fold the K_PROOFS verifier-circuit instances together. ---
    let mut transcript = CircuitTranscript::<PoseidonState<F>>::init();

    let prover_init = Instant::now();
    let protogalaxy = ProtogalaxyProver::<F, KZGCommitmentScheme<E>, { K as usize }>::init(
        &outer_srs,
        pk.clone(),
        verifier_circuits[0].clone(),
        1,
        &[&[], &public_inputs[0]],
        &mut rng,
        &mut transcript,
    )
    .expect("Failed to initialise folder");
    println!("Time for ProtogalaxyProver::init: {:?}", prover_init.elapsed());

    let prover_fold = Instant::now();
    let protogalaxy = protogalaxy
        .fold(
            &outer_srs,
            &pk,
            verifier_circuits[1..].to_vec(),
            1,
            &[
                &[&[], &public_inputs[1]],
                &[&[], &public_inputs[2]],
                &[&[], &public_inputs[3]],
            ],
            &mut rng,
            &mut transcript,
        )
        .expect("Failed to fold verifier-circuit instances");
    println!("Time for ProtogalaxyProver::fold: {:?}", prover_fold.elapsed());

    // --- 5. Fold (and check) on the verifier side. ---
    let folded_proof_bytes = transcript.finalize();
    let mut transcript = CircuitTranscript::<PoseidonState<F>>::init_from_bytes(&folded_proof_bytes);

    let verifier_init = Instant::now();
    let protogalaxy_verifier = ProtogalaxyVerifier::<F, KZGCommitmentScheme<E>, { K as usize }>::init(
        &vk,
        &[&[C::identity()]],
        &[&[&public_inputs[0]]],
        &mut transcript,
    )
    .expect("Failed - unexpected");
    println!("Time for ProtogalaxyVerifier::init: {:?}", verifier_init.elapsed());

    let verifier_fold = Instant::now();
    let protogalaxy_verifier = protogalaxy_verifier
        .fold(
            &vk,
            &[&[C::identity()]],
            &[
                &[&public_inputs[1]],
                &[&public_inputs[2]],
                &[&public_inputs[3]],
            ],
            &mut transcript,
        )
        .expect("Failed to fold instances by the verifier");
    println!("Time for ProtogalaxyVerifier::fold: {:?}", verifier_fold.elapsed());

    let is_sat = Instant::now();
    protogalaxy_verifier
        .is_sat(
            &outer_srs,
            &vk,
            &pk.ev.clone(),
            protogalaxy.folded_trace.clone(),
            &protogalaxy.folding_pk.l0,
            &protogalaxy.folding_pk.l_last,
            &protogalaxy.folding_pk.l_active_row,
            &protogalaxy.folding_pk.permutation_pk_cosets,
        )
        .expect("Folding finalizer failed");
    println!("Time for ProtogalaxyVerifier::is_sat: {:?}", is_sat.elapsed());

    println!("IVC folding was a success");

    // --- 6. Soundness check: verifying a TAMPERED folded proof must fail.
    // Here the fold itself was entirely legitimate; we corrupt the resulting
    // proof bytes (as an on-the-wire attacker would) before handing them to
    // the verifier, and confirm it doesn't accept them. ---
    {
        let mut tampered_bytes = folded_proof_bytes.clone();
        let mid = tampered_bytes.len() / 2;
        tampered_bytes[mid] ^= 0xFF;

        let default_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));

        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut transcript =
                CircuitTranscript::<PoseidonState<F>>::init_from_bytes(&tampered_bytes);

            ProtogalaxyVerifier::<F, KZGCommitmentScheme<E>, { K as usize }>::init(
                &vk,
                &[&[C::identity()]],
                &[&[&public_inputs[0]]],
                &mut transcript,
            )?
            .fold(
                &vk,
                &[&[C::identity()]],
                &[
                    &[&public_inputs[1]],
                    &[&public_inputs[2]],
                    &[&public_inputs[3]],
                ],
                &mut transcript,
            )?
            .is_sat(
                &outer_srs,
                &vk,
                &pk.ev.clone(),
                protogalaxy.folded_trace.clone(),
                &protogalaxy.folding_pk.l0,
                &protogalaxy.folding_pk.l_last,
                &protogalaxy.folding_pk.l_active_row,
                &protogalaxy.folding_pk.permutation_pk_cosets,
            )
        }));

        std::panic::set_hook(default_hook);

        match outcome {
            Err(_) => println!("Verifying a tampered folded proof correctly PANICKED (rejected)"),
            Ok(Err(e)) => {
                println!("Verifying a tampered folded proof correctly returned an error: {e:?}")
            }
            Ok(Ok(())) => panic!("UNSOUND: a tampered folded proof was verified successfully!"),
        }
    }

    // --- 7. Soundness check: fold a verifier circuit that witnesses
    // `tampered_inner_proof` together with three otherwise-correct verifier
    // circuits, on the PROVER side. The prover dishonestly claims the
    // original (valid) public input for that slot while actually supplying a
    // corrupted inner proof. If the fold itself doesn't already reject this
    // (the prover's internal consistency checks currently panic rather than
    // return an `Err`), then the resulting fold MUST fail verification. ---
    {
        let (tampered_output, tampered_bytes) = tampered_inner_proof.clone();
        let bad_verifier_circuit = VerifierCircuit {
            inner_vk: (
                inner_vk.get_domain().clone(),
                inner_vk.cs().clone(),
                Value::known(inner_vk.transcript_repr()),
            ),
            inner_committed_instance: Value::known(C::identity()),
            inner_instance: Value::known(tampered_output),
            inner_proof: Value::known(tampered_bytes),
        };

        let default_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));

        let fold_outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || -> Result<(ProtogalaxyProver<F, KZGCommitmentScheme<E>, { K as usize }>, Vec<u8>), Error> {
            let mut rng = ChaCha8Rng::from_entropy();
            let mut transcript = CircuitTranscript::<PoseidonState<F>>::init();
            let protogalaxy = ProtogalaxyProver::<F, KZGCommitmentScheme<E>, { K as usize }>::init(
                &outer_srs,
                pk.clone(),
                bad_verifier_circuit,
                1,
                &[&[], &public_inputs[0]],
                &mut rng,
                &mut transcript,
            )?;

            let protogalaxy = protogalaxy.fold(
                &outer_srs,
                &pk,
                vec![
                    verifier_circuits[1].clone(),
                    verifier_circuits[2].clone(),
                    verifier_circuits[3].clone(),
                ],
                1,
                &[
                    &[&[], &public_inputs[1]],
                    &[&[], &public_inputs[2]],
                    &[&[], &public_inputs[3]],
                ],
                &mut rng,
                &mut transcript,
            )?;

            Ok((protogalaxy, transcript.finalize()))
        }));

        std::panic::set_hook(default_hook);

        match fold_outcome {
            Err(_) => println!(
                "Step 7: folding a tampered inner proof correctly PANICKED \
                 (rejected by the prover's internal consistency checks)"
            ),
            Ok(Err(e)) => {
                println!("Step 7: folding a tampered inner proof correctly returned an error: {e:?}")
            }
            Ok(Ok((bad_protogalaxy, bad_proof_bytes))) => {
                println!(
                    "Step 7: folding a tampered inner proof SUCCEEDED -- asserting that \
                     verification of it fails"
                );

                let default_hook = std::panic::take_hook();
                std::panic::set_hook(Box::new(|_| {}));

                let verify_outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut transcript =
                        CircuitTranscript::<PoseidonState<F>>::init_from_bytes(&bad_proof_bytes);

                    ProtogalaxyVerifier::<F, KZGCommitmentScheme<E>, { K as usize }>::init(
                        &vk,
                        &[&[C::identity()]],
                        &[&[&public_inputs[0]]],
                        &mut transcript,
                    )?
                    .fold(
                        &vk,
                        &[&[C::identity()]],
                        &[
                            &[&public_inputs[1]],
                            &[&public_inputs[2]],
                            &[&public_inputs[3]],
                        ],
                        &mut transcript,
                    )?
                    .is_sat(
                        &outer_srs,
                        &vk,
                        &pk.ev.clone(),
                        bad_protogalaxy.folded_trace,
                        &bad_protogalaxy.folding_pk.l0,
                        &bad_protogalaxy.folding_pk.l_last,
                        &bad_protogalaxy.folding_pk.l_active_row,
                        &bad_protogalaxy.folding_pk.permutation_pk_cosets,
                    )
                }));

                std::panic::set_hook(default_hook);

                match verify_outcome {
                    Err(_) => println!(
                        "Step 7: verification correctly PANICKED (rejected the tampered fold)"
                    ),
                    Ok(Err(e)) => {
                        println!("Step 7: verification correctly returned an error: {e:?}")
                    }
                    Ok(Ok(())) => panic!(
                        "UNSOUND: folding AND verifying a tampered inner proof both succeeded!"
                    ),
                }
            }
        }
    }
}
