//! The ODE solver reproduces closed-form and reference solutions.

use lfns_model::{
    DeterministicSimulator, Expression, OdeSettings, Quantity, Reaction, ReactionNetwork, SimulationModel, Species,
};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn quantity(id: &str, value: f64) -> Quantity {
    Quantity {
        id: id.to_owned(),
        value,
    }
}

/// The value buffer of `model` at time zero with the given parameters.
fn initial_values(model: &SimulationModel, parameters: &[f64]) -> Vec<f64> {
    let mut values = vec![0.0; model.value_width()];
    model.write_initial_values(parameters, &mut values);
    values
}

fn reaction(id: &str, net_changes: &[(usize, f64)], rate: &str) -> Result<Reaction, lfns_model::ModelError> {
    Ok(Reaction {
        id: id.to_owned(),
        net_changes: net_changes.to_vec(),
        rate: Expression::parse(rate)?,
    })
}

#[test]
fn birth_death_follows_its_exponential_approach_to_steady_state() -> TestResult {
    // dm/dt = k - g m with m(0) = 0 has the solution m(t) = k/g (1 - exp(-g t)).
    let (k, g) = (1.3, 0.2);
    let reactions = vec![
        reaction("birth", &[(0, 1.0)], "k")?,
        reaction("death", &[(0, -1.0)], "g * m")?,
    ];
    let network = ReactionNetwork::new(
        vec![Species::amount("m", 0.0)],
        vec![quantity("k", k), quantity("g", g)],
        reactions,
    );
    let model = SimulationModel::new(&network, &[])?;
    let times = [0.0, 0.5, 2.0, 10.0, 50.0];
    let solution =
        DeterministicSimulator::new(&model, OdeSettings::default()).solve(&initial_values(&model, &[k, g]), &times)?;
    for (time, state) in times.iter().zip(&solution) {
        let exact = k / g * (1.0 - (-g * time).exp());
        assert!((state[0] - exact).abs() < 1e-6, "t = {time}: {} vs {exact}", state[0]);
    }
    Ok(())
}

#[test]
fn stiff_robertson_kinetics_match_the_reference_solution() -> TestResult {
    // Robertson's problem, the standard stiff test: rate constants span nine
    // orders of magnitude. Reference values at t = 40 from the literature.
    let species = vec![
        Species::amount("a", 1.0),
        Species::amount("b", 0.0),
        Species::amount("c", 0.0),
    ];
    let reactions = vec![
        reaction("a_to_b", &[(0, -1.0), (1, 1.0)], "0.04 * a")?,
        reaction("b_to_c", &[(1, -1.0), (2, 1.0)], "3e7 * b ^ 2")?,
        reaction("b_to_a", &[(1, -1.0), (0, 1.0)], "1e4 * b * c")?,
    ];
    let model = SimulationModel::new(&ReactionNetwork::new(species, Vec::new(), reactions), &[])?;
    let solution = DeterministicSimulator::new(&model, OdeSettings::default())
        .solve(&initial_values(&model, &[]), &[0.4, 40.0])?;
    let reference = [0.715_827_068_7, 9.185_534_764e-6, 0.284_163_745_7];
    for (value, expected) in solution[1][..3].iter().zip(reference) {
        assert!((value / expected - 1.0).abs() < 1e-5, "{value} vs {expected}");
    }
    // Mass is conserved throughout.
    assert!((solution[0][..3].iter().sum::<f64>() - 1.0).abs() < 1e-8);
    Ok(())
}

#[test]
fn exhausting_the_step_limit_is_reported_as_a_failure() -> TestResult {
    let reactions = vec![
        reaction("birth", &[(0, 1.0)], "k")?,
        reaction("death", &[(0, -1.0)], "g * m")?,
    ];
    let network = ReactionNetwork::new(
        vec![Species::amount("m", 0.0)],
        vec![quantity("k", 1.0), quantity("g", 0.1)],
        reactions,
    );
    let model = SimulationModel::new(&network, &[])?;
    let settings = OdeSettings::default().with_max_steps(3);
    let outcome = DeterministicSimulator::new(&model, settings).solve(&initial_values(&model, &[1.0, 0.1]), &[100.0]);
    assert!(
        matches!(outcome, Err(lfns_model::SimulationFailure::SolverFailed { .. })),
        "{outcome:?}"
    );
    Ok(())
}
