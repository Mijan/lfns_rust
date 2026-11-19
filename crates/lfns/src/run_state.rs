//! Everything a run has produced so far.

use crate::evidence_accumulator::EvidenceAccumulator;
use crate::evidence_estimate::EvidenceEstimate;
use crate::iteration_report::IterationReport;
use crate::lfns_error::LfnsError;
use crate::live_set::LiveSet;
use crate::particle::Particle;
use crate::particle_collector::Collection;
use crate::point_matrix::PointMatrix;
use crate::population::Population;
use crate::posterior::Posterior;

/// The live and dead particles of a run and the evidence moments derived
/// from them.
///
/// A state is obtained from [`NestedSampler::initialize`] or
/// [`CheckpointStore::load_state`] and advanced only by
/// [`NestedSampler::run`], so its parts are always consistent.
///
/// [`NestedSampler::initialize`]: crate::NestedSampler::initialize
/// [`NestedSampler::run`]: crate::NestedSampler::run
/// [`CheckpointStore::load_state`]: crate::CheckpointStore::load_state
#[derive(Debug, Clone, PartialEq)]
pub struct RunState {
    population: Population,
    n_dimensions: usize,
    n_proposals: u64,
    latest_acceptance_rate: Option<f64>,
    live: LiveSet,
    dead: Vec<Particle>,
    dead_log_volume_weights: Vec<f64>,
    accumulator: EvidenceAccumulator,
}

impl RunState {
    /// Starts a run from its initial population.
    pub(crate) fn begin(population: Population, initial: Collection) -> Result<Self, LfnsError> {
        Self::restore(population, initial.particles, &[], initial.n_proposals, None)
    }

    /// Rebuilds a state from stored particles by replaying the iterations.
    /// `latest_acceptance_rate` is that of the last iteration replayed,
    /// `None` if there is none.
    pub(crate) fn restore(
        population: Population,
        live: Vec<Particle>,
        dead: &[Particle],
        n_proposals: u64,
        latest_acceptance_rate: Option<f64>,
    ) -> Result<Self, LfnsError> {
        let n_dimensions = live.first().map_or(0, |particle| particle.unit_coordinates().len());
        Self::check_dimensions(live.iter().chain(dead), n_dimensions)?;
        let batch_size = population.n_replaced() as usize;
        if dead.len() % batch_size != 0 {
            let expected = dead.len().next_multiple_of(batch_size);
            return Err(LfnsError::ParticleCountMismatch {
                context: "dead particles",
                expected,
                actual: dead.len(),
            });
        }
        let mut state = Self {
            population,
            n_dimensions,
            n_proposals,
            latest_acceptance_rate,
            live: LiveSet::from_particles(population, live)?,
            dead: Vec::with_capacity(dead.len()),
            dead_log_volume_weights: Vec::new(),
            accumulator: EvidenceAccumulator::new(population),
        };
        dead.chunks(batch_size)
            .for_each(|batch| state.record_dead_batch(batch.to_vec()));
        Ok(state)
    }

    /// Number of completed iterations.
    #[must_use]
    pub fn iteration(&self) -> u32 {
        // One weight is stored per iteration and iterations are counted in u32.
        u32::try_from(self.dead_log_volume_weights.len()).unwrap_or(u32::MAX)
    }

    /// Number of proposals evaluated since the start of the run.
    #[must_use]
    pub fn n_proposals(&self) -> u64 {
        self.n_proposals
    }

    /// Fraction of the most recent iteration's proposals that were
    /// accepted; `None` before the first iteration.
    #[must_use]
    pub fn latest_acceptance_rate(&self) -> Option<f64> {
        self.latest_acceptance_rate
    }

    /// Number of parameters of each particle.
    #[must_use]
    pub fn n_dimensions(&self) -> usize {
        self.n_dimensions
    }

    /// Population sizes of the run.
    #[must_use]
    pub fn population(&self) -> Population {
        self.population
    }

    /// The live particles.
    #[must_use]
    pub fn live(&self) -> &LiveSet {
        &self.live
    }

    /// All dead particles in order of removal.
    #[must_use]
    pub fn dead(&self) -> &[Particle] {
        &self.dead
    }

    /// The dead particles of the most recent iteration.
    #[must_use]
    pub fn latest_dead_batch(&self) -> &[Particle] {
        let batch_size = (self.population.n_replaced() as usize).min(self.dead.len());
        &self.dead[self.dead.len() - batch_size..]
    }

    /// Estimates the evidence from the dead and live particles.
    #[must_use]
    pub fn estimate_evidence(&self) -> EvidenceEstimate {
        let live = self.live.summarize_likelihoods(self.accumulator.log_n_live());
        self.accumulator.estimate(live)
    }

    /// Assembles the weighted posterior sample.
    ///
    /// # Errors
    ///
    /// Returns [`LfnsError::NoLikelihoodMass`] if every particle has zero
    /// likelihood, in which case no posterior exists.
    pub fn assemble_posterior(&self) -> Result<Posterior, LfnsError> {
        let log_evidence = self.estimate_evidence().log_evidence();
        if log_evidence == f64::NEG_INFINITY {
            return Err(LfnsError::NoLikelihoodMass);
        }
        let batch_size = self.population.n_replaced() as usize;
        let dead_volume_weights = self
            .dead_log_volume_weights
            .iter()
            .flat_map(|w| std::iter::repeat_n(*w, batch_size));
        let live_volume_weight = self.accumulator.log_volume() - self.accumulator.log_n_live();
        let volume_weights = dead_volume_weights.chain(std::iter::repeat(live_volume_weight));
        let particles: Vec<Particle> = self.dead.iter().chain(self.live.particles()).cloned().collect();
        let log_weights = particles
            .iter()
            .zip(volume_weights)
            .map(|(p, w)| p.log_likelihood() + w - log_evidence);
        let log_weights = log_weights.collect();
        Ok(Posterior::from_weighted_particles(particles, log_weights))
    }

    /// Coordinates of the particles currently alive.
    pub(crate) fn live_coordinates(&self) -> PointMatrix {
        self.live.coordinates(self.n_dimensions)
    }

    /// Moves the lowest particles to the dead set and returns the
    /// log-likelihood threshold: that of the last particle removed.
    pub(crate) fn retire_lowest(&mut self) -> f64 {
        let batch = self.live.remove_lowest();
        let log_threshold = batch.last().map_or(f64::NEG_INFINITY, Particle::log_likelihood);
        self.record_dead_batch(batch);
        log_threshold
    }

    /// Refills the live set and reports on the finished iteration.
    pub(crate) fn admit_replacements(&mut self, log_threshold: f64, collection: Collection) -> IterationReport {
        self.n_proposals += collection.n_proposals;
        self.live.add_replacements(collection.particles);
        let report = IterationReport::of_iteration(
            self.iteration(),
            log_threshold,
            self.population.n_replaced(),
            (collection.n_proposals, collection.n_failures),
            self.estimate_evidence(),
        );
        self.latest_acceptance_rate = Some(report.acceptance_rate());
        report
    }

    fn record_dead_batch(&mut self, batch: Vec<Particle>) {
        let log_likelihoods: Vec<f64> = batch.iter().map(Particle::log_likelihood).collect();
        self.dead_log_volume_weights
            .push(self.accumulator.log_next_dead_weight());
        self.accumulator.record_iteration(&log_likelihoods);
        self.dead.extend(batch);
    }

    fn check_dimensions<'a>(
        mut particles: impl Iterator<Item = &'a Particle>,
        expected: usize,
    ) -> Result<(), LfnsError> {
        match particles.find(|particle| particle.unit_coordinates().len() != expected) {
            None => Ok(()),
            Some(particle) => Err(LfnsError::DimensionMismatch {
                context: "particles of a run",
                expected,
                actual: particle.unit_coordinates().len(),
            }),
        }
    }
}
