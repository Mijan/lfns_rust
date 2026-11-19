//! Independent random number streams derived from one run seed.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Hands out one independent generator per proposal and per proposal fit.
///
/// Every proposal owns the stream numbered by its global index, so its
/// outcome does not depend on which thread evaluates it or in which order.
/// Fits count their streams down from the top of the stream space, which
/// proposal indices never reach.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RngStreams {
    seed: u64,
}

impl RngStreams {
    /// Stream of the fit of iteration 0; iteration `i` uses this minus `i`.
    const FIRST_FIT_STREAM: u64 = u64::MAX;

    pub(crate) fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Generator of the proposal with global index `proposal_index`.
    pub(crate) fn proposal_rng(self, proposal_index: u64) -> ChaCha8Rng {
        self.stream(proposal_index)
    }

    /// Generator for fitting the proposal of `iteration`.
    pub(crate) fn fit_rng(self, iteration: u32) -> ChaCha8Rng {
        self.stream(Self::FIRST_FIT_STREAM - u64::from(iteration))
    }

    fn stream(self, stream: u64) -> ChaCha8Rng {
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        rng.set_stream(stream);
        rng
    }
}
