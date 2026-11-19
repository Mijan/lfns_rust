//! The measurements of one observed trajectory.

/// One measured value of one observable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reading {
    /// Position of the observable in the model.
    pub observable_index: usize,
    /// The measured value.
    pub measurement: f64,
}

/// The measurements of one experimental unit (for example one cell or one
/// bioreactor run) over time: strictly increasing time points, each with its readings.
#[derive(Debug, Clone, PartialEq)]
pub struct Trajectory {
    condition_index: usize,
    time_points: Vec<(f64, Vec<Reading>)>,
}

impl Trajectory {
    /// Builds the trajectory measured under the condition at
    /// `condition_index` from `(time, reading)` pairs in any order; readings
    /// taken at the same time are grouped.
    #[must_use]
    pub fn from_readings(condition_index: usize, mut readings: Vec<(f64, Reading)>) -> Self {
        readings.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut time_points: Vec<(f64, Vec<Reading>)> = Vec::new();
        for (time, reading) in readings {
            match time_points.last_mut() {
                Some((last_time, group)) if last_time.total_cmp(&time).is_eq() => group.push(reading),
                _ => time_points.push((time, vec![reading])),
            }
        }
        Self {
            condition_index,
            time_points,
        }
    }

    /// Position of the experimental condition the trajectory was measured
    /// under among the conditions of its problem.
    #[must_use]
    pub fn condition_index(&self) -> usize {
        self.condition_index
    }

    /// The time points in increasing order, each with its readings.
    #[must_use]
    pub fn time_points(&self) -> &[(f64, Vec<Reading>)] {
        &self.time_points
    }
}
