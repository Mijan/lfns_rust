//! A contiguous set of points handed to proposal fitters.

/// Points of equal dimension stored row by row in one flat buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct PointMatrix {
    values: Vec<f64>,
    n_dimensions: usize,
}

impl PointMatrix {
    /// Collects points of dimension `n_dimensions` into one buffer.
    ///
    /// # Panics
    ///
    /// Panics if a point does not have `n_dimensions` coordinates; callers
    /// inside this crate guarantee it.
    pub(crate) fn from_points<'a>(points: impl Iterator<Item = &'a [f64]>, n_dimensions: usize) -> Self {
        let mut values = Vec::new();
        for point in points {
            assert_eq!(
                point.len(),
                n_dimensions,
                "points handed to a PointMatrix share one dimension"
            );
            values.extend_from_slice(point);
        }
        Self { values, n_dimensions }
    }

    /// Number of coordinates per point.
    #[must_use]
    pub fn n_dimensions(&self) -> usize {
        self.n_dimensions
    }

    /// Number of points.
    #[must_use]
    pub fn n_points(&self) -> usize {
        self.values.len().checked_div(self.n_dimensions).unwrap_or(0)
    }

    /// Iterates over the points, each as a slice of `n_dimensions` values.
    pub fn points(&self) -> impl ExactSizeIterator<Item = &[f64]> {
        self.values.chunks_exact(self.n_dimensions.max(1))
    }

    /// All coordinates, point after point.
    #[must_use]
    pub fn as_flat_slice(&self) -> &[f64] {
        &self.values
    }
}
