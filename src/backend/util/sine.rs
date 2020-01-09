const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

/// Lookup table for sin() between 0 and 2*PI.
pub struct LookupTable {
    array: Box<[f64]>,
    /// The size of the array.
    samples: usize,
    /// The "step" of the array (2*PI divided by the size, i.e. number of
    /// samples). The change in the domain of sin between adjacent samples.
    step: f64
}

impl LookupTable {

    /// Make a lookup table with a given number of samples.
    pub fn new(sample_rate: usize) -> LookupTable {
        let step = TWO_PI / (sample_rate as f64);
        let mut vec = Vec::with_capacity(sample_rate);
        let mut t : f64 = 0.0;
        for i in 0..sample_rate {
            vec.insert(i, t.sin());
            t += step;
        }
        return LookupTable {
            array: vec.into_boxed_slice(),
            samples: sample_rate,
            step: step
        };
    }

    /// Sample the lookup table to approximate t.sin().
    pub fn at(&self, t: f64) -> f64 {
        let sampleno = f64::round(t / self.step) as usize;
        return (*self.array)[sampleno % self.samples];
    }
}
