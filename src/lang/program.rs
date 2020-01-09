#[derive(Copy, Clone)]
pub enum Term<Identifier, Number> {
    /// A constant signal.
    Constant(Number),
    /// Sine wave. Frequency and phase shift are expected on the stack (in that
    /// order).
    Sine(),
    Sawtooth(),
    Square(),
    Triangle(),
    /// An input sample.
    Input(Identifier),
    Add(),
    Subtract(),
    Multiply(),
    Divide(),
    Mod(),
}

#[derive(Clone)]
/// A program is a sequence of terms, to be interpreted by a stack evaluator.
pub struct Program<Identifier, Number> {
    prog_terms: Vec<Term<Identifier, Number>>
}

impl<Identifier, Number> Program<Identifier, Number> {

    pub fn from_array(terms: &[Term<Identifier, Number>]) -> Program<Identifier, Number>
        where Number: Clone, Identifier: Clone
    {
        return Program { prog_terms: Vec::from(terms) };
    }

    pub fn terms(&self) -> &Vec<Term<Identifier, Number>> {
        return &self.prog_terms;
    }

    /// The length of the program. Can be useful to know because in order to
    /// evaluate it, one only needs a stack of at most this size.
    pub fn length(&self) -> usize {
        return self.prog_terms.len();
    }
}
