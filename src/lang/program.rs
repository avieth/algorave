use std::path::PathBuf;

#[derive(Copy, Clone)]
pub enum Term<Name, Number> {
    /// A constant signal.
    Constant(Number),
    /// Sine wave. Frequency and phase shift are expected on the stack (in that
    /// order).
    Sine(),
    Sawtooth(),
    Square(),
    Triangle(),
    /// An input sample.
    Input(Name),
    Add(),
    Subtract(),
    Multiply(),
    Divide(),
    Mod(),
}

#[derive(Clone)]
/// The pure-functional FRP part of the language:
/// a program is a sequence of terms, to be interpreted by a stack evaluator.
pub struct Program<Name, Number> {
    prog_terms: Vec<Term<Name, Number>>
}

impl<Name, Number> Program<Name, Number> {

    pub fn from_array(terms: &[Term<Name, Number>]) -> Program<Name, Number>
        where Number: Clone, Name: Clone
    {
        return Program { prog_terms: Vec::from(terms) };
    }

    pub fn terms(&self) -> &Vec<Term<Name, Number>> {
        return &self.prog_terms;
    }

    /// The length of the program. Can be useful to know because in order to
    /// evaluate it, one only needs a stack of at most this size.
    pub fn length(&self) -> usize {
        return self.prog_terms.len();
    }
}

pub enum Resource {
    /// Name of the input as it should appear to the outside world.
    /// The name used by the language is given in the `Add` constructor of
    /// `Command`.
    AudioInput(String),
    /// String argument is as for `AudioInput`.
    AudioOutput(String),
    MidiInput(String),
    MidiOutput(String),
    /// Random-access PCM data from disk.
    /// TODO more parameters in the future? For now we can just assume (and
    /// even check in the decoded data!) that it's 44.1KHz 16 bit uncompressed
    /// LPCM.
    /// Of course, the entire thing will be loaded into memory when the
    /// corresponding command is evaluated (before modifying the running system)
    /// so the programmer will have to be conscious of memory resource use.
    Block(PathBuf)
}

pub enum Command<Name, Number> {
    Add(Name, Resource),
    Drop(Name),
    /// Assign a program to an output. The identifier must be an output.
    Set(Name, Program<Name, Number>)
}

/// The imperative part of the language: a sequence of commands.
///
/// This type is what "acts" on the actual denotation of the program (the
/// `DeclarativeState`, defined in the JACK backend). It's a monoid action, one
/// which factors into "additive" parts, their inverse "subtractive" parts, and
/// the "neither" parts, as clearly seen from the definition of `Command`:
/// - Add is the additive part
/// - Drop is the subtractive part
/// - Set is neither
/// So given a list of `Command`, partitioning by these constructors gives the
/// factoring.
///
/// The additive and subtractive parts do not in general commute. If names are
/// never shadowed, then all additive and subtractive parts commute with
/// each other. Regardless of naming, the assignment part is not commutative,
/// but it does commute with additive and subtractive parts (assigning to name
/// which was dropped means no change).
pub struct Instructions<Name, Number> {
    instr_commands: Vec<Command<Name, Number>>
}

impl<Name, Number> Instructions<Name, Number> {
    pub fn commands(&self) -> &Vec<Command<Name, Number>> {
        return &self.instr_commands;
    }
}

// TODO we should give the "proper" "pure functional" definition of the
// language.
// The semantics are of "changes". An `Instructions` corresponds to an
//
//   State -> State
//
// which we shall represent concretely as a big HashMap that describes all of
// the changes. Let's call it `Endo State`.
//
// But we shall also be dealing with [changes on [changes to the state]]
// bracketed for clarity.
//
//   Endo State -> Endo State
//
// or in other words
//
//   Endo (Endo State)
//
// That's because, for practical reasons, we want the performer to be able to
// stage / typecheck / prepare changes (type `Endo State`) before they are
// commited (the process thread takes them on).
//
// That's to say: there is an `Endo State` held by the program, and the
// performer may iteratively build this up by giving `Endo (Endo State)`s
// (ultimately encoded in their concrete syntax of choice).
//
// Then, they may signal that whatever the current `Endo State` is, it should
// be commited, giving a realization in side-effects (the system that the
// performer may observe in physical reality is changed).
//
//
//
//
//
//












