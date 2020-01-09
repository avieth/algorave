use std::thread;
use std::time;
use jack;

// Apparently this is idiomatic? Even though algorave is this crate?
extern crate algorave;
use algorave::backend::jack as jj;
use algorave::lang::program::{Term, Program};

fn main() {
    let client_result = jack::Client::new(
        "algorave",
        jack::ClientOptions::NO_START_SERVER
    );
    let (client, _status) = client_result.unwrap();

    // Run the processor, giving us a controller.
    // A JACK client called "algorave" will appear.
    let mut controller = jj::Processor::run(client);
    controller.add_output("left", 0);
    controller.add_output("right", 1);
    controller.add_output("fooo", 2);
    controller.add_output("barr", 3);
    controller.add_input("mic", 4);
    // Program 1: take the input and make an envelope.
    let prog1 = Program::from_array(&[
        // Sample the input.
        Term::Input(4),
        // Make an envelope.
        Term::Constant(2.0), // Frequency
        Term::Constant(0.0), // Phase
        Term::Sine(),
        Term::Multiply()
    ]);
    // Program 2 is just a 2Hz sine wave
    let prog2 = Program::from_array(&[
        Term::Constant(2.0), // Frequency
        Term::Constant(0.0), // Phase
        Term::Sine()
    ]);
    let prog3 = Program::from_array(&[
        Term::Constant(2.0),
        Term::Constant(0.0),
        // Derive a ramp from a sawtooth:
        Term::Sawtooth(),
        // Flip it and scale it
        Term::Constant(-0.5),
        Term::Multiply(),
        Term::Constant(0.5),
        Term::Add()
    ]);
    let prog4 = Program::from_array(&[
        Term::Constant(2.0),
        Term::Constant(0.0),
        Term::Triangle(),
        Term::Constant(0.5),
        Term::Multiply()
    ]);
    controller.set_program(0, prog1);
    controller.set_program(1, prog2);
    controller.set_program(2, prog3);
    controller.set_program(3, prog4);
    controller.stage();

    thread::sleep(time::Duration::from_millis(1000 * 1000 * 1000));
}
