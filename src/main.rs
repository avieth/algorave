use std::thread;
use std::time;
use jack;

// Apparently this is idiomatic? Even though algorave is this crate?
extern crate algorave;
use algorave::backend::jack as jj;

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
    controller.add_input("mic", 2);
    let prog1 = jj::Program::from_array(&[
        // Sample the input.
        jj::Term::Input(2),
        // Compute a sawtooth every second between -0.5 and 0.5
        jj::Term::Now(),
        jj::Term::ConstInt(48000),
        jj::Term::Mod(),
        jj::Term::Cast(),
        jj::Term::ConstFloat(48000.0),
        jj::Term::Divide(),
        jj::Term::ConstFloat(0.5),
        jj::Term::Subtract(),
        // Then use it as an envelope by multiplying the input wave, effectively
        // attenuating it.
        jj::Term::Multiply()
    ]);
    let prog2 = jj::Program::from_array(&[
        // Put now on the stack.
        jj::Term::Now(),
        // Take it modulo 48000 (the sample rate).
        jj::Term::ConstInt(48000),
        jj::Term::Mod(),
        // Cast it to a float.
        jj::Term::Cast(),
        // Divide it by 48000
        jj::Term::ConstFloat(48000.0),
        jj::Term::Divide(),
        // Multiply by 2Pi
        jj::Term::ConstFloat(2.0 * std::f64::consts::PI),
        jj::Term::Multiply(),
        // Take the sine.
        jj::Term::Sine()
    ]);
    controller.set_program(0, prog1);
    controller.set_program(1, prog2);
    controller.stage();

    thread::sleep(time::Duration::from_millis(1000 * 1000 * 1000));
}
