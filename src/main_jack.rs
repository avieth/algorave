use jack;

// Apparently this is idiomatic? Even though algorave is this crate?
extern crate algorave;
use crate::backend::jack as backend;

fn main() {

    let client = jack::Client::new("algorave", jack::ClientOptions::NO_START_SERVER).unwrap();

    let out_left = client.register_port("out_left", jack::AudioOut).unwrap();
    let out_right = client.register_port("out_right", jack::AudioOut).unwrap();
    let out_midi = client.register_port("out_midi", jack::MidiOut).unwrap();
    let in_midi = client.register_port("in_midi", jack::MidiIn).unwrap();

    // 256 mibibytes
    let memory_size = 256 * 2.pow(20);
    // 256 kibibytes
    let midi_buffer_size = 256 * 2.pow(10);
    let ports = vec![
        backend::AudioOut(0, out_left),
        backend::AudioOut(1, out_right),
        backend::MidiIn(3, midi_buffer_size, in_midi),
        backend::MidiOut(3, midi_buffer_size, out_midi);
    ];

    let controller = backend::run(memory_size, client, ports).unwrap();

}
