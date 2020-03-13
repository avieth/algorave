use std::thread;
use std::io::Read;

extern crate algorave;

use algorave::lang::update::Update;
use algorave::lang::codec::{*};
use algorave::backend::jack as backend;

use jack;

mod examples;

struct StdioSource {
    stdin_bytes: std::io::Bytes<std::io::Stdin>
}

impl StdioSource {
    fn new(stdin: std::io::Stdin) -> StdioSource {
        return StdioSource { stdin_bytes: stdin.bytes() };
    }
}

#[derive(Debug)]
enum StdioSourceError {
    EndOfStream,
    Error(std::io::Error)
}

impl std::convert::From<std::io::Error> for StdioSourceError {
    fn from(io_error: std::io::Error) -> StdioSourceError {
        return StdioSourceError::Error(io_error);
    }
}

impl Source for StdioSource {
    type Token = u8;
    type Error = StdioSourceError;
    fn get(&mut self) -> Result<u8, StdioSourceError> {
        let mbyte: Option<Result<u8, std::io::Error>> = self.stdin_bytes.next();
        match mbyte {
            None => { return Err(StdioSourceError::EndOfStream); }
            Some(result) => {
                let byte = result?;
                return Ok(byte);
            }
        }
    }
}


/// Tries to decode a complete Update from stdin.
/// Retries if it fails to decode, and gives None when/if stdin is EOF.
/// In case of IO errors, 
fn get_update_from_stdin() -> Result<Update, StdioSourceError> {
    let mut src = StdioSource::new(std::io::stdin());
    loop {
        let result = decode_update(&mut src);
        println!("Decoded update {:?}", result);
        match result {
            Ok(update) => {
                return Ok(update);
            }
            // This includes StdioSourceError::EndOfStream
            Err(DecodeError::NoRead(err)) => {
                return Err(err);
            }
            Err(DecodeError::NoParse(_err)) => {
                // TODO stderr debug message?
                continue;
            }
        }
    }
}

fn shutdown_thread(mut shutdown_notifier: backend::ShutdownNotifier) {
    shutdown_notifier.wait_for_shutdown();
}

fn main() {

    let (client, _) = jack::Client::new("algorave", jack::ClientOptions::NO_START_SERVER).unwrap();

    let out_left = client.register_port("out_left", jack::AudioOut).unwrap();
    let out_right = client.register_port("out_right", jack::AudioOut).unwrap();
    let out_midi = client.register_port("out_midi", jack::MidiOut).unwrap();
    let in_midi = client.register_port("in_midi", jack::MidiIn).unwrap();

    // TODO find some CLI parsing library and make the memory sizes, and
    // number of ports, configurable at launch.

    // 256 mibibytes
    let memory_size = 256 * 2usize.pow(20);
    // 256 kibibytes
    let midi_buffer_size = 256 * 2usize.pow(10);
    let ports = vec![
        backend::SomePort::AudioOut(0, out_left),
        backend::SomePort::AudioOut(1, out_right),
        backend::SomePort::MidiIn(3, midi_buffer_size, in_midi),
        backend::SomePort::MidiOut(3, midi_buffer_size, out_midi)
    ];

    match backend::run(memory_size, client, ports) {
        Err(_) => { panic!("error starting JACK controller"); }
        Ok(mut controller) => {
            // The main control loop: take updates from stdin and give them
            // to the controller to process.
            // If for some reason this part should fail, the system will not
            // stop running.
            let shutdown_notifier = controller.shutdown_notifier();
            let wait_t    = thread::spawn(move || { shutdown_thread(shutdown_notifier); });
            // It was observed that if we do the stdin control loop in a
            // separate thread
            //   let control_t = thread::spawn(move || { control_thread(controller); });
            //   control_t.join().unwrap();
            // the JACK process callback stops whenever the loop stops.
            // But doing the loop here in the main thread is fine...
            // No idea why.
            loop {
                if let Err(_) = controller.next(get_update_from_stdin) {
                    break;
                }
            }
            wait_t.join().unwrap();
        }
    }

}
