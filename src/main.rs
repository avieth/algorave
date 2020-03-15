use std::thread;
use std::io::Read;

extern crate clap;
use clap::{Arg, App};

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

fn to_hex_u32(st: String) -> Option<u32> {
    match u32::from_str_radix(&st, 16) {
        Err(_) => { return None; }
        Ok(it) => { return Some(it); }
    }
}

/// Validator function for the memory command line argument.
///
/// Ideally the CLI library would be more like Haskell's optparse-applicative
/// and give the validated thing (not necessarily a string) but oh well can't
/// have it all.
fn memory_arg_validator(st: String) -> Result<(), String> {
    match to_hex_u32(st) {
        None    => { return Err(String::from("expected 32-bit hex")); }
        Some(_) => { return Ok(()); }
    }
}


/// Validator function for the port command line arguments.
fn port_validator(st: String) -> Result<(), String> {
    let splits: Vec<&str> = st.split(':').collect();
    if splits.len() < 2 { return Err(String::from("expected id:name")); }
    let id_string = splits[0];
    match u8::from_str_radix(splits[0], 16) {
        Err(_) => { return Err(String::from("expected 8-bit hex identifier")); }
        // Shame we can't give the value here in the validator, we have to
        // parse it again. Shame shame shame -_-
        Ok(_)  => { return Ok(()); }
    }
}

fn main() {

    // Use the clap library to get command-line arguments for
    // - amount of memory available to object program
    // - how many JACK I/O ports and their identifiers.
    //
    // NB: duplicate identifiers and names are not excluded here. I believe
    // duplicate names will cause JACK to blow up. Duplicate identifiers
    // will be OK you just need to realize that later identifiers will shadow
    // earlier ones and the shadowed I/O ports will not be addressable by
    // object programs.
    // FIXME actually it refuses to run with duplicates...
    let matches = App::new("algorave JACK client")
          .version("0.1.0.0")
          .arg(Arg::with_name("memory")
               .long("memory")
               .long_help("32-bit hex giving the maximum memory address for object programs")
               .takes_value(true)
               .validator(memory_arg_validator)
               .required(true)
               )
          .arg(Arg::with_name("audio input")
               .long("audio-in")
               .long_help("id:name where id is an 8 bit hex number for an audio input")
               .multiple(true)
               .takes_value(true)
               .validator(port_validator)
               )
          .arg(Arg::with_name("audio output")
               .long("audio-out")
               .long_help("id:name where id is an 8 bit hex number for an audio output")
               .multiple(true)
               .takes_value(true)
               .validator(port_validator)
               )
          .arg(Arg::with_name("MIDI input")
               .long("midi-in")
               .long_help("id:name where id is an 8 bit hex number for a MIDI input")
               .multiple(true)
               .takes_value(true)
               .validator(port_validator)
               )
          .arg(Arg::with_name("MIDI output")
               .long("midi-out")
               .long_help("id:name where id is an 8 bit hex number for a MIDI output")
               .multiple(true)
               .takes_value(true)
               .validator(port_validator)
               )
          .get_matches();

    // It's required... why do I have to unwrap it?
    let memory = matches.value_of("memory").unwrap();
    let audio_ins  = matches.values_of("audio input");
    let audio_outs = matches.values_of("audio output");
    let midi_ins  = matches.values_of("MIDI input");
    let midi_outs = matches.values_of("MIDI output");

    // TODO make this configurable, per port.
    let midi_buffer_size = 256 * 2usize.pow(10);
    // Already validated.
    let memory_size = to_hex_u32(memory.to_string()).unwrap() as usize;
    println!("memory size is {}", memory_size);

    let (client, _) = jack::Client::new("algorave", jack::ClientOptions::NO_START_SERVER).unwrap();

    let mut ports = Vec::new();
    // Can't find a way to go from Option(Values ...) to Values ... because
    // I don't see a way to give an empty values. Not at all happy with this
    // command-line argument library, but then I'm a spoilt Haskell programmer
    // after all.
    match audio_ins {
        Some(the_thing_i_want) => for st in the_thing_i_want {
            // It's already validated so we just assume this will work. Must repeat
            // myself in the comments again: we should NOT have to do this twice,
            // but I'm not so familiar with rust so I'm not sure if it's possible
            // to get an optparse-applicative style interface.
            let splits: Vec<&str> = st.split(':').collect();
            let id_string = splits[0];
            let name = splits[1];
            let id = u8::from_str_radix(splits[0], 16).unwrap();
            let port = client.register_port(name, jack::AudioIn).unwrap();
            ports.push(backend::SomePort::AudioIn(id, port));
        }
        None => {}
    }
    match audio_outs {
        Some(the_thing_i_want) => for st in the_thing_i_want {
            let splits: Vec<&str> = st.split(':').collect();
            let id_string = splits[0];
            let name = splits[1];
            let id = u8::from_str_radix(splits[0], 16).unwrap();
            let port = client.register_port(name, jack::AudioOut).unwrap();
            ports.push(backend::SomePort::AudioOut(id, port));
        }
        None => {}
    }
    match midi_ins {
        Some(the_thing_i_want) => for st in the_thing_i_want {
            let splits: Vec<&str> = st.split(':').collect();
            let id_string = splits[0];
            let name = splits[1];
            let id = u8::from_str_radix(splits[0], 16).unwrap();
            let port = client.register_port(name, jack::MidiIn).unwrap();
            ports.push(backend::SomePort::MidiIn(id, midi_buffer_size, port));
        }
        None => {}
    }
    match midi_outs {
        Some(the_thing_i_want) => for st in the_thing_i_want {
            let splits: Vec<&str> = st.split(':').collect();
            let id_string = splits[0];
            let name = splits[1];
            let id = u8::from_str_radix(splits[0], 16).unwrap();
            let port = client.register_port(name, jack::MidiOut).unwrap();
            ports.push(backend::SomePort::MidiOut(id, midi_buffer_size, port));
        }
        None => {}
    }

    match backend::run(memory_size, client, ports) {
        // TODO print it.
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
