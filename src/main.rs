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

// Goal for today: we want to be able to write a text file, dump it to the
// encoding that rust will understand (this can be done in Haskell) and then
// send it along so that rust will load it up.
//
// Also: be able to set memory using a hex address and hex or base64 data, or
// even a file (also a Haskell program)
//
// Barring any mistakes present, the rust program is all set to do this.
// The outstanding work is:
// - How to properly pipe data to the rust program
//   Using a named pipe is probably best. nc is also another good source.
//
//     algorave ... < my_fifo
//     nc -l ip_address port | algorave ...
//
//   In both cases, it's essential that algorave does not observe an EOF on
//   stdin. For the named pipe, EOF will happen when all writers close...
//   What it all comes down to really is editor integration. Ideally it would
//   be possible to write directly from vim. Whether the source is a named
//   pipe or a socket/netcat, we have the same issue: vim just writes out a
//   file. Evidently we need some other program which will, when prompted, read
//   a file and pipe it to stdout.
//   Done. intercat and infinicat.
//
// - A Haskell program that translates human-readable text to the rust
//   encoding.
//   This program too should read from stdin and write to stdout. What goes to
//   stdout will be an encoding of the rust Update type.
//   The Haskell executable will parse a higher-level abstract syntax from a
//   hopefully fairly nice concrete syntax. We can start with an assembly-like
//   language. We can even have syntactic sugar to do, for instance, if/else.
//   CRUCIALLY we will have not only programs but _commands_, i.e. set this
//   program or set this memory.
//   We do require a transactional semantics, so we could have a bracketing
//   syntax.
//   The pattern is STAGE/COMMIT. You can STAGE changes to the program or to
//   the memory, and you can also UNSTAGE them. But how do you identify the
//   change to UNSTAGE? I think instead we should just have COMMIT/RESET
//   semantics just like we have for intercat.
//
//   There is an analogy to keep in mind. This is a complete compiler chain
//   (concrete syntax through to running executable) but it doesn't just run the
//   chain and then quit. Instead, the programmer can instruct it to run any
//   step in the chain, by issuing instructions on stdin.
//   - parse some concrete syntax
//   - typecheck the whole program
//   - optimize the whole program
//   - reset the staged changes
//   - commit the staged changes, by writing out an encoded update that the
//     rust program understands (analogy: code generation and linking).
//     This includes loading any blob assets requested.
//   - ^ In fact, that should be a "generate code (update)" step, and a separate
//     "commit update" step that just writes it out. This way the performer can
//     get the lowest latency update possible (although it still would not be
//     reliably fast).
//
// CONCRETE TODO STEPS for this weekend?
// 1. Make an assembly language featuring
//    - nice/easy mnemonics and address formats
//    - labels for branching
//    - syntactic sugar for using literals? Not essential, and not trivial since
//      it requires choosing a memory location
// 2. Wire up a parser for that which reads from stdin and then produces
//    "machine code" for the rust executable on stdout.
//    Must think about the future: we'll also want to be able to tell it to load
//    up WAV files, so the concrete syntax must be suitable for that.
//    How about this?
//
//      TEXT;<0 or more assembly instructions ; separated>;END;
//      DATA;<0 or more data items ; separated>;END;
//
//    Reserved words are TEXT, DATA, END (cannot be used a labels in the
//    assembly).
//    In the data section, you can give explicit data in hex or base64, but also
//    filenames
//
//      DATA;ADDRESS 0xFF;HEX 0xabcdef;B64 91792873813hab++==;BLOB ./foo.bin;END;
//
//    semantics? Give an address, then 1 or more sources. The bytes will be
//    appended in that order and written at that address.
//    Special source: a WAV file, which is decoded, resampled if necessary.

/// Tries to decode a complete Update from stdin.
/// Retries if it fails to decode, and gives None when/if stdin is EOF.
/// In case of IO errors, 
fn get_update_from_stdin() -> Result<Update, StdioSourceError> {
    /*
    let p = Some(examples::program_3());
    let mut string = String::new();
    std::io::stdin().read_line(&mut string).unwrap();
    return Ok((Update { program: p, memory: Vec::new() }));
    */

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
