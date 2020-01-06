use std::collections::HashMap;
use std::sync::{Mutex, Arc};
use jack;

/// A buffer and a write index. Reads are taken from the write index plus 1
/// modulo the buffer size.
pub struct Loopback {
    // Can't figure out how to construct a slice from a non-statically-known
    // size (should be easy/obvious...) so whatever, use Vec instead.
    lo_buffer: Vec<f32>,
    // lo_buffer.len()
    // Using u64 because lo_write and lo_read take a u64 index and we'll want
    // to take it modulo this size.
    lo_size: u64
}

impl Loopback {

    /// Create a loopback of a given size (in frames). This determines how
    /// much of a delay the loopback induces: this many frames will pass after
    /// writing a value before it is read back.
    ///
    /// Do not give a size larger than 2^32 or they may be some overflow on
    /// the index computation (in case usize is 32 bits). But you wouldn't
    /// ever do that anyway would you? Besides the stupid memory requirements,
    /// that would induce a 12 hour signal delay at 96KHz sampling rate.
    pub fn new(size: usize, fill: f32) -> Loopback {
        let msize = std::cmp::max(size, 2);
        let lo = Loopback {
            lo_buffer: vec![fill; msize],
            lo_size: msize as u64 // definitely fits.
        };
        return lo;
    }

    /// We write to the frame modulo the length of the buffer.
    pub fn lo_write(&mut self, frame : u64, x : f32) -> () {
        let idx = (frame % self.lo_size) as usize;
        self.lo_buffer[idx] = x;
    }

    /// We read from the frame modulo the length of the buffer, plus 1.
    /// This will have been written `lo_size - 1` frames ago.
    pub fn lo_read(&self, frame : u64) -> f32 {
        let idx = ((frame + 1) % self.lo_size) as usize;
        return self.lo_buffer[idx];
    }

}

pub struct JackAudioInput {
    jai_buffer: Vec<f32>,
    /// The port from which the input is sourced. Will be copied to the
    /// jai_buffer at each process callback.
    jai_port: jack::Port<jack::AudioIn>
}

pub struct JackAudioOutput {
    jao_buffer: Vec<f32>,
    /// The port to which we'll copy data, from the jao_buffer.
    jao_port: jack::Port<jack::AudioOut>
}

// What will a program look like?
// Should be inspired by FRP. It's a set of non-recursive bindings, each of
// which defines a time-varying signal. Primitives like numbers or bools are
// constant signals. Non-constant signals include
// - now, giving the current frame
// - JACK audio and MIDI inputs
// Output signals cannot be used, but loopbacks can be declared, effectively
// creating an input from an output that is n frames behind the output.
//
// What changes can be made to the system? You can
// - Add inputs and outputs. This is all done in the control thread, not the
//   process callback.
//   It should also make sense to remove inputs and outputs. Any references
//   to inputs in the program can be defaulted to constant 0, and a set
//   reference to a missing output can become a no-op.
// - Declare loopbacks, bringing a new input name into scope, but not a new
//   JACK port.
//
//   What is the meaning of this though?
//
//     declare output output1
//     let input1  = loopback 0 output1
//     set output1 = input1
//
//   We declare an output, then we assign it to itself at the prior frame.
//   This _ought_ to be disallowed by no-recursive-bindings. It seems like it's
//   not though, because of the initial 'declare output output1'. We want a
//   pure functional language with equational reasoning, so we should have that
//   this program is the same as
//
//     set output1 = loopback 0 output1
//
//    which is nonsense.
//
//
// What does execution look like? Surely what we want is something like this:
//
// 1. memcpy each of the JACK inputs to buffers owned by the process callback
//    state.
// 2. For each frame, for each output, evaluate the output and write it to
//    buffers owned by the process callback state.
//    This deals with loopback state.
// 3. For each JACK output, memcpy the corresponding buffer.
//
// So the program state includes:
// - for each input, a buffer, keyed by name.
// - for each output, a buffer and a program.
// The program can reference input buffers by name, which we look up in the
// hash map.

pub struct Program {
}

pub enum Instruction {
    /// Add a loopback called first identifier, referencing second identifier,
    /// using this Loopback buffer.
    AddLoopback(u32, u32, Loopback),
    /// Add an output with a given identifier, set to run a given program at
    /// each frame.
    AddOutput(u32, Program, JackAudioOutput),
    AddInput(u32, JackAudioInput)
}

pub enum Garbage {}

pub struct Stage {
    instructions: Vec<Instruction>,
    garbage: Vec<Garbage>
}

impl Stage {
    pub fn new() -> Stage {
        return Stage {
            instructions: Vec::new(),
            garbage: Vec::new()
        }
    }
}

pub struct Controller {
    cont_client:   jack::AsyncClient<(), Processor>,
    cont_stage:    Arc<Mutex<Stage>>
}

impl Controller {
    /// The string name is required because it will show up on JACK, but the
    /// u32 identifier is used internally.
    pub fn new_output(&self, name: &str, id: u32) -> () {
        // Do all necessary allocation here before passing instructions to the
        // process callback by way of the shared Stage value.
        let port = self.cont_client.as_client().register_port(name, jack::AudioOut).unwrap();
        let vbuffer = Vec::with_capacity(self.cont_client.as_client().buffer_size() as usize);
        let output = JackAudioOutput {
            jao_buffer: vbuffer,
            jao_port: port
        };
        let mutex = &(*self.cont_stage);
        let mut stage = mutex.lock().unwrap();
        stage.instructions.push(Instruction::AddOutput(id, Program {}, output));
        stage.garbage.clear();
    }
    pub fn new_input(&self, name: &str, id: u32) -> () {
        let port = self.cont_client.as_client().register_port(name, jack::AudioIn).unwrap();
        let vbuffer = Vec::with_capacity(self.cont_client.as_client().buffer_size() as usize);
        let input = JackAudioInput {
            jai_buffer: vbuffer,
            jai_port:port
        };
        let mutex = &(*self.cont_stage);
        let mut stage = mutex.lock().unwrap();
        stage.instructions.push(Instruction::AddInput(id, input));
        stage.garbage.clear();
    }
    pub fn new_loopback(&self, name: u32, output: u32, size: usize) -> () {
        let lo = Loopback::new(size, 0.0);
        // Inform the process callback that this loopback should be
        // associated with the output identified by the string `output` name.
        let mutex = &(*self.cont_stage);
        let mut stage = mutex.lock().unwrap();
        stage.instructions.push(Instruction::AddLoopback(name, output, lo));
        // This should de-allocate all of the garbage.
        stage.garbage.clear();
    }
}

pub struct Processor {
    // jack::Frames is 32 bits.
    // If we're keeping a counter of how many frames have passed since the
    // beginning of time, then we'll need something bigger.
    // If we use 64 bits, we can go for over 2 trillion days at 96KHz.
    // If we use 32 bits, we can only go for half a day!
    // So, just use u64... but that complicates the implementation of buffer
    // indexing for 32 bit machines.
    current_frame: u64,
    // Is usize just because the jack library gives usize from the
    // Client.sample_rate function.
    sample_rate:   usize,
    buffer_size:   jack::Frames,
    // For each named loopback, its buffer data and the name of the proper
    // output which it samples.
    loopbacks:     HashMap<u32, (u32, Loopback)>,
    jack_inputs:   HashMap<u32, JackAudioInput>,
    jack_outputs:  HashMap<u32, JackAudioOutput>,
    // Changes to the system come in through here.
    proc_stage:    Arc<Mutex<Stage>>
}

impl Processor {
    /// Run a processor using a jack client. This will activate the client.
    /// You get a Controller in return, which can be used to make the JACK
    /// client sing.
    pub fn run(client: jack::Client) -> Controller {
        let arc_mutex = Arc::new(Mutex::new(Stage::new()));
        let processor = Processor {
            current_frame: 0,
            sample_rate: client.sample_rate(),
            buffer_size: client.buffer_size(),
            loopbacks: HashMap::new(),
            jack_inputs: HashMap::new(),
            jack_outputs: HashMap::new(),
            proc_stage: arc_mutex.clone()
        };
        let async_client = client.activate_async((), processor).unwrap();
        let controller = Controller {
            cont_client: async_client,
            cont_stage: arc_mutex
        };
        return controller;
    }
}

impl jack::ProcessHandler for Processor {
    /// The JACK process callback.
    /// It runs the multimedia program for the number of frames given by the
    /// scope.
    /// It will also try, without blocking, to acquire changes to the program
    /// from the stage mutex.
    fn process(&mut self, _client: &jack::Client, scope: &jack::ProcessScope) -> jack::Control {
        // Must "dereference" using *, to get out of the Arc, but also take a
        // reference using &, so that we don't try to claim ownership.
        {
            let mutex = &(*self.proc_stage);
            let mut lock = mutex.try_lock();
            // rust has some silly magic that makes the try_lock method work on
            // the Arc, so could have done:
            //let mut _lock = self.proc_stage.try_lock();
            if let Ok(ref mut _staged) = lock {
                // Digest updates to the system by mutating self.
                // Possible changes?
                // - Add real JACK input or output
                // - Set program at output
                //
                // TODO TBD we probably want to limit how much work is done to
                // update state here, because it cuts into actual processing
                // time and could induce xruns.
                // Hm.....
                // Ideally there would be literally only 2 things to do: bring
                // in the new state and push out the garbage state.
                //
                // NB: in fact it MUST be like this; we can't do a bunch of
                // HashMap.add calls because that will allocate.
                //
                // If we didnt' have loopbacks, we could just swap in the
                // entire state value because the whole thing has no memory:
                // it only depends upon JACK inputs and current frame.
                // So that's the sensible thing to do: the controller has its
                // own state bundle, the processor has one as well. Controller
                // does updates in staging and then swaps it in, gets the other
                // one and operates on that next.
                //
                // Things are complicated by loopbacks, because the controller
                // thread can't possibly have that data in a recent form. A
                // possible solution: share the loopback buffers and pass
                // pointers around. Rust makes that kinda painful
                // But still that seems like the right way... the controller can
                // put the new state in, and the processor will grab it later
                // and put its old state into the "garbage" spot.
                //
                //   data Stage state = Staged state | Processed state
                //
                // The controller takes the lock and sets it to staged, either
                // from Processed or Staged, although getting a Staged will be
                // unlikely.
                // The process thread checks the lock and if it gets it, ignores
                // Processed, but takes a Staged and sets it to Processed with
                // its old state.
                // Yes, good.
                // But there's a problem: the controller doesn't have the same
                // state description that the processor has.
                // So how about the controller maintains the description of
                // what the state should be, including ownership of all JACK
                // ports and loopback buffers. This is what's modified by the
                // programmer. Then from this an executable state can be
                // generated. Throw that into the shared state, and deallocate
                // the processed state.
                // Will have to use unsafe rust but whatever.... yes, because
                // rust does not know that the controller thread outlives the
                // process thread.  It's also not comfortable with having a
                // mut reference to the shared buffers.
                //
            }
            // If we got the lock, it will be unlocked here where lock goes
            // out of scope.
        }

        let frames_to_process : jack::Frames = scope.n_frames();

        for (_input_name, jai) in self.jack_inputs.iter_mut() {
            jai.jai_buffer.copy_from_slice(jai.jai_port.as_slice(scope));
            // TODO also copy in MIDI events.
        }

        for _frame in 0..frames_to_process {
            for (_output_name, _output) in self.jack_outputs.iter_mut() {
                // Run the program for this output.
                // It may reference current frame, inputs, and do math and
                // stuff.
            }
            for (_loopback_name, (_output_id, loopback)) in self.loopbacks.iter_mut() {
                // Sample the output source for this loopback and write it to
                // the loopback buffer at the current frame.
                loopback.lo_write(self.current_frame, 42.0);
            }
            self.current_frame += 1;
        }

        // For every audio output buffer, do a memcpy.
        // Their jao_buffer fields will have been updated by the above for
        // loop.
        for (_output_name, jao) in self.jack_outputs.iter_mut() {
            jao.jao_port.as_mut_slice(scope).copy_from_slice(&jao.jao_buffer);
            // TODO also copy out MIDI events.
        }

        return jack::Control::Continue;
    }
}

/*
fn main() {
    let client_result = jack::Client::new("algorave", jack::ClientOptions::NO_START_SERVER);
    let (client, _status) = client_result.unwrap();

    // TODO port addition should be dynamic, controlled by the programming
    // language.
    //let port_left = async_client.as_client().register_port("left", jack::AudioOut);
    //let port_right = async_client.as_client().register_port("right", jack::AudioOut);
    let port_left = client.register_port("left", jack::AudioOut).unwrap();
    //let _port_right = client.register_port("right", jack::AudioOut).unwrap();
    let mut oports = HashMap::new();
    let mut iports = HashMap::new();
    oports.insert("left", Output::JackAudioOutput(port_left));
    // ports.insert("right", Output::JackAudioOutput(port_right));
}
*/
