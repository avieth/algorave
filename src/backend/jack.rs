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

type Identifier = u32;

/// Completely describes a multimedia program. An ExecutableState may be
/// generated from this.
///
/// The descriptive state allocates and owns all of the buffers used by the
/// processor. They are accessed by way of unsafe rust: giving raw pointers
/// and unsafe dereferencing them within the processor is the only thing I
/// could come up with to make it work. I don't see any way to convince rust
/// of memory safety here.
///
/// The descriptive state never actually reads or writes from these buffers,
/// and it never de-allocates any of them until the processor has indicated, by
/// way of the stage mutex, that it is done with them.
pub struct DescriptiveState {
    ds_jack_inputs: HashMap<Identifier, JackAudioInput>,
    ds_jack_outputs: HashMap<Identifier, JackAudioOutput>,
}

impl DescriptiveState {

    /// This descriptive state holds all of the buffers that will be used by
    /// the executable state. Rust won't like this though. We rely entirely
    /// upon raw pointers which will be unsafely dereferenced by the processor.
    pub fn make_executable(&mut self) -> ExecutableState {
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        for (_name, input) in self.ds_jack_inputs.iter_mut() {
            inputs.push(InputPortAndBuffer {
                ipab_port: &mut input.jai_port as *mut jack::Port<jack::AudioIn>,
                ipab_buffer: &mut input.jai_buffer as *mut Vec<f32>
            });
        }
        for (_name, output) in self.ds_jack_outputs.iter_mut() {
            outputs.push(OutputPortAndBuffer {
                opab_port: &mut output.jao_port as *mut jack::Port<jack::AudioOut>,
                opab_buffer: &mut output.jao_buffer as *mut Vec<f32>
            });
        }
        return ExecutableState {
            es_jack_inputs: inputs,
            es_jack_outputs: outputs,
            es_programs: Vec::new()
        };
    }

}

struct InputPortAndBuffer {
    ipab_port: *mut jack::Port<jack::AudioIn>,
    ipab_buffer: *mut Vec<f32>
}

impl InputPortAndBuffer {
    /// Copy from the JACK audio port to the buffer.
    /// Call from within a process callback using the ProcessScope provided.
    /// The JACK port and the buffer are unsafely dereferenced.
    pub fn copy(&mut self, scope: &jack::ProcessScope) -> () {
        unsafe {
            let vec = &mut (*self.ipab_buffer);
            let port = &mut (*self.ipab_port);
            vec.copy_from_slice(port.as_slice(scope));
        }
    }
}

struct OutputPortAndBuffer {
    opab_port: *mut jack::Port<jack::AudioOut>,
    opab_buffer: *mut Vec<f32>
}

impl OutputPortAndBuffer {
    /// Copy from the buffer to the JACK audio port.
    /// Call from within a process callback using the ProcessScope provided.
    /// The JACK port and the buffer are unsafely dereferenced. They are
    /// assumed to come from a DescriptiveState by way of make_executable,
    /// such that the DescriptiveState is wise enough to keep the port and
    /// buffer around until the process callback has indicated it does not
    /// need them.
    pub fn copy(&mut self, scope: &jack::ProcessScope) -> () {
        unsafe {
            let vec = &mut (*self.opab_buffer);
            let port = &mut (*self.opab_port);
            port.as_mut_slice(scope).copy_from_slice(vec);
        }
    }
}

/// Everything needed by the process callback:
/// - JACK audio inputs and buffers to copy to at the beginning of process.
/// - JACK audio outputs and buffers to copy from at the end of process.
/// - Loopback buffers and their associated real outputs.
/// - For each real output, a program to run for each frame.
/// NB this does _not_ include the current frame, because the descriptive state
/// does not know it.
///
/// How about ownership? The descriptive state owns the JACK ports and
/// the loopback buffers. It must be that way. But then, it needs to give
/// mutable references to the processor... can unsafe rust help? It can't
/// bypass the borrow checker, but surely by deferencing raw pointers we can
/// get what we want? 
///
///   let raw_pointer_mut = &mut the_port as *mut jack::Port
///
/// So, the executable state can be a big mess of raw pointers, derived such
/// that it can do exactly what the process callback must do, efficiently.
/// Everything shall be unsafe pointer dereferencing to get buffers and ports,
/// then memcpy or loopback set.
///
/// We'd want maybe
///
///   Vec<InputPortAndBuffer>
///   Vec<OutputPortAndBuffer>
///
/// We can just run through those and copy one array to another.
/// Then the only remaining thing is the output programs, and the loopbacks.
///
///   Vec<OutputProgramAndBuffer>
///   Vec<Loopback
pub struct ExecutableState {
    es_jack_inputs: Vec<InputPortAndBuffer>,
    es_jack_outputs: Vec<OutputPortAndBuffer>,
    es_programs: Vec<Program>
}

impl ExecutableState {
    /// Given the current frame at the start of processing, and the number of
    /// frames to process, run the program for each output and do the necessary
    /// copying to make JACK go.
    pub fn run(&mut self, current_frame: u64, frames_to_process: jack::Frames, scope: &jack::ProcessScope) -> () {
        for ipab in self.es_jack_inputs.iter_mut() {
            ipab.copy(scope);
        }
        // Now the input buffers have been updated by way of unsafe raw pointer
        // dereferencing. Every program which must be run should now be safe to
        // run. Doing so will further update raw pointers within this
        // ExecutableState, and they will be written out in the next loop over
        // the es_jack_outputs.
        for program in self.es_programs.iter_mut() {
            // What does the program need?
            // Morally it should only get the input buffers.
            // The program can then mutate the relevant output buffer (only one
            // per-program).
            //
            // Obvious idea is to have a stack machine: a list of op-codes,
            // where references are fulfilled by the supplied inputs.
            // But the key point is: we must be able to evaluate it with no
            // allocations. Is that even feasible, given the language we want?
            //
            // Let's start with something simple
            //program.run();
        }
        for opab in self.es_jack_outputs.iter_mut() {
            opab.copy(scope);
        }
    }
}

/// OK to send those raw pointers. The controller implementation guarantees
/// they are always valid, and that there are no races.
unsafe impl Send for ExecutableState {}


struct Stage<T> {
    stage_value: T,
    stage_processed: bool
}

// How to represent a program?
// We'd prefer to not have a recursive datatype.

pub struct Program {}

pub enum Instruction {
    /// Add a loopback called first identifier, referencing second identifier,
    /// using this Loopback buffer.
    AddLoopback(u32, u32, Loopback),
    /// Add an output with a given identifier, set to run a given program at
    /// each frame.
    AddOutput(u32, Program, JackAudioOutput),
    AddInput(u32, JackAudioInput)
}

pub struct Controller {
    cont_client:   jack::AsyncClient<(), Processor>,
    cont_state:    DescriptiveState,
    cont_stage:    Arc<Mutex<Option<Stage<ExecutableState>>>>
}

impl Controller {
    /// Stage the current state by generating the executable state and updating
    /// the mutex-protected location.
    pub fn stage(&mut self) {
        let mut executable_state = self.cont_state.make_executable();
        let mutex = &(*self.cont_stage);
        let mut mstage = mutex.lock().unwrap();
        match &mut (*mstage) {
            // Drop in the new stage and that's all.
            None => {
                let new_stage = Stage {
                    stage_value: executable_state,
                    stage_processed: false
                };
                *mstage = Some(new_stage);
            },
            // In this case we need to put in the new stage but must also
            // get the old stage and de-allocate it.
            // IIUC this should work: we swap the references, so that when
            // executable_state goes out of scope, it actually de-allocates
            // what was formerly stage_value.
            Some(stage) => {
                std::mem::swap(&mut stage.stage_value, &mut executable_state);
                stage.stage_processed = false;
            }
        }
    }

    /// The string name is required because it will show up on JACK, but the
    /// u32 identifier is used internally.
    pub fn add_output(&mut self, name: &str, id: u32) -> () {
        let port = self.cont_client.as_client().register_port(name, jack::AudioOut).unwrap();
        // NB: vec! acutally fills the vector with zeros, which is key since we
        // unsafely reference it in the processor and copy shit to it. If we
        // used Vec::with_capacity it would not work.
        let vbuffer = vec![0.0; self.cont_client.as_client().buffer_size() as usize];
        let output = JackAudioOutput {
            jao_buffer: vbuffer,
            jao_port: port
        };
        // TODO TBD what if it overwrites?
        // Could cause a segfault I think, in case the process callback is
        // still referencing it, because the HasHMap will delete the thing?
        self.cont_state.ds_jack_outputs.insert(id, output);
    }
    pub fn add_input(&mut self, name: &str, id: u32) -> () {
        let port = self.cont_client.as_client().register_port(name, jack::AudioIn).unwrap();
        let vbuffer = vec![0.0; self.cont_client.as_client().buffer_size() as usize];
        let input = JackAudioInput {
            jai_buffer: vbuffer,
            jai_port:port
        };
        // TODO TBD what if it overwrites?
        // Could cause a segfault I think, in case the process callback is
        // still referencing it, because the HasHMap will delete the thing?
        self.cont_state.ds_jack_inputs.insert(id, input);
    }
    pub fn add_loopback(&self, name: u32, output: u32, size: usize) -> () {
        /*let lo = Loopback::new(size, 0.0);
        // Inform the process callback that this loopback should be
        // associated with the output identified by the string `output` name.
        let mutex = &(*self.cont_stage);
        let mut stage = mutex.lock().unwrap();
        stage.instructions.push(Instruction::AddLoopback(name, output, lo));
        // This should de-allocate all of the garbage.
        stage.garbage.clear();*/
    }
}

/// Synchronization mechanism between processor and controller
/// Will be found inside an Arc, of which the processor and controller threads
/// each have a clone.
/// The controller thread will wait on the condition variable immediately after
/// it passes a new executable state to the processor through the mutex.
/// Only after the processor signals it will the controller wake up and perform
/// any necessary deallocation.
pub struct Synchro<T> {
    synchro_mutex: Mutex<T>,
    synchro_cond: Condvar
}

/// Shared state between processor and controller. Will be held inside a mutex
/// by way of Synchro<SharedState>.
///   data SharedState t = NoChange | Staged change | Processed change
pub enum Stage<T> {
    NoChange,
    Staged(t),
    Processed(t)
}

type SharedState = Arc<Synchro<Stage<ExecutableState>>>

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
    // TODO subsumes the above 3 fields.
    proc_state:    ExecutableState,
    // Changes to the system come in through here.
    proc_stage:    Arc<Mutex<Option<Stage<ExecutableState>>>>
}

// TODO FIXME
// The raw pointer dereferencing scheme is actually no good: the controller
// thread may de-allocate the buffers while the process callback is going.
// Somehow the Stage value needs to indicate a shrinking of the descriptive
// state... 
// Indeed, the descriptive state will be an algebraic thing, probably a monoid.
// The programmer changes it by giving "commands". Yeah it's actually these
// commands which give a monoid action on the descriptive state.
//
//   cmd1 <> cmd2 |> ds = (cmd1 |> ds) <> (cmd2 |> ds)
//
// assuming both things "happen" instantaneously (if they did not, we could
// hear the intermediate result from the running system).
//
// Then, every cmd1 factors into an additive and subtractive part.
//
// NB we could probably get away with using a condition variable so that the
// control thread can sync with the processor thread and know when it has
// digested the new state and can retrieve the subtractive part.
//
//   Processor:
//     Try to lock the mutex (non blocking)
//       If not locked no problem
//         continue with current state
//       If locked and no staged change
//         continue with current state
//       If locked and staged change:
//         take the new processor state
//         put the old processor state in the shared cell
//         set to no staged change
//         signal a condition variable
//
//    Controller:
//      Lock the mutex (blocking)
//        If staged change (very rare case):
//          release that staged change (deallocate)
//        Put in the staged change
//        Wait on the condition variable (block)
//        Assert that there is now no staged change (processor took it)
//        Assert that "garbage" has appeared from the processor thread, and
//          then deallocate it
//
// That should work, with no blocking ever in the processor.
//
// So what we'll do is use the descriptive state and the update command to
// figure out
//   1. What must be allocated
//   2. What can be deallocated
// This can be done with or without holding the lock, doesn't matter, as the
// processor will never block.
// Then we allocate what must be allocated, remove but do not deallocate what
// must deallocated from the descriptive state, generate the executable state
// (a bunch of raw pointers), pass it to the processor, and block on the
// condition variable.
// When we get back, deallocate the previous executable state and also those
// buffers that were removed. Voila, all done.
//

impl Processor {
    /// Run a processor using a jack client. This will activate the client.
    /// You get a Controller in return, which can be used to make the JACK
    /// client sing.
    pub fn run(client: jack::Client) -> Controller {
        let mut descriptive_state = DescriptiveState {
            ds_jack_inputs: HashMap::new(),
            ds_jack_outputs: HashMap::new()
        };
        let executable_state = descriptive_state.make_executable();
        let stage = None;
        let arc_mutex = Arc::new(Mutex::new(stage));
        let processor = Processor {
            current_frame: 0,
            sample_rate: client.sample_rate(),
            buffer_size: client.buffer_size(),
            loopbacks: HashMap::new(),
            jack_inputs: HashMap::new(),
            jack_outputs: HashMap::new(),
            proc_state: executable_state,
            proc_stage: arc_mutex.clone()
        };
        let async_client = client.activate_async((), processor).unwrap();
        let controller = Controller {
            cont_client: async_client,
            cont_state: descriptive_state,
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
        // First, check whether there is a new executable state that we should
        // use.
        {
            let mutex = &(*self.proc_stage);
            let lock = mutex.try_lock();
            if let Ok(mut mguard) = lock {
                let mstage = &mut (*mguard);
                match mstage {
                    // Nothing has ever been staged.
                    None => {},
                    Some(stage) => {
                        if stage.stage_processed {
                            // We put this here, but the controller thread has not
                            // yet cleaned it up. No problem. This is the typical case
                            // of "nothing to do" and will be hit on almost every
                            // process callback.
                        } else {
                            // The controller thread has staged a new executable state.
                            // Take it and pass the current one back.
                            // The next time the controller thread stages a new
                            // executable state, it will flip the processed
                            // bool back to false and we'll go here again.
                            std::mem::swap(&mut stage.stage_value, &mut self.proc_state);
                            stage.stage_processed = true;
                        }
                    }
                }
            }
        }

        // The ExecutableState at self.proc_state tracks all state _except_ for
        // the current frame, because the controller thread does not have access
        // to that.
        // TODO TBD alternatively, we could deal with the current frame by way
        // of raw pointers as well: the descriptive state in the controller
        // would have a u64 and would pass a raw pointer to it.
        // Would be more consistent, but would also be adding unsafe calls for
        // no other benefit, so bad idea.
        let frames_to_process = scope.n_frames();
        self.proc_state.run(self.current_frame, frames_to_process, scope);
        // Cast is fine, frames_to_process is u32
        self.current_frame += frames_to_process as u64;

        return jack::Control::Continue;
    }
}
