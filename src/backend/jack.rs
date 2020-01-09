use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex};
use jack;

use crate::backend::util;
use crate::lang::program::{Program, Term};

/*
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
*/

/// A JACK audio input port along with a buffer of the same size as the
/// audio input buffer (JACK server buffer size).
///
/// The buffer is only ever read from by way of unsafe raw pointers found
/// in `InputPortAndBuffer`, from a JACK process callback. That callback will
/// also memcpy to the buffer from JACK port buffer, also by way of an unsafe
/// raw pointer.
pub struct JackAudioInput {
    jai_buffer: Vec<f32>,
    /// The port from which the input is sourced. Will be copied to the
    /// jai_buffer at each process callback.
    jai_port: jack::Port<jack::AudioIn>
}

/// A JACK audio output port along with a buffer of the same size as the
/// audio output buffer (JACK server buffer size).
///
/// The buffer is only ever written to by way of unsafe raw pointers found
/// in `OutputPortAndBuffer`, from a JACK process callback. That callback will
/// also memcpy the buffer to the JACK port buffer, also by way of an unsafe
/// raw pointer.
pub struct JackAudioOutput {
    jao_buffer: Vec<f32>,
    /// The port to which we'll copy data, from the jao_buffer.
    jao_port: jack::Port<jack::AudioOut>
}

/// Type of identifier used in the program description and executable state.
/// It identifies JACK inputs and outputs. In the future it will also identify
/// loopbacks.
type Identifier = u32;

/// Type of numbers used in the program definition. Should be fine, and is
/// convenient because JACK uses it for everything, but maybe we'll want to
/// use fixed point instead.
type Number = f32;

/// Completely describes a multimedia program. An `ExecutableState` may be
/// generated from this.
///
/// The `DescriptiveState` owns all of the buffers used by the processor.
/// They are accessed by way of unsafe rust: giving raw pointers and unsafe
/// dereferencing them within the process callback is the only thing I
/// could come up with to make it work. I don't see any way to convince rust
/// of memory safety here while simultaneously ensuring the process callback
/// never allocates, deallocates, or blocks.
///
/// The owner of the descriptive state (the controller) never actually reads or
/// writes from these buffers, and it never deallocates any of them until the
/// processor has indicated, by way of mutex and condition variable, that it is
/// done with them.
pub struct DescriptiveState {
    ds_jack_inputs: HashMap<Identifier, JackAudioInput>,
    ds_jack_outputs: HashMap<Identifier, JackAudioOutput>,
    ds_programs: HashMap<Identifier, Program<Identifier, Number>>
}

impl DescriptiveState {

    /// Create a whole new `ExecutableState` which reflects this
    /// `DescriptiveState`.
    ///
    /// The programs are cloned.
    ///
    /// The buffers in the inputs and ouputs are not cloned; instead, raw
    /// pointers to them are put into the `ExecutableState`. Thus we must be
    /// very careful to only deallocate parts of this `DescriptiveState` after
    /// we're sure no `ExecutableState` is referencing them.
    pub fn make_executable(&mut self) -> ExecutableState {
        let mut inputs  = HashMap::new();
        let mut outputs = HashMap::new();
        let mut executables = HashMap::new();
        for (identifier, input) in self.ds_jack_inputs.iter_mut() {
            inputs.insert(*identifier, InputPortAndBuffer {
                ipab_port: &mut input.jai_port as *mut jack::Port<jack::AudioIn>,
                ipab_buffer: &mut input.jai_buffer as *mut Vec<f32>
            });
        }
        for (identifier, output) in self.ds_jack_outputs.iter_mut() {
            outputs.insert(*identifier, OutputPortAndBuffer {
                opab_port: &mut output.jao_port as *mut jack::Port<jack::AudioOut>,
                opab_buffer: &mut output.jao_buffer as *mut Vec<f32>
            });
        }
        for (identifier, program) in self.ds_programs.iter() {
            executables.insert(*identifier, Executable {
                exec_program: (*program).clone(),
                // The required space for a program to run can't be more than
                // the number of instructions, because each instruction grows
                // the stack by at most 1.
                exec_stack: vec![0.0; program.length()]
            });
        }
        return ExecutableState {
            es_jack_inputs: inputs,
            es_jack_outputs: outputs,
            es_executables: executables
        };
    }

}

/// Part of the executable state. This holds raw pointers, one to a JACK input
/// port and the other to a buffer into which that port's data can be copied
/// at each process callback.
pub struct InputPortAndBuffer {
    ipab_port: *mut jack::Port<jack::AudioIn>,
    ipab_buffer: *mut Vec<f32>
}

unsafe impl Sync for InputPortAndBuffer {}

impl InputPortAndBuffer {

    /// Get the value of the _buffer_ at a given offset. Uses an unsafe raw
    /// pointer dereference.
    ///
    /// This will be affected by prior calls to copy.
    ///
    /// TBD does rust still do bounds checking? I hope not.
    pub fn get(&self, offset: usize) -> f32 {
        unsafe {
            return (&mut (*self.ipab_buffer))[offset];
        }
    }

    /// Copy from the JACK audio port to the buffer. Uses an unsafe raw
    /// pointer dereferences.
    ///
    /// Call from within a process callback using the ProcessScope provided.
    pub fn copy(&mut self, scope: &jack::ProcessScope) -> () {
        unsafe {
            let vec = &mut (*self.ipab_buffer);
            let port = &mut (*self.ipab_port);
            vec.copy_from_slice(port.as_slice(scope));
        }
    }
}

/// Part of the executable state. This holds raw pointers, one to a JACK output
/// port and the other to a buffer from which that port can be copied to at
/// each process callback.
pub struct OutputPortAndBuffer {
    opab_port: *mut jack::Port<jack::AudioOut>,
    opab_buffer: *mut Vec<f32>
}

unsafe impl Sync for OutputPortAndBuffer {}

impl OutputPortAndBuffer {

    /// Set the value of the _buffer_ at a given offset. Uses an unsafe raw
    /// pointer dereference.
    ///
    /// This should be commited to the port by way of copy later.
    ///
    /// TBD does rust still do bounds checking? I hope not.
    pub fn set(&mut self, offset: usize, value: f32) -> () {
        unsafe {
            (&mut (*self.opab_buffer))[offset] = value;
        }
    }

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

/// Derived from a program.
pub struct Executable {
    exec_program: Program<Identifier, Number>,
    exec_stack: Vec<Number>
}

const TWO_PI: f32 = 2.0 * std::f32::consts::PI;

impl Executable {
    /// Run the executable producing a value.
    /// Assumes the program actually makes sense. Will do no error handling.
    /// Needs the current frame, sample rate, and input buffers.
    pub fn run(&mut self,
               current_frame: u64,
               sample_rate: u32,
               inputs: &HashMap<Identifier, InputPortAndBuffer>,
               frame_offset: u32,
               // TODO figure out whether a lookup table is actually useful.
               _sine_lookup_table: &util::sine::LookupTable) -> f32 {
        for term in self.exec_program.terms().iter() {
            match term {
                Term::Constant(x) => { self.exec_stack.push(*x) },
                Term::Sine() => {
                    match self.exec_stack.pop() {
                        Some(phase) => match self.exec_stack.pop() {
                            Some(freq) => {
                                /* let seconds    = current_frame / (sample_rate as u64);
                                // The sample rate is not going to be very big
                                // (at most 192000 in all practicality) so it
                                // should be safe to cast the remainder to u32.
                                let subseconds = (current_frame % (sample_rate as u64)) as u32;
                                // This will be problematic...
                                let t          = seconds as f32
                                               + (subseconds as f32 / sample_rate as f32);
                                self.exec_stack.push((TWO_PI * freq * (t + phase)).sin()); */
                                let s = (current_frame % (sample_rate as u64)) as f32
                                      / sample_rate as f32;
                                let t = freq * (s + phase);
                                self.exec_stack.push((TWO_PI * t).sin());
                            },
                            _ => panic!("sine: no frequency given")
                        },
                        _ => panic!("sine: no phase given")
                    }
                },
                Term::Sawtooth() => {
                    match self.exec_stack.pop() {
                        Some(phase) => match self.exec_stack.pop() {
                            Some(freq) => {
                                // Since sample_rate is never going to be too
                                // big (probably at most 192000) the cast is
                                // OK.
                                let s = (current_frame % (sample_rate as u64)) as f32
                                      / sample_rate as f32;
                                // s will go from 0.0 to 1.0 every second.
                                // To match canonical sawtooth, we flip it,
                                // scale it, and translate it.
                                let t = freq * (s + phase);
                                self.exec_stack.push(-2.0 * (t.fract() - 0.5));
                            },
                            _ => panic!("sine: no frequency given")
                        },
                        _ => panic!("sine: no phase given")
                    }
                },
                Term::Square() => {
                    match self.exec_stack.pop() {
                        Some(phase) => match self.exec_stack.pop() {
                            Some(freq) => {
                                let s = (current_frame % (sample_rate as u64)) as f32
                                      / sample_rate as f32;
                                let t = freq * (s + phase);
                                // If t is in [0.0,0.5) then give 1.0
                                // If t is in [0.5,1.0] then give -1.0
                                // Alternatively:
                                //   t in [0.0,0.5) -> 0.0
                                //   t in [0.5,1.0] -> 1.0
                                // then subtract 0.5, negate, and multiply by
                                // 2.0
                                self.exec_stack.push(-2.0 * (t.fract().round() - 0.5));
                            },
                            _ => panic!("square: no frequency given")
                        },
                        _ => panic!("square: no phase given")
                    }
                },
                Term::Triangle() => {
                    match self.exec_stack.pop() {
                        Some(phase) => match self.exec_stack.pop() {
                            Some(freq) => {
                                let s = (current_frame % (sample_rate as u64)) as f32
                                      / sample_rate as f32;
                                let t = freq * (s + phase);
                                // Our triangle wave will be
                                //   0.0 -> -1.0
                                //   0.5 ->  1.0
                                //   1.0 -> -1.0
                                // We'll use the absolute value of t - 0.5.
                                // To match the canonical triangle waveform, we
                                // must negate it, move it up by 1/4, then
                                // scale it by 4.
                                self.exec_stack.push(-4.0 * (t.fract() - 0.5).abs() + 1.0);
                            },
                            _ => panic!("triangle: no frequency given")
                        },
                        _ => panic!("triangle: no phase given")
                    }
                },
                Term::Input(identifier) => {
                    match inputs.get(identifier) {
                        Some(ipab) => {
                            self.exec_stack.push(
                                ipab.get(frame_offset as usize)
                            );
                        },
                        None => panic!("Undefined input reference")
                    }
                },
                Term::Add() => {
                    match self.exec_stack.pop() {
                        Some(v1) => match self.exec_stack.pop() {
                            Some(v2) => self.exec_stack.push(v2 + v1),
                            _ => panic!("Add")
                        },
                        _ => panic!("Add")
                    }
                },
                Term::Subtract() => {
                    match self.exec_stack.pop() {
                        Some(v1) => match self.exec_stack.pop() {
                            Some(v2) => self.exec_stack.push(v2 - v1),
                            _ => panic!("Subtract")
                        },
                        _ => panic!("Subtract")
                    }
                },
                Term::Multiply() => {
                    match self.exec_stack.pop() {
                        Some(v1) => match self.exec_stack.pop() {
                            Some(v2) => self.exec_stack.push(v2 * v1),
                            _ => panic!("Multiply")
                        },
                        _ => panic!("Multiply")
                    }
                },
                Term::Divide() => {
                    match self.exec_stack.pop() {
                        Some(v1) => match self.exec_stack.pop() {
                            Some(v2) => self.exec_stack.push(v2 / v1),
                            _ => panic!("Divide")
                        },
                        _ => panic!("Divide")
                    }
                },
                Term::Mod() => {
                    match self.exec_stack.pop() {
                        Some(v1) => match self.exec_stack.pop() {
                            Some(v2) => self.exec_stack.push(v2 % v1),
                            _ => panic!("Mod")
                        },
                        _ => panic!("Mod")
                    }
                }
            }
        }
        match self.exec_stack.pop() {
            Some(val) => return val,
            None => return 0.0
        }
    }
}

/// Everything needed by the process callback:
///
/// - JACK audio inputs and buffers to copy to at the beginning of process.
/// - JACK audio outputs and buffers to copy from at the end of process.
/// - For each real output, a program to run for each frame.
/// - TODO Loopback buffers and their associated real outputs.
///
/// NB this does _not_ include the current frame, because the descriptive state
/// does not know it, and we need an `ExecutableState` to be derivable from
/// only a `DescriptiveState`.
pub struct ExecutableState {
    /// Inputs to read at the beginning of the process callback.
    //es_jack_inputs: Vec<InputPortAndBuffer>,
    es_jack_inputs: HashMap<Identifier, InputPortAndBuffer>,
    /// Outputs to write at the end of the process callback.
    es_jack_outputs: HashMap<Identifier, OutputPortAndBuffer>,
    /// What to do in-between, reading from the buffers in es_jack_inputs, and
    /// writing to the buffers in es_jack_outputs.
    es_executables: HashMap<Identifier, Executable>
}

/// It's OK to send the raw pointers found in an ExecutableState.
/// The controller implementation must guarantee that they are always valid,
/// and that there are no races.
unsafe impl Send for ExecutableState {}

impl ExecutableState {

    /// Given the current frame at the start of processing, and the number of
    /// frames to process, run the program for each output and do the necessary
    /// copying to make JACK go.
    ///
    /// In addition to the current frame, we take the number of frames to
    /// process. This must not be bigger than the buffer sizes of the jack
    /// inputs and outputs. It should in fact be exactly the same size: the
    /// buffer size given by the JACK client.
    ///
    /// The `ProcessScope` is needed in order to deal with the input and output
    /// ports.
    ///
    /// A sine wave lookup table is used in the hope that this is faster than
    /// calling sin() all the time.
    pub fn run(&mut self,
               current_frame: u64,
               sample_rate: u32,
               frames_to_process: jack::Frames,
               scope: &jack::ProcessScope,
               sine_lookup_table: &util::sine::LookupTable) -> () {
        // Copy all of the inputs to their buffers.
        for (_, ipab) in self.es_jack_inputs.iter_mut() {
            ipab.copy(scope);
        }
        // Run every program for the given number of frames, populating the
        // output buffers they are associated with.
        for (output_id, executable) in self.es_executables.iter_mut() {
            if let Some(opab) = self.es_jack_outputs.get_mut(output_id) {
                for frame in 0..frames_to_process {
                    opab.set(
                        frame as usize,
                        executable.run(
                            current_frame,
                            sample_rate,
                            &self.es_jack_inputs,
                            frame,
                            sine_lookup_table
                        )
                    );
                }
            }
        }
        // Now that we've run all of the programs, we can memcopy the JACK
        // outputs.
        for (_, opab) in self.es_jack_outputs.iter_mut() {
            opab.copy(scope);
        }
    }
}

/// Synchronization mechanism between processor and controller
///
/// Will be found inside an Arc, of which the processor and controller threads
/// each have a clone.
///
/// The controller thread will wait on the condition variable immediately after
/// it passes a new executable state to the processor through the mutex.
///
/// Only after the processor signals it will the controller wake up and perform
/// any necessary deallocation.
pub struct Synchro<T> {
    synchro_mutex: Mutex<T>,
    synchro_cond: Condvar
}

/// Shared state between processor and controller. Will be held inside a mutex
/// by way of Synchro<SharedState>.
///
/// Morally I wanted
///
///   data SharedState t = NoChange | Staged change | Processed change
///
/// But, for technical reasons regarding its use in a mutex, and dealing with
/// mutable references to it, "Processed" is expressed as "Staged" with a "true"
/// second field. That's because I can't figure out how to (or whether it should
/// be possible to) change a Staged(mut_ref_to_thing) (given by locking the
/// mutex) into a Processed(thing), which I want to replace the mutex value
/// with. Since mut_ref_to_thing is a mutable reference, I can't make a
/// Processed from it. By using a bool instead I can just flip it and everything
/// works.
pub enum Stage<T> {
    NoStage,
    Staged(T, bool),
}

type SharedState = Arc<Synchro<Stage<ExecutableState>>>;

/// The `Controller` (contrast with the `Processor`) represents the
/// non-time-critical piece of the software.
///
/// Where as the processor has hard deadlines to read, process, and write data
/// (controlled by a JACK client), the controller mostly sits idle, waiting for
/// user input which determines what the processor will do when it runs.
///
/// Transferring data between the controller and the processor is a key part of
/// this software. The processor callback must never yield: no allocations or
/// deallocations, no potentially blocking calls. This is done by way of the
/// `SharedState` (which the processor has a clone of) and by deriving the
/// processor's `ExecutableState` from the controller's `DescriptiveState`.
///
/// See `Controller.stage` and the `jack::ProcessHandler` implementation for
/// `Processor`.
pub struct Controller {
    cont_client:   jack::AsyncClient<(), Processor>,
    cont_state:    DescriptiveState,
    cont_shared:   SharedState
}

impl Controller {

    pub fn client(&self) -> &jack::Client {
        return self.cont_client.as_client();
    }

    /// Stage the current state by generating the executable state and updating
    /// the mutex-protected location.
    ///
    /// Expects that when the lock is acquired, it contains `NoStage`.
    /// When this routine finishes, the mutex will again contain `NoStage`.
    /// It will put a `Staged(st, false)` into the mutex, then wait on the
    /// condition variable until it becomes `Staged(st', true)`, at which point
    /// it will set the mutex to `NoStage`, release the lock, and deallocate
    /// all necessary things including st' (thereby ensuring the process
    /// callback does not deallocate it).
    ///
    /// See the `jack::ProcessHandler` implementation for `Processor` for the
    /// other side of the concurrency story.
    pub fn stage(&mut self) {
        let new_executable_state = self.cont_state.make_executable();
        // Dereference goes through the Arc to get the Synchro.
        // We take a reference; don't want to move it from self.
        let synchro: &Synchro<Stage<ExecutableState>> = &(*self.cont_shared);
        // We get a mutex guard (the thing that locks the mutex when it goes
        // out of scope). Dereferencing gives the stage, of which we'll take a
        // mutable reference.
        let mut guard = synchro.synchro_mutex.lock().unwrap();
        // Stage is mutable because we'll use it to dereference from the guard
        // later on, and in a loop.
        let mut stage = &mut (*guard);
        match stage {
            // TODO better panic message.
            // Why is it impossible? Because we set this back to NoStage
            // at the end of this route.
            Stage::Staged(_, _) => panic!("impossible"),
            Stage::NoStage => {}
        }
        // Write the new executable state to the mutex, then wait on the
        // condition variable for the processor thread to signal it.
        *stage = Stage::Staged(new_executable_state, false);
        // Docs say spurious wakeups are possible, so if we still see a
        // Staged value we try again.
        // Possibility of livelock in case the processor thread never sets
        // processed to true. But the only case it wouldn't do that is if it
        // panics.
        loop {
            guard = synchro.synchro_cond.wait(guard).unwrap();
            stage = &mut (*guard);
            match stage {
                // Processor changed it back to NoStage. That's a bug.
                Stage::NoStage => panic!("processor put NoStage"),
                Stage::Staged(_old_executable_state, processed) => {
                    // If *processed is false, assume it's a spurious wakeup.
                    if *processed {
                        // Break from the loop, thereby de-allocating
                        // _old_executable_state
                        break;
                    }
                }
            }
        }
        *stage = Stage::NoStage;
        // TODO FIXME we need to ensure that the deallocation of any prior
        // descriptive state parts does not happen until here.
    }

    /// Set the program for a given output identifier.
    pub fn set_program(&mut self, id: u32, prog: Program<Identifier, Number>) -> () {
        self.cont_state.ds_programs.insert(id, prog);
    }

    /// Add a JACK audio output. The JACK client is synchronously updated.
    ///
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
        // TODO FIXME you'd think this can cause a segfault if it overwrites an
        // existing output. The process callback still has raw pointers to
        // the thing that was removed (returned by the insert call). But I can't
        // make it happen...
        // Well, sure it won't happen, because the program still owns the page
        // (we don't do very much allocation/deallocation in the demos so far).
        //
        // Anyway, the solution is to keep track of the change and only apply it
        // at the call to `stage`, keeping all of the deleted things around
        // until the whole transaction with the process callback has been
        // finished.
        self.cont_state.ds_jack_outputs.insert(id, output);
    }

    /// Add a JACK audio input. The JACK client is synchronously updated.
    ///
    /// The string name is required because it will show up on JACK, but the
    /// u32 identifier is used internally.
    pub fn add_input(&mut self, name: &str, id: u32) -> () {
        let port = self.cont_client.as_client().register_port(name, jack::AudioIn).unwrap();
        let vbuffer = vec![0.0; self.cont_client.as_client().buffer_size() as usize];
        let input = JackAudioInput {
            jai_buffer: vbuffer,
            jai_port:port
        };
        self.cont_state.ds_jack_inputs.insert(id, input);
    }
}

/// TODO document
pub struct Processor {
    // jack::Frames is 32 bits.
    // If we're keeping a counter of how many frames have passed since the
    // beginning of time, then we'll need something bigger.
    // If we use 64 bits, we can go for over 2 trillion days at 96KHz.
    // If we use 32 bits, we can only go for half a day!
    // So, just use u64... but that complicates the implementation of buffer
    // indexing for 32 bit machines.
    current_frame: u64,
    sample_rate: u32,
    sine_lookup_table: util::sine::LookupTable,
    // TODO subsumes the above 3 fields.
    proc_state:    ExecutableState,
    // Changes to the system come in through here.
    proc_shared:   SharedState
}

impl Processor {
    /// Run a processor using a jack client. This will activate the client.
    /// You get a Controller in return, which can be used to make the JACK
    /// client sing.
    pub fn run(client: jack::Client) -> Controller {
        let jack_inputs = HashMap::new();
        let jack_outputs = HashMap::new();
        let programs = HashMap::new();
        let mut descriptive_state = DescriptiveState {
            ds_jack_inputs: jack_inputs,
            ds_jack_outputs: jack_outputs,
            ds_programs: programs
        };
        let executable_state = descriptive_state.make_executable();
        let stage = Stage::NoStage;
        let synchro = Synchro {
            synchro_mutex: Mutex::new(stage),
            synchro_cond: Condvar::new()
        };
        let shared = Arc::new(synchro);
        let processor = Processor {
            current_frame: 0,
            sample_rate: client.sample_rate() as u32,
            sine_lookup_table: util::sine::LookupTable::new(client.sample_rate()),
            proc_state: executable_state,
            proc_shared: shared.clone()
        };
        let async_client = client.activate_async((), processor).unwrap();
        let controller = Controller {
            cont_client: async_client,
            cont_state: descriptive_state,
            cont_shared: shared
        };
        return controller;
    }
}

impl jack::ProcessHandler for Processor {

    /// The JACK process callback.
    /// It runs the multimedia program for the number of frames given by the
    /// scope.
    /// It will also try, without blocking, to acquire changes to the program
    /// from the shared state.
    fn process(&mut self, _client: &jack::Client, scope: &jack::ProcessScope) -> jack::Control {

        // TODO FIXME it's probably possible to check for staged changes without
        // acquiring a lock. We don't even need atomic operations for that.
        // Can just read a cell and if the controller puts something in,
        // we'll _eventually_ see it and the controller will be waiting for
        // us to take the mutex and wake it by signalling the condition var.

        // To begin, try to acquire the mutex in a non-blocking way.
        let synchro: &Synchro<Stage<ExecutableState>> = &(*self.proc_shared);
        let mut result = synchro.synchro_mutex.try_lock();
        // If this pattern doesn't match, it means the controller thread has
        // the lock, but that's totally fine: the pointers in our executable
        // state are guaranteed to still be valid so we can carry on processing
        // this frame.
        if let Ok(ref mut mutex) = result {
            // Got the lock. The typical case is that there's no staged changes.
            let stage: &mut Stage<ExecutableState> = &mut (*mutex);
            match stage {
                Stage::NoStage => {},
                Stage::Staged(new_executable_state, processed) => {
                    // NB: it's possible that we see processed = true, because
                    // the controller thread has not yet picked this up.
                    if !(*processed) {
                        // Swap the states, and then update the stage.
                        // new_executable_state already is an &mut type, because
                        // stage is &mut Stage<ExecutableState>
                        std::mem::swap(new_executable_state, &mut self.proc_state);
                        *processed = true;
                        synchro.synchro_cond.notify_one();
                    }
                }
            }
        }

        let frames_to_process = scope.n_frames();
        self.proc_state.run(
            self.current_frame,
            self.sample_rate,
            frames_to_process,
            scope,
            &self.sine_lookup_table
        );
        // Cast is fine, frames_to_process is u32
        self.current_frame += frames_to_process as u64;
        return jack::Control::Continue;
    }
}
