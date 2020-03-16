use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex};
use jack;

use crate::lang::instruction::ExecutionState;
use crate::lang::instruction::IsInputRegion;
use crate::lang::instruction::IsOutputRegion;
use crate::lang::instruction::InputId;
use crate::lang::instruction::OutputId;
use crate::lang::instruction::RawPointer;
use crate::lang::instruction;
use crate::lang::update::Update;

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
pub enum Synchro<S> {
    Staged(S, Processed),
    Empty
}

type Processed = bool;

/// Has a NotificationHandler trait.
/// TODO: deal with sample rate and buffer size changes, by killing the system
/// (unless they can actually be supported...)
pub struct JackNotifier {
    synchro: Arc<(Mutex<bool>, Condvar)>
}

impl jack::NotificationHandler for JackNotifier {
    fn shutdown(&mut self, _status: jack::ClientStatus, _reason: &str) {
        let (mutex, condvar) = &*self.synchro;
        let mut guard = mutex.lock().unwrap();
        *guard = true;
        condvar.notify_one();
    }
}

pub struct ShutdownNotifier {
    synchro: Arc<(Mutex<bool>, Condvar)>
}

impl ShutdownNotifier {
    /// Returns when the JACK client has shutdown.
    /// "The JACK client" refers to the one on the JackController from which
    /// this ShutdownNotifier was created by way of shutdown_notifier.
    pub fn wait_for_shutdown(&mut self) {
        let (mutex, condvar) = &*self.synchro;
        let mut guard = mutex.lock().unwrap();
        while !(*guard) {
            guard = condvar.wait(guard).unwrap();
        }
    }
}

/// When this is dropped, the associated JACK async client is dropped, and
/// therefore deactivated.
pub struct JackController {
    jack_async_client: jack::AsyncClient<JackNotifier, JackProcessor>,
    // Mutex/condvar for waiting for system shutdown. Will be 'true' when the
    // system has shut down (when JACK calls the shutdown notification callback)
    system_shutdown: Arc<(Mutex<bool>, Condvar)>,
    synchro: Arc<(Mutex<Synchro<instruction::Program>>, Condvar)>,
    // Points to the same memory that an associated JackProcessor's
    // ExecutableState has in its RawPointer.
    memory: RawPointer
}

// It contains a RawPointer so rust is wary of moving it to a new thread.
unsafe impl Send for JackController {}

impl JackController {

    pub fn async_client(&self) -> &jack::AsyncClient<JackNotifier, JackProcessor> {
        return &self.jack_async_client;
    }

    pub fn shutdown_notifier(&self) -> ShutdownNotifier {
        return ShutdownNotifier { synchro: self.system_shutdown.clone() };
    }

    /// Use some FnMut to get an update, while holding the lock that causes
    /// the JACK processor to skip over its try_lock call at each period.
    ///
    /// Intended use is to call this function in a loop, where the update
    /// comes from some I/O source like stdin.
    ///
    /// If the function gives Err, this call releases the lock and returns that
    /// value. If it gives Ok, the update is digested and Ok(()) is returned.
    ///
    pub fn next<E, F: FnMut() -> Result<Update, E>>(&mut self, mut get_next: F) -> Result<(), E> {

        // Take the lock while waiting for the update to come in.
        let (mutex, condvar) = &*self.synchro;
        let mut guard = mutex.lock().unwrap();
        if let Synchro::Staged(_, _) = *guard {
            panic!("JackController.next: observed staged at first lock");
        }

        let (m_program, memory_updates) = match get_next() {
            Err(e) => { return Err(e); }
            Ok(update) => update.get_parts()
        };

        // Always update the memory.
        // Yes, this is terribly unsafe, but that's the idea. The user
        // programmer must ensure that their updates are safe, i.e. that the
        // running program (which they control) does not access the memory which
        // they update here.
        for memory_update in memory_updates.iter() {
            self.memory.set(memory_update.offset as usize, &memory_update.data);
        }

        // If there's no new program, we'll skip this block and drop the guard
        // (and therefore unlock the mutex). None means "no change" rather than
        // "set to empty program".
        //
        // But if there is a new program then we stage it (put in a Staged
        // value) and wait on the condvar until the process thread has
        // flipped the bool.
        //
        // How does this actually work though? When is _old_program
        // deallocated, given that all we have is a mutable reference to it?
        if let Some(new_program) = m_program {
            *guard = Synchro::Staged(new_program, false);
            loop {
                guard = condvar.wait(guard).unwrap();
                match &*guard {
                    Synchro::Empty => {
                        panic!("JackController.next: observed empty after staging");
                    }
                    Synchro::Staged(old_program, processed) => {
                        // processed could be false, due to a spurious wakeup.
                        // That's not an error.
                        if *processed {
                            // Make explicit the fact that the controller is
                            // the one to de-allocate the program that was
                            // running prior.
                            std::mem::drop(old_program);
                            *guard = Synchro::Empty;
                            break;
                        }
                    }
                }
            }
        }
        return Ok(());
    }

}

pub enum SomePort {
    AudioIn(InputId,  jack::Port<jack::AudioIn>),
    AudioOut(OutputId, jack::Port<jack::AudioOut>),

    /// Specify a buffer size for the MIDI data, i.e. how much MIDI data can
    /// be handled per frame (not per period).
    MidiIn(InputId, usize, jack::Port<jack::MidiIn>),
    /// As for MidiIn, give a buffer size.
    MidiOut(OutputId, usize, jack::Port<jack::MidiOut>)
}

pub enum Error {
    DuplicateInputRegionId(InputId),
    DuplicateOutputRegionId(OutputId),
    JackClientError(jack::Error)
}

/// Bring up a JackController against a JACK client, with a given amount of
/// memory available to the object program to be run. The initial program is
/// the empty_program (stops immediately).
/// The given JACK client will be activated.
/// Every port in the list should be registered with the JACK client.
/// Input/output regions for each port will be created with the given
/// identifier. Duplicate port numbers give errors.
pub fn run(
    memory_size: usize,
    jack_client: jack::Client,
    mut jack_ports: Vec<SomePort>) -> Result<JackController, Error> {

    // Create the synchronization structure between controller and processor.
    let mutex = Mutex::new(Synchro::Empty);
    let condvar = Condvar::new();
    let arc_controller = Arc::new((mutex, condvar));
    let arc_processor  = arc_controller.clone();

    // Mutex/condvar for system shutdown notification.
    let n_mutex = Mutex::new(false);
    let n_condvar = Condvar::new();
    let arc_notifier = Arc::new((n_mutex, n_condvar));
    let arc_shutdown = arc_notifier.clone();

    // Create the memory area for the executable. We shall retain our own
    // RawPointer to it so that we can do (incredibly unsafe) mutations to it
    // according to the user programmer's commands.
    let mut boxed_memory = (vec![0x00; memory_size]).into_boxed_slice();
    let memory_controller = RawPointer { ptr: (*boxed_memory).as_mut_ptr(), size: memory_size };
    let memory_processor  = memory_controller.clone();
    // Crucial: make sure rust doesn't drop the boxed memory.
    // FIXME probably better to just throw it onto the Controller, rather than
    // giving a RawPointer.
    // We could surely maintain the invariant that the Controller is not
    // dropped before the JACK clientis deactivated.
    std::mem::forget(boxed_memory);

    // Create JACK I/O regions according to the list of ports provided.
    // This means allocating memory for the MIDI I/O buffers.
    let mut inputs: HashMap<InputId, JackInputRegion> = HashMap::new();
    let mut outputs: HashMap<OutputId, JackOutputRegion> = HashMap::new();
    for port_and_id in jack_ports.drain(..) {
        match port_and_id {
            SomePort::AudioIn(in_id, port_in) => {
                if let Some(_) = inputs.insert(in_id, JackInputRegion::JackAudioInput(port_in)) {
                    return Err(Error::DuplicateInputRegionId(in_id));
                }
            }
            SomePort::AudioOut(out_id, port_out) => {
                if let Some(_) = outputs.insert(out_id, JackOutputRegion::JackAudioOutput(port_out)) {
                    return Err(Error::DuplicateOutputRegionId(out_id));
                }
            }
            SomePort::MidiIn(in_id, buf_size, port_in) => {
                let buf = (vec![0x00; buf_size]).into_boxed_slice();
                if let Some(_) = inputs.insert(in_id, JackInputRegion::JackMidiInput(port_in, buf)) {
                    return Err(Error::DuplicateInputRegionId(in_id));
                }
            }
            SomePort::MidiOut(out_id, buf_size, port_out) => {
                let buf = (vec![0x00; buf_size]).into_boxed_slice();
                if let Some(_) = outputs.insert(out_id, JackOutputRegion::JackMidiOutput(port_out, buf)) {
                    return Err(Error::DuplicateOutputRegionId(out_id));
                }
            }
        }
    }

    let execution_state = ExecutionState {
        program: instruction::empty_program(),
        memory: memory_processor,
        inputs: inputs,
        outputs: outputs,
        global_frame: 0,
        local_frame: 0,
        sample_rate: jack_client.sample_rate() as u32
    };
    let processor = JackProcessor {
        execution_state: execution_state,
        synchro: arc_processor
    };

    let notification_handler = JackNotifier {
        synchro: arc_notifier
    };

    let async_client = match jack_client.activate_async(notification_handler, processor) {
        Err(err) => { return Err(Error::JackClientError(err)); }
        Ok(async_client) => { async_client }
    };

    let controller = JackController {
        jack_async_client: async_client,
        system_shutdown: arc_shutdown,
        synchro: arc_controller,
        memory: memory_controller
    };

    return Ok(controller);

}


/// Important because it has the jack::ProcessHandler trait.
pub struct JackProcessor {
    execution_state: ExecutionState<JackInputRegion, JackOutputRegion>,
    synchro: Arc<(Mutex<Synchro<instruction::Program>>, Condvar)>
}

// ugh. rust refuses to let me pass raw mutable pointers between threads...
// unless I explicitly say yeah I wanna do that. Why the hell am I using a
// low-level language in the first place? It's exactly what I want to do,
// obviously.
unsafe impl Send for JackProcessor {}
unsafe impl Sync for JackProcessor {}

impl JackProcessor {
    /// Assumptions:
    /// - There is precisely one other thread that has access to the mutex and
    ///   condvar in self.synchro.
    /// - That thread will hold almost always have the lock.
    /// - When that thread releases the lock, the mutex contains Staged, and
    ///   it then waits on the condvar.
    /// - When it returns from the condvar it expects the mutex to contain
    ///   Returned.
    fn update_state_nonblocking(&mut self) {
        let (mutex, condvar) = &*self.synchro;
        let mut lock = mutex.try_lock();
        if let Ok(ref mut synchro) = lock {
            // When we get the lock, we have a MutexGuard which gives us only
            // a mutable reference.
            match **synchro {
                // Can and will happen, in particular at startup, when the
                // control thread (which should hold the lock almost always)
                // and the JACK process thread spin up concurrently.
                Synchro::Empty => {}
                Synchro::Staged(ref mut s, ref mut processed) => {
                    if *processed {
                        // The controller thread must put `processed` in as
                        // false. However, it may be possible that we get the
                        // lock again before the controller thread does, even
                        // though we hit the condvar that it must be waiting on.
                    } else {
                        // FIXME is this the "proper" way to do this? I want to
                        // take out the old program, then put in the new one, but
                        // that can't be done directly due to the ownership/borrow
                        // checker.
                        std::mem::swap(&mut self.execution_state.program, s);
                        self.execution_state.local_frame = 0;
                        *processed = true;
                        condvar.notify_one();
                    }
                }
            }
        }
    }
}

impl jack::ProcessHandler for JackProcessor {
    fn process(&mut self, _: &jack::Client, ps: &jack::ProcessScope) -> jack::Control {
        self.update_state_nonblocking();
        let frames_to_process = ps.n_frames();
        let result = instruction::execute_period(
                &mut self.execution_state,
                frames_to_process as usize,
                ps,
                ps
            );
        match result {
            Ok(()) => { return jack::Control::Continue; }
            Err(_err) => { 
                // FIXME TODO what really should we do in case of an
                // execution error? Are they always fatal?
                //
                // TODO print the error
                println!("JACK client has quit. You probably want to kill this program.");
                return jack::Control::Quit;
            }
        }
    }
}


pub enum JackInputRegion {
    JackAudioInput(jack::Port<jack::AudioIn>),
    JackMidiInput(jack::Port<jack::MidiIn>, Box<[u8]>)
}

impl IsInputRegion for JackInputRegion {

    type T = jack::ProcessScope;
    // TODO better type for this
    type E = ();

    fn prepare_frame(&mut self, rframe: usize, ps: &jack::ProcessScope) {
        match self {
            JackInputRegion::JackAudioInput(_) => {}
            // For MIDI inputs we make a length-prefixed array where each
            // element is a pointer to some place in this region (a u32).
            // It points to a length-prefixed string of bytes where the
            // length is a u8.
            //
            // FIXME this implementation is not ideal, since it iterates over
            // all MIDI events _for the period_ at each _frame_.
            JackInputRegion::JackMidiInput(port, boxed_buffer) => {
                let buffer = &mut (*boxed_buffer);
                let mut offset = 1;
                let mut event_count: u8 = 0x00;
                // At most 256 MIDI events. The actual MIDI data starts at the
                // end of the array of pointers, which is 256*4 + 1 bytes
                // (1025).
                let mut addr: usize = 0x00000401;
                for raw_midi in port.iter(ps) {
                    if raw_midi.time != rframe as u32 { continue; }
                    // FIXME check for overflow.
                    // This will panic and kill the program :O :O
                    buffer[offset..offset+3].copy_from_slice(&addr.to_le_bytes());
                    offset += 4;
                    let size = raw_midi.bytes.len() as usize;
                    buffer[addr..addr+3].copy_from_slice(&size.to_le_bytes());
                    addr += 4;
                    buffer[addr..(addr+size)].copy_from_slice(raw_midi.bytes);
                    addr += size;
                    event_count += 1;
                }
                buffer[0] = event_count;
            }
        }
    }

    fn read(&self, offset: usize, size: usize, rframe: usize, ps: &jack::ProcessScope) -> Result<*const u8, ()> {
        match self {
            JackInputRegion::JackAudioInput(port) => {
                // JACK input values are 32-bit floats, and we index by the byte, so
                // we'll check that we're not overstepping it first.
                if offset + size > 4 {
                    return Err(());
                } else {
                    unsafe {
                        let f32_ptr: *const f32 = port.as_slice(ps).as_ptr().add(rframe);
                        let ptr: *const u8 = f32_ptr as *const u8;
                        return Ok(ptr.add(offset));
                    }
                }
            }
            JackInputRegion::JackMidiInput(_port, boxed_buffer) => {
                let buffer = &(*boxed_buffer);
                if (offset + size) > (buffer.len() + 1) {
                    return Err(());
                } else {
                    unsafe {
                        let ptr: *const u8 = buffer.as_ptr().add(offset);
                        return Ok(ptr);
                    }
                }
            }
        }
    }
}

pub enum JackOutputRegion {
    JackAudioOutput(jack::Port<jack::AudioOut>),
    JackMidiOutput(jack::Port<jack::MidiOut>, Box<[u8]>)
}

impl IsOutputRegion for JackOutputRegion {
    type T = jack::ProcessScope;
    type E = ();
    fn prepare_frame(&mut self, rframe: usize, ps: &jack::ProcessScope) {
        match self {
            // Must write to the entire JACK buffer at every process callback.
            // FIXME TODO this is probably not the most efficient way.
            // Could instead do double-buffering and zero the buffer at the
            // beginning of every period.
            // OR just say it's the user programmer's fault if they fail to
            // write on every frame... anyway, TBD
            JackOutputRegion::JackAudioOutput(port) => {
                port.as_mut_slice(ps)[rframe] = 0.0f32;
            }
            JackOutputRegion::JackMidiOutput(_, _) => {}
        }
    }
    fn write(&mut self, offset: usize, size: usize, rframe: usize, ps: &jack::ProcessScope) -> Result<*mut u8, ()> {
        match self {
            JackOutputRegion::JackAudioOutput(port) => {
                if offset + size > 4 {
                    return Err(());
                } else {
                    unsafe {
                        let f32_ptr: *mut f32 = port.as_mut_slice(ps).as_mut_ptr().add(rframe);
                        let ptr: *mut u8 = f32_ptr as *mut u8;
                        return Ok(ptr.add(offset));
                    }
                }
            }
            // The output region is the buffer at the given offset.
            // It's flushed after each frame.
            JackOutputRegion::JackMidiOutput(_port, boxed_buffer) => {
                let buffer = &mut (*boxed_buffer);
                if (offset + size) > (buffer.len() + 1) {
                    return Err(());
                } else {
                    unsafe {
                        let ptr: *mut u8 = buffer.as_mut_ptr().add(offset);
                        return Ok(ptr);
                    }
                }
            }
        }
    }
    fn flush_frame(&mut self, rframe: usize, ps: &jack::ProcessScope) {
        match self {
            // Outputs are written directly by `write`.
            JackOutputRegion::JackAudioOutput(_) => {}
            // For MIDI, we write at `flush_frame`, because the MIDI data is composite
            // unlike the f32 for audio ports. We want the object program to
            // have a consistent memory model: if you write something to a
            // MIDI output region, then overwrite it, the first thing should
            // not be written out to the wire, which is exactly the case for
            // audio outputs.
            //
            // This is called after each frame execution. We just check
            // the first byte, and interpret it as the number of MIDI events
            // to write out.
            //
            // Each event is a length-prefixed (u32) byte string.
            JackOutputRegion::JackMidiOutput(port, boxed_buffer) => {
                let buffer = &mut (*boxed_buffer);
                let mut offset: usize = 0;
                let num_events = u8::from_le_bytes([buffer[offset]]);
                if num_events == 0 { return; }
                offset += 1;
                let mut writer = port.writer(ps);
                let mut processed_events = 0;
                while processed_events < num_events {
                    let addr = u32::from_le_bytes([
                            buffer[offset],
                            buffer[offset+1],
                            buffer[offset+2],
                            buffer[offset+3]
                        ]) as usize;
                    offset += 4;
                    let size = u32::from_le_bytes([
                            buffer[addr],
                            buffer[addr+1],
                            buffer[addr+2],
                            buffer[addr+3]
                        ]) as usize;
                    let bytes = &buffer[(addr+4)..(addr+4+size)];
                    let rawmidi = jack::RawMidi { time: rframe as u32, bytes: bytes };
                    // FIXME use the result value.
                    let _ = writer.write(&rawmidi);
                    processed_events += 1;
                }
                // Reset the event count to 0, so the next frame doesn't
                // get the same events.
                // It would be better to zero the whole array, but this is
                // an output port, so it cannot be read from. A user programmer
                // could abuse this by just setting the event count back again
                // to replay MIDI events but whatever, no big deal.
                buffer[offset] = 0x00;
            }
        }
    }
}
