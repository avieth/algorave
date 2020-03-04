use std::cmp::Ordering;
use std::collections::HashMap;

// FIXME TODO factor out the stuff only related to execution into its own
// module.

/* # Note: definition of a program
 *
 * To avoid doing anything too clever, and to ensure fast-as-possible execution,
 * we'll use an architecture very close to the typical microprocessor:
 * instructions and data. However we'll not store instructions in the same
 * memory as data. The rust program can accept uploads of
 *
 * - Program instructions (text section). This completely replaces the previous
 *   instructions.
 * - A zero-initialized blob of a given size (data section). Program
 *   instructions may address these.
 * - An explicit blob (data section). Useful for working with audio samples:
 *   the rust program itself does not read files or transcode/resample the
 *   audio format, it just takes a bunch of bytes and puts them somewhere on
 *   the heap. Program instructions may address these just like zero-initialized
 *   data sections.
 * - JACK PCM/MIDI I/O ports, which appear as memory regions as well, but are
 *   treated specially (random-access only per-frame, never R/W).
 * 
 * ## What can the programmer do?
 *
 * - Change the program instructions.
 *   The entire new program instruction list must be uploaded.
 * - Add, remove, or overwrite any data section.
 *   That's to say: set it to an optional value.
 *   It is impossible to change the size of a data section without losing its
 *   data. That's just due to technical reasons: if we wanted to copy over the
 *   contents from the existing buffer to the new one, we would need to
 *   synchronize with (i.e. risk blocking) the process thread.
 * - Add or remove a JACK I/O port.
 *   Adding ports happens before the process thread becomes aware of them. But
 *   when a port is removed, the JACK client won't actually update until the
 *   process thread has received the new system and the control thread has
 *   recovered the port.
 *
 * Multiple such actions may be done atomically, so that we can ensure that the
 * program instructions never address something that does not exist.
 *
 * ## Identifying data sections and JACK ports
 *
 * The simplest thing, it would seem, is to give each of these things a
 * programmer-chosen `usize` identifier.
 * The problem is: how do we efficiently index these in rust? If we use a
 * HashMap, we don't get constant-time lookups. If we use an array, then the
 * programmer must use dense indentifiers, or we'll waste memory.
 * Maybe HashMap is the right choice? Its lookup is O(1) "expected". Ah,
 * because it depends upon the distribution of the keys.
 *
 * Ok, so we use a `HashMap<u32, Region>`. The `HashMap` owns its values.
 * A `Region` probably must be a `*mut u8` with a size. Reading or writing
 * to/from a region means: check the bounds, then read the given slice (offset
 * plus size) from the mut pointer.
 *
 * What you should get from a read is a `*const u8`. What you should get from a
 * write is a `*mut u8`. That's to say, regions should just expose
 *
 *   get_section(offset: usize, size: usize) -> *const u8
 *   get_mut_section(offset: usize, size: usize) -> *mut u8
 *
 * then we can define the typed read/write functions on *const u8 and *mut u8.
 * Done.
 *
 * In fact, get_section and get_mut_section shall need the current sample as
 * well
 *
 *   // None means out-of-bounds.
 *   get_section(frame: usize, offset: usize, size: usize) -> Option<*const u8>
 *   get_mut_section(frame: usize, offset: usize, size: usize) -> Option<*mut u8>
 *
 * Really though, memory read/write happen basically at every instruction.
 * Putting each of these behind a HashMap lookup seems like a bad idea.
 * Jack port I/O can be behind a HashMap, since these don't happen as often,
 * but main memory... should just be a pointer dereference.
 *
 * Another feasible option is to take the "heap size" as a parameter at startup.
 * The rust program allocates that much contiguous memory, and it cannot grow
 * or shrink.
 *
 *   let buffer: Vec<u8> = vec![0; dynamic_config_length];
 *   let boxed_buffer: Box<[u8]> = buffer.into_boxed_slice();
 *
 * What would it be like to work with this for "non-volatile" memory? The
 * frontend code generator would remember where various non-volatile things are
 * stored. They could in theory be moved, too, using the "on-first-frame"
 * construction.
 *
 * How does it work for uploading PCM data? Ahhhhh it can't work, because
 * we would need to synchronize with the process thread in order to write it
 * in. Damn.
 * The point is: the memory that the object program deals with, you cannot touch
 * it once you have made it available.
 * In theory, we could maybe get away with memcpy'ing to some region, if
 * we're sure the program does not touch that area...
 * The frontend is able to determine which memory areas are safe to write... it
 * determines the entire program text, after all.
 * Maybe let's just go with that? It allows you to do really stupid shit but
 * whatever. Programmer can
 *
 * - Set the program (text section).
 * - Upload data to a particular address (data section).
 * - Add and remove JACK I/O ports (PCM and MIDI).
 *
 * and this can all be done atomically.
 * For data uploads: they will be, in order, memcpy'd to the buffer that the
 * process thread is actively working on, _before_ the new program is
 * swapped in. If this is to give predicable results, you must ensure that the
 * target area is not being written to nor read from by the running program.
 *
 * Who owns the actual memory buffer? The control thread, surely.
 * And we may as well just put it in an Arc to share it safely with the
 * process handler... that won't be a big deal: dereference the Arc once...
 * ah no, but then rust will make us use a mutex on it. Forget it, just use
 * an unsafe from_raw_parts.
 * The lifetime of this thing is the lifetime of the control thread, which is
 * longer than that of the JACK realtime process thread, so it's all good.
 */

// Not using usize for these. We need to define an address space for the object
// language. 32 bits ought to be enough.

/// Errors relating to reading and writing from and to I/O ports.
#[derive(Debug)]
pub enum IOError<InErr, OutErr> {
    UnknownInputPort(InputId),
    UnknownOutputPort(OutputId),
    InputError(InErr),
    OutputError(OutErr)
}

#[derive(Debug)]
pub enum AccessError<InErr, OutErr> {
    MemoryAccessError(MemoryError),
    IOAccessError(IOError<InErr, OutErr>)
}

#[derive(Debug)]
pub enum MemoryError {
    /// Attempted to access memory out of bounds.
    /// This uses the native usize, rather than the object-language size of
    /// u32, because the memory access happens ultimately on the host (duh).
    MemoryAccessOutOfBounds(usize, usize)
}

#[derive(Debug)]
pub enum ControlError {
    /// Attempted to jump to an index that is out of bounds in the program
    /// instruction vector.
    InvalidJump(i32)
}

#[derive(Debug)]
pub enum ExecutionError<In, Out> {
    JumpError(ControlError),
    AccessError(AccessError<In, Out>),
}

pub struct RawPointer {
    pub ptr: *mut u8,
    pub size: usize
}

impl RawPointer {
    /// Get another RawPointer which points to the same memory and has the same
    /// size.
    pub fn clone(&self) -> RawPointer {
        return RawPointer { ptr: self.ptr, size: self.size };
    }

    /// Copy bytes in from a vector, writing them at a given offset.
    pub fn set(&mut self, offset: usize, src: &Vec<u8>) {
        unsafe {
            std::ptr::copy(src.as_ptr(), self.ptr.add(offset), src.len());
        }
    }
}

pub struct FrameInfo<'a, In, Out> {
    pub rframe: usize,
    pub memory: &'a RawPointer,
    pub inputs: &'a mut HashMap<InputId, In>,
    pub outputs: &'a mut HashMap<OutputId, Out>
}

pub struct ExecutionState<In, Out> {
    pub program: Vec<Instruction>,
    /// "data section": r/w memory addressable by the program instructions.
    /// It's a mut pointer because, in practice, we'll want some other part of
    /// the program to have ownership of the array which it points to. This is
    /// the only way that we can get multiple mutable references, as far as I
    /// know.
    pub memory: RawPointer,
    pub inputs: HashMap<InputId, In>,
    pub outputs: HashMap<OutputId, Out>,
    /// Frames processed since the system was brought up.
    pub global_frame: u64,
    /// Frames processed since the current program was loaded in.
    pub local_frame: u64,
    /// System sample rate. Should never change.
    pub sample_rate: u32
}

pub fn execute_period<In: IsInputRegion, Out: IsOutputRegion>(
    es: &mut ExecutionState<In, Out>,
    frames_to_process: usize,
    input_t: &In::T,
    output_t: &Out::T
    ) -> Result<(), ExecutionError<In::E, Out::E>> {
    let mut frame_info = FrameInfo {
            rframe: 0,
            memory: &es.memory,
            inputs: &mut es.inputs,
            outputs: &mut es.outputs
        };
    for _ in 0..frames_to_process {
        prepare_frame_inputs(&mut frame_info, input_t);
        prepare_frame_outputs(&mut frame_info, output_t);
        let mut instruction_pointer: u32 = 0;
        loop {
            // get_unchecked is unsafe -_-
            unsafe {
                let instruction = es.program.get_unchecked(instruction_pointer as usize);
                // FIXME perhaps an unresolvable problem is that, if the
                // program loops, it is impossible for the user programmer
                // to update the program to make it not loop. Not sure how
                // this could be done in rust... perhaps the program should
                // periodically allow for interrupt? Maybe do that on every
                // jump?
                let result = execute_instruction(
                        instruction,
                        &mut frame_info,
                        input_t,
                        output_t,
                        es.global_frame,
                        es.local_frame,
                        es.sample_rate
                    );
                match result {
                    Err(err) => {
                        return Err(ExecutionError::AccessError(err));
                    }
                    Ok(Control::Jump(roffset)) => {
                        let destination = instruction_pointer as i32 + roffset;
                        if destination < 0 {
                            return Err(ExecutionError::JumpError(ControlError::InvalidJump(destination)));
                        } else if destination as usize > es.program.len() {
                            return Err(ExecutionError::JumpError(ControlError::InvalidJump(destination)));
                        } else {
                            instruction_pointer = destination as u32;
                            // Continue in the `loop`, not the outer `for`.
                            continue;
                        }
                    }
                    // Break out of the `loop`, returning to the outer `for`.
                    Ok(Control::Stop) => { break; }
                    Ok(Control::Next) => {
                        instruction_pointer += 1;
                        continue;
                    }
                }
            }
        }
        // Flush outputs. This is usually a no-op.
        // For MIDI ports though, this is quite convenient: they can pass the
        // MIDI data to JACK and then reset their buffer, avoiding book-keeping
        // w.r.t. the relative frame.
        flush_frame_outputs(&mut frame_info, output_t);
        // Whole program has been run for this frame. Must make some updates
        // before the next frame.
        frame_info.rframe += 1;
        es.global_frame += 1;
        es.local_frame += 1;

    }
    return Ok(());
}

pub fn prepare_frame_inputs<In: IsInputRegion, Out: IsOutputRegion>(
    frame_info: &mut FrameInfo<In, Out>,
    input_t : &In::T) {
    for (_, input) in frame_info.inputs.iter_mut() {
        input.prepare_frame(frame_info.rframe, input_t);
    }
}

pub fn prepare_frame_outputs<In: IsInputRegion, Out: IsOutputRegion>(
    frame_info: &mut FrameInfo<In, Out>,
    output_t : &Out::T) {
    for (_, output) in frame_info.outputs.iter_mut() {
        output.prepare_frame(frame_info.rframe, output_t);
    }
}

pub fn flush_frame_outputs<In: IsInputRegion, Out: IsOutputRegion>(
    frame_info: &mut FrameInfo<In, Out>,
    output_t: &Out::T) {
    for (_, output) in frame_info.outputs.iter_mut() {
        output.flush_frame(frame_info.rframe, output_t);
    }
}


// TODO explain the idea of an input region and why it depends upon the
// "relative frame" or rframe.

pub trait IsInputRegion {
    type T;
    type E;
    //fn prepare_period(&mut self, frames: usize, t: &Self::T);
    fn prepare_frame(&mut self, rframe: usize, t: &Self::T);
    /// Get a const pointer of a given size, at a given offset, relative to
    /// the rframe
    fn read(&self, offset: usize, size: usize, rframe: usize, t: &Self::T) -> Result<*const u8, Self::E>;
}

pub trait IsOutputRegion {
    type T;
    type E;
    fn prepare_frame(&mut self, rframe: usize, t: &Self::T);
    /// Get a mut pointer of a given size, at a given offset, relative to
    /// the rframe.
    fn write(&mut self, offset: usize, size: usize, rframe: usize, _: &Self::T) -> Result<*mut u8, Self::E>;
    fn flush_frame(&mut self, rframe: usize, t: &Self::T);
}

/// Useful if you want to run a program with fixed simulated "inputs".
/// The buffer size must be length * stride. The bytes for a given `rframe`
/// begin at `rframe * stride`.
///
/// Of course, you are responsible for ensuring that the pointer remains
/// valid whenever this region is used.
pub struct MemoryIORegion {
    pub buffer: *mut u8,
    pub stride: usize,
    pub length: usize
}

impl IsInputRegion for MemoryIORegion {
    type T = ();
    type E = ();
    fn prepare_frame(&mut self, _: usize, _: &()) {}
    fn read(&self, offset: usize, size: usize, rframe: usize, _: &()) -> Result<*const u8, ()> {
        if offset > (self.length - 1) || offset + size > self.length {
            return Err(());
        } else {
            unsafe {
                return Ok(self.buffer.add(self.stride * rframe + offset));
            }
        }
    }
}

impl IsOutputRegion for MemoryIORegion {
    type T = ();
    type E = ();
    fn prepare_frame(&mut self, _: usize, _: &()) {}
    fn write(&mut self, offset: usize, size: usize, rframe: usize, _: &()) -> Result<*mut u8, ()> {
        if offset > (self.length - 1) || offset + size > self.length {
            return Err(());
        } else {
            unsafe {
                return Ok(self.buffer.add(self.stride * rframe + offset));
            }
        }
    }
    fn flush_frame(&mut self, _: usize, _: &()) {}
}

// TODO sharpen this type. It used to be this would also read from memory, then
// that was removed but the type was left wide.
fn region_slice<Void, In: IsInputRegion, AnyOut>(
    input_id: &InputId,
    offset: usize,
    size: usize,
    frame_info: &FrameInfo<'_, In, AnyOut>,
    t: &In::T,
    ) -> Result<*const u8, AccessError<In::E, Void>> {
    match frame_info.inputs.get(input_id) {
        Some(input_region) => {
            match input_region.read(offset, size, frame_info.rframe, t) {
                Err(err) => { return Err(AccessError::IOAccessError(IOError::InputError(err))); }
                Ok(ptr) => { return Ok(ptr); }
            }
        }
        None => { return Err(AccessError::IOAccessError(IOError::UnknownInputPort(*input_id))); }
    }
}

// TODO sharpen this type. It used to be this would also read from memory, then
// that was removed but the type was left wide.
fn region_slice_mut<Void, AnyIn, Out: IsOutputRegion>(
    output_id: &OutputId,
    offset: usize,
    size: usize,
    frame_info: &mut FrameInfo<'_, AnyIn, Out>,
    t: &Out::T) -> Result<*mut u8, AccessError<Void, Out::E>> {
    match frame_info.outputs.get_mut(output_id) {
        Some(output_region) => {
            match output_region.write(offset, size, frame_info.rframe, t) {
                Err(err) => { return Err(AccessError::IOAccessError(IOError::OutputError(err))); }
                Ok(ptr) => { return Ok(ptr); }
            }
        }
        None => { return Err(AccessError::IOAccessError(IOError::UnknownOutputPort(*output_id))); }
    }
}

/// Get a memory slice at the given offset, of a given size.
pub fn memory_slice(memory: &RawPointer, offset: usize, size: usize) -> Result<*const u8, MemoryError> {
    // If the memory is of size n, the highest address is n-1.
    // The largest address that will be read is offset  offset + (size - 1)
    // (no address is read if size is 0).
    //
    // What if the size is 0? The only case where the _user_ programmer can
    // make that happen is by doing an arbitrary-length copy. But in that
    // case, it will work out to a copy(void, src, 0) which is fine.
    if size == 0 {
        return Ok(std::ptr::null());
    } else if offset + (size - 1) >= memory.size {
        return Err(MemoryError::MemoryAccessOutOfBounds(offset, size));
    } else {
        unsafe {
            return Ok(memory.ptr.add(offset));
        }
    }
}

/// Get a mutable memory slice at the given offset, of a given slice.
pub fn memory_slice_mut(memory: &RawPointer, offset: usize, size: usize) -> Result<*mut u8, MemoryError> {
    // See comment in `memory_slice`.
    //
    // The null pointer given back here will only be written to by copying
    // 0 bytes, which is fine.
    if size == 0 {
        return Ok(std::ptr::null_mut());
    } else if offset + (size - 1) >= memory.size {
        return Err(MemoryError::MemoryAccessOutOfBounds(offset, size));
    } else {
        unsafe {
            return Ok(memory.ptr.add(offset));
        }
    }
}

impl<I, O> From<MemoryError> for AccessError<I, O> {
    fn from(err: MemoryError) -> AccessError<I, O> {
        return AccessError::MemoryAccessError(err);
    }
}

pub type Program = Vec<Instruction>;

/// An empty vector would cause a panic because we don't do bounds checking when
/// retrieving the first instruction. The empty program is in fact one
/// instruction: stop.
pub fn empty_program() -> Program {
    return vec![Instruction::Stop];
}

pub type Size = u32;

pub type Offset = u32;

pub type Address = u32;

pub type InputId = u8;
pub type OutputId = u8;

/// Unsigned integral types.
#[derive(Debug)]
pub enum UIType {
    TU8,
    TU16,
    TU32,
    TU64
}

impl UIType {
    pub fn size(&self) -> usize {
        match self {
            UIType::TU8  => { return 1; },
            UIType::TU16 => { return 2; },
            UIType::TU32 => { return 4; },
            UIType::TU64 => { return 8; }
        }
    }
    pub unsafe fn add(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            UIType::TU8  => { write_u8 (c, read_u8(a)  + read_u8(b));  }
            UIType::TU16 => { write_u16(c, read_u16(a) + read_u16(b)); }
            UIType::TU32 => { write_u32(c, read_u32(a) + read_u32(b)); }
            UIType::TU64 => { write_u64(c, read_u64(a) + read_u64(b)); }
        }
    }
    pub unsafe fn sub(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            UIType::TU8  => { write_u8 (c, read_u8(a)  - read_u8(b));  }
            UIType::TU16 => { write_u16(c, read_u16(a) - read_u16(b)); }
            UIType::TU32 => { write_u32(c, read_u32(a) - read_u32(b)); }
            UIType::TU64 => { write_u64(c, read_u64(a) - read_u64(b)); }
        }
    }
    pub unsafe fn mul(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            UIType::TU8  => { write_u8 (c, read_u8(a)  * read_u8(b));  }
            UIType::TU16 => { write_u16(c, read_u16(a) * read_u16(b)); }
            UIType::TU32 => { write_u32(c, read_u32(a) * read_u32(b)); }
            UIType::TU64 => { write_u64(c, read_u64(a) * read_u64(b)); }
        }
    }
    pub unsafe fn div(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            UIType::TU8  => { write_u8 (c, read_u8(a)  / read_u8(b));  }
            UIType::TU16 => { write_u16(c, read_u16(a) / read_u16(b)); }
            UIType::TU32 => { write_u32(c, read_u32(a) / read_u32(b)); }
            UIType::TU64 => { write_u64(c, read_u64(a) / read_u64(b)); }
        }
    }
    pub unsafe fn modulo(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            UIType::TU8  => { write_u8 (c, read_u8(a)  % read_u8(b));  }
            UIType::TU16 => { write_u16(c, read_u16(a) % read_u16(b)); }
            UIType::TU32 => { write_u32(c, read_u32(a) % read_u32(b)); }
            UIType::TU64 => { write_u64(c, read_u64(a) % read_u64(b)); }
        }
    }
    pub unsafe fn or(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            UIType::TU8  => { write_u8 (c, read_u8(a)  | read_u8(b));  }
            UIType::TU16 => { write_u16(c, read_u16(a) | read_u16(b)); }
            UIType::TU32 => { write_u32(c, read_u32(a) | read_u32(b)); }
            UIType::TU64 => { write_u64(c, read_u64(a) | read_u64(b)); }
        }
    }
    pub unsafe fn and(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            UIType::TU8  => { write_u8 (c, read_u8(a)  & read_u8(b));  }
            UIType::TU16 => { write_u16(c, read_u16(a) & read_u16(b)); }
            UIType::TU32 => { write_u32(c, read_u32(a) & read_u32(b)); }
            UIType::TU64 => { write_u64(c, read_u64(a) & read_u64(b)); }
        }
    }
    pub unsafe fn xor(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            UIType::TU8  => { write_u8 (c, read_u8(a)  ^ read_u8(b));  }
            UIType::TU16 => { write_u16(c, read_u16(a) ^ read_u16(b)); }
            UIType::TU32 => { write_u32(c, read_u32(a) ^ read_u32(b)); }
            UIType::TU64 => { write_u64(c, read_u64(a) ^ read_u64(b)); }
        }
    }
    pub unsafe fn not(&self, a: *const u8, b: *mut u8) {
        match self {
            UIType::TU8  => { write_u8 (b, !read_u8(a));  }
            UIType::TU16 => { write_u16(b, !read_u16(a)); }
            UIType::TU32 => { write_u32(b, !read_u32(a)); }
            UIType::TU64 => { write_u64(b, !read_u64(a)); }
        }
    }
    /// Second *const u8 is always read as a u8.
    pub unsafe fn shiftl(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            UIType::TU8  => { write_u8 (c, read_u8(a)  << read_u8(b));  }
            UIType::TU16 => { write_u16(c, read_u16(a) << read_u8(b)); }
            UIType::TU32 => { write_u32(c, read_u32(a) << read_u8(b)); }
            UIType::TU64 => { write_u64(c, read_u64(a) << read_u8(b)); }
        }
    }
    /// Second *const u8 is always read as a u8.
    pub unsafe fn shiftr(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            UIType::TU8  => { write_u8 (c, read_u8(a)  >> read_u8(b)); }
            UIType::TU16 => { write_u16(c, read_u16(a) >> read_u8(b)); }
            UIType::TU32 => { write_u32(c, read_u32(a) >> read_u8(b)); }
            UIType::TU64 => { write_u64(c, read_u64(a) >> read_u8(b)); }
        }
    }
    pub unsafe fn eq(&self, a: *const u8, b: *const u8, c: *mut u8) {
        let re = match self {
            UIType::TU8  => { if read_u8(a)  == read_u8(b)  { 0x01 } else { 0x00 } }
            UIType::TU16 => { if read_u16(a) == read_u16(b) { 0x01 } else { 0x00 } }
            UIType::TU32 => { if read_u32(a) == read_u32(b) { 0x01 } else { 0x00 } }
            UIType::TU64 => { if read_u64(a) == read_u64(b) { 0x01 } else { 0x00 } }
        };
        write_u8(c, re);
    }
    pub unsafe fn lt(&self, a: *const u8, b: *const u8, c: *mut u8) {
        let re = match self {
            UIType::TU8  => { if read_u8(a)  < read_u8(b)  { 0x01 } else { 0x00 } }
            UIType::TU16 => { if read_u16(a) < read_u16(b) { 0x01 } else { 0x00 } }
            UIType::TU32 => { if read_u32(a) < read_u32(b) { 0x01 } else { 0x00 } }
            UIType::TU64 => { if read_u64(a) < read_u64(b) { 0x01 } else { 0x00 } }
        };
        write_u8(c, re);
    }
    pub unsafe fn gt(&self, a: *const u8, b: *const u8, c: *mut u8) {
        let re = match self {
            UIType::TU8  => { if read_u8(a)  > read_u8(b)  { 0x01 } else { 0x00 } }
            UIType::TU16 => { if read_u16(a) > read_u16(b) { 0x01 } else { 0x00 } }
            UIType::TU32 => { if read_u32(a) > read_u32(b) { 0x01 } else { 0x00 } }
            UIType::TU64 => { if read_u64(a) > read_u64(b) { 0x01 } else { 0x00 } }
        };
        write_u8(c, re);
    }
}

/// Signed integral types.
#[derive(Debug)]
pub enum SIType {
    TI8,
    TI16,
    TI32,
    TI64
}

impl SIType {
    pub fn size(&self) -> usize {
        match self {
            SIType::TI8  => { return 1; },
            SIType::TI16 => { return 2; },
            SIType::TI32 => { return 4; },
            SIType::TI64 => { return 8; }
        }
    }
    pub unsafe fn cast_to(&self, fty: &FType, a: *const u8, b: *mut u8) {
        match (fty, self) {
            (FType::TF32, SIType::TI8)  => { write_f32(b, read_i8(a)  as f32); }
            (FType::TF32, SIType::TI16) => { write_f32(b, read_i16(a) as f32); }
            (FType::TF32, SIType::TI32) => { write_f32(b, read_i32(a) as f32); }
            (FType::TF32, SIType::TI64) => { write_f32(b, read_i64(a) as f32); }
            (FType::TF64, SIType::TI8)  => { write_f64(b, read_i8(a)  as f64); }
            (FType::TF64, SIType::TI16) => { write_f64(b, read_i16(a) as f64); }
            (FType::TF64, SIType::TI32) => { write_f64(b, read_i32(a) as f64); }
            (FType::TF64, SIType::TI64) => { write_f64(b, read_i64(a) as f64); }
        }
    }
    pub unsafe fn add(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            SIType::TI8  => { write_i8 (c, read_i8(a)  + read_i8(b));  }
            SIType::TI16 => { write_i16(c, read_i16(a) + read_i16(b)); }
            SIType::TI32 => { write_i32(c, read_i32(a) + read_i32(b)); }
            SIType::TI64 => { write_i64(c, read_i64(a) + read_i64(b)); }
        }
    }
    pub unsafe fn sub(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            SIType::TI8  => { write_i8 (c, read_i8(a)  - read_i8(b));  }
            SIType::TI16 => { write_i16(c, read_i16(a) - read_i16(b)); }
            SIType::TI32 => { write_i32(c, read_i32(a) - read_i32(b)); }
            SIType::TI64 => { write_i64(c, read_i64(a) - read_i64(b)); }
        }
    }
    pub unsafe fn mul(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            SIType::TI8  => { write_i8 (c, read_i8(a)  * read_i8(b));  }
            SIType::TI16 => { write_i16(c, read_i16(a) * read_i16(b)); }
            SIType::TI32 => { write_i32(c, read_i32(a) * read_i32(b)); }
            SIType::TI64 => { write_i64(c, read_i64(a) * read_i64(b)); }
        }
    }
    pub unsafe fn div(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            SIType::TI8  => { write_i8 (c, read_i8(a)  / read_i8(b));  }
            SIType::TI16 => { write_i16(c, read_i16(a) / read_i16(b)); }
            SIType::TI32 => { write_i32(c, read_i32(a) / read_i32(b)); }
            SIType::TI64 => { write_i64(c, read_i64(a) / read_i64(b)); }
        }
    }
    pub unsafe fn modulo(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            SIType::TI8  => { write_i8 (c, read_i8(a)  % read_i8(b));  }
            SIType::TI16 => { write_i16(c, read_i16(a) % read_i16(b)); }
            SIType::TI32 => { write_i32(c, read_i32(a) % read_i32(b)); }
            SIType::TI64 => { write_i64(c, read_i64(a) % read_i64(b)); }
        }
    }
    pub unsafe fn abs(&self, a: *const u8, b: *mut u8) {
        match self {
            SIType::TI8  => { write_i8 (b, read_i8(a).abs()); }
            SIType::TI16 => { write_i16(b, read_i16(a).abs()); }
            SIType::TI32 => { write_i32(b, read_i32(a).abs()); }
            SIType::TI64 => { write_i64(b, read_i64(a).abs()); }
        }
    }
    pub unsafe fn eq(&self, a: *const u8, b: *const u8, c: *mut u8) {
        let re = match self {
            SIType::TI8  => { if read_i8(a)  == read_i8(b)  { 0x01 } else { 0x00 } }
            SIType::TI16 => { if read_i16(a) == read_i16(b) { 0x01 } else { 0x00 } }
            SIType::TI32 => { if read_i32(a) == read_i32(b) { 0x01 } else { 0x00 } }
            SIType::TI64 => { if read_i64(a) == read_i64(b) { 0x01 } else { 0x00 } }
        };
        write_u8(c, re);
    }
    pub unsafe fn lt(&self, a: *const u8, b: *const u8, c: *mut u8) {
        let re = match self {
            SIType::TI8  => { if read_i8(a)  < read_i8(b)  { 0x01 } else { 0x00 } }
            SIType::TI16 => { if read_i16(a) < read_i16(b) { 0x01 } else { 0x00 } }
            SIType::TI32 => { if read_i32(a) < read_i32(b) { 0x01 } else { 0x00 } }
            SIType::TI64 => { if read_i64(a) < read_i64(b) { 0x01 } else { 0x00 } }
        };
        write_u8(c, re);
    }
    pub unsafe fn gt(&self, a: *const u8, b: *const u8, c: *mut u8) {
        let re = match self {
            SIType::TI8  => { if read_i8(a)  > read_i8(b)  { 0x01 } else { 0x00 } }
            SIType::TI16 => { if read_i16(a) > read_i16(b) { 0x01 } else { 0x00 } }
            SIType::TI32 => { if read_i32(a) > read_i32(b) { 0x01 } else { 0x00 } }
            SIType::TI64 => { if read_i64(a) > read_i64(b) { 0x01 } else { 0x00 } }
        };
        write_u8(c, re);
    }
}

#[derive(Debug)]
pub enum IType {
    TUnsigned(UIType),
    TSigned(SIType)
}

impl IType {
    pub fn size(&self) -> usize {
        match self {
            IType::TUnsigned(uitype) => { return uitype.size(); },
            IType::TSigned(sitype)   => { return sitype.size(); }
        }
    }
    pub unsafe fn add(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            IType::TUnsigned(uitype) => { uitype.add(a, b, c); }
            IType::TSigned(sitype)   => { sitype.add(a, b, c); }
        }
    }
    pub unsafe fn sub(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            IType::TUnsigned(uitype) => { uitype.sub(a, b, c); }
            IType::TSigned(sitype)   => { sitype.sub(a, b, c); }
        }
    }
    pub unsafe fn mul(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            IType::TUnsigned(uitype) => { uitype.mul(a, b, c); }
            IType::TSigned(sitype)   => { sitype.mul(a, b, c); }
        }
    }
    pub unsafe fn div(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            IType::TUnsigned(uitype) => { uitype.div(a, b, c); }
            IType::TSigned(sitype)   => { sitype.div(a, b, c); }
        }
    }
    pub unsafe fn modulo(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            IType::TUnsigned(uitype) => { uitype.modulo(a, b, c); }
            IType::TSigned(sitype)   => { sitype.modulo(a, b, c); }
        }
    }
    pub unsafe fn eq(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            IType::TUnsigned(uitype) => { uitype.eq(a, b, c); }
            IType::TSigned(sitype)   => { sitype.eq(a, b, c); }
        }
    }
    pub unsafe fn lt(&self, a: *const u8, b: *const u8, c: *mut u8) { 
        match self {
            IType::TUnsigned(uitype) => { uitype.lt(a, b, c); }
            IType::TSigned(sitype)   => { sitype.lt(a, b, c); }
        }
    }
    pub unsafe fn gt(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            IType::TUnsigned(uitype) => { uitype.gt(a, b, c); }
            IType::TSigned(sitype)   => { sitype.gt(a, b, c); }
        }
    }
}

/// Fractional types.
#[derive(Debug)]
pub enum FType {
    TF32,
    TF64
}

impl FType {
    pub fn size(&self) -> usize {
        match self {
            FType::TF32 => { return 4; },
            FType::TF64 => { return 8; }
        }
    }
    pub unsafe fn cast_to(&self, ity: &SIType, a: *const u8, b: *mut u8) {
        match (self, ity) {
            (FType::TF32, SIType::TI8)  => { write_i8(b, read_f32(a) as i8); }
            (FType::TF32, SIType::TI16) => { write_i16(b, read_f32(a) as i16); }
            (FType::TF32, SIType::TI32) => { write_i32(b, read_f32(a) as i32); }
            (FType::TF32, SIType::TI64) => { write_i64(b, read_f32(a) as i64); }
            (FType::TF64, SIType::TI8)  => { write_i8(b, read_f64(a) as i8); }
            (FType::TF64, SIType::TI16) => { write_i16(b, read_f64(a) as i16); }
            (FType::TF64, SIType::TI32) => { write_i32(b, read_f64(a) as i32); }
            (FType::TF64, SIType::TI64) => { write_i64(b, read_f64(a) as i64); }
        }
    }
    pub unsafe fn add(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            FType::TF32 => { write_f32(c, read_f32(a) + read_f32(b)); }
            FType::TF64 => { write_f64(c, read_f64(a) + read_f64(b)); }
        }
    }
    pub unsafe fn sub(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            FType::TF32 => { write_f32(c, read_f32(a) - read_f32(b)); }
            FType::TF64 => { write_f64(c, read_f64(a) - read_f64(b)); }
        }
    }
    pub unsafe fn mul(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            FType::TF32 => { write_f32(c, read_f32(a) * read_f32(b)); }
            FType::TF64 => { write_f64(c, read_f64(a) * read_f64(b)); }
        }
    }
    pub unsafe fn div(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            FType::TF32 => { write_f32(c, read_f32(a) / read_f32(b)); }
            FType::TF64 => { write_f64(c, read_f64(a) / read_f64(b)); }
        }
    }
    pub unsafe fn modulo(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            FType::TF32 => { write_f32(c, read_f32(a) % read_f32(b)); }
            FType::TF64 => { write_f64(c, read_f64(a) % read_f64(b)); }
        }
    }
    pub unsafe fn pow(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            FType::TF32 => { write_f32(c, read_f32(a).powf(read_f32(b))); }
            FType::TF64 => { write_f64(c, read_f64(a).powf(read_f64(b))); }
        }
    }
    pub unsafe fn log(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            FType::TF32 => { write_f32(c, read_f32(a).log(read_f32(b))); }
            FType::TF64 => { write_f64(c, read_f64(a).log(read_f64(b))); }
        }
    }
    pub unsafe fn sin(&self, a: *const u8, b: *mut u8) {
        match self {
            FType::TF32 => { write_f32(b, read_f32(a).sin()); }
            FType::TF64 => { write_f64(b, read_f64(a).sin()); }
        }
    }
    pub unsafe fn exp(&self, a: *const u8, b: *mut u8) {
        match self {
            FType::TF32 => { write_f32(b, read_f32(a).exp()); }
            FType::TF64 => { write_f64(b, read_f64(a).exp()); }
        }
    }
    pub unsafe fn ln(&self, a: *const u8, b: *mut u8) {
        match self {
            FType::TF32 => { write_f32(b, read_f32(a).ln()); }
            FType::TF64 => { write_f64(b, read_f64(a).ln()); }
        }
    }
    pub unsafe fn abs(&self, a: *const u8, b: *mut u8) {
        match self {
            FType::TF32 => { write_f32(b, read_f32(a).abs()); }
            FType::TF64 => { write_f64(b, read_f64(a).abs()); }
        }
    }
    pub unsafe fn eq(&self, a: *const u8, b: *const u8, c: *mut u8) {
        let re = match self {
            FType::TF32 => {
                let fa = read_f32(a);
                let fb = read_f32(b);
                if fa.eq(&fb) { 0x01 } else { 0x00 }
            }
            FType::TF64 => {
                let fa = read_f64(a);
                let fb = read_f64(b);
                if fa.eq(&fb) { 0x01 } else { 0x00 }
            }
        };
        write_u8(c, re);
    }
    pub unsafe fn lt(&self, a: *const u8, b: *const u8, c: *mut u8) {
        let re = match self {
            FType::TF32 => {
                let fa = read_f32(a);
                let fb = read_f32(b);
                match fa.partial_cmp(&fb) {
                    Some(Ordering::Less) => { 0x01 }
                    _ => { 0x00 }
                }
            }
            FType::TF64 => {
                let fa = read_f64(a);
                let fb = read_f64(b);
                match fa.partial_cmp(&fb) {
                    Some(Ordering::Less) => { 0x01 }
                    _ => { 0x00 }
                }
            }
        };
        write_u8(c, re);
    }
    pub unsafe fn gt(&self, a: *const u8, b: *const u8, c: *mut u8) {
        let re = match self {
            FType::TF32 => {
                let fa = read_f32(a);
                let fb = read_f32(b);
                match fa.partial_cmp(&fb) {
                    Some(Ordering::Greater) => { 0x01 }
                    _ => { 0x00 }
                }
            }
            FType::TF64 => {
                let fa = read_f64(a);
                let fb = read_f64(b);
                match fa.partial_cmp(&fb) {
                    Some(Ordering::Greater) => { 0x01 }
                    _ => { 0x00 }
                }
            }
        };
        write_u8(c, re);
    }
}

#[derive(Debug)]
pub enum Type {
    Integral(IType),
    Fractional(FType)
}

impl Type {
    pub fn size(&self) -> usize {
        match self {
            Type::Integral(itype)   => { return itype.size(); },
            Type::Fractional(ftype) => { return ftype.size(); }
        }
    }
    pub unsafe fn add(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            Type::Integral(itype)   => { itype.add(a, b, c); }
            Type::Fractional(ftype) => { ftype.add(a, b, c); }
        }
    }
    pub unsafe fn sub(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            Type::Integral(itype)   => { itype.sub(a, b, c); }
            Type::Fractional(ftype) => { ftype.sub(a, b, c); }
        }
    }
    pub unsafe fn mul(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            Type::Integral(itype)   => { itype.mul(a, b, c); }
            Type::Fractional(ftype) => { ftype.mul(a, b, c); }
        }
    }
    pub unsafe fn div(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            Type::Integral(itype)   => { itype.div(a, b, c); }
            Type::Fractional(ftype) => { ftype.div(a, b, c); }
        }
    }
    pub unsafe fn modulo(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            Type::Integral(itype)   => { itype.modulo(a, b, c); }
            Type::Fractional(ftype) => { ftype.modulo(a, b, c); }
        }
    }
    pub unsafe fn eq(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            Type::Integral(itype)   => { itype.eq(a, b, c); }
            Type::Fractional(ftype) => { ftype.eq(a, b, c); }
        }
    }
    pub unsafe fn gt(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            Type::Integral(itype)   => { itype.gt(a, b, c); }
            Type::Fractional(ftype) => { ftype.gt(a, b, c); }
        }
    }
    pub unsafe fn lt(&self, a: *const u8, b: *const u8, c: *mut u8) {
        match self {
            Type::Integral(itype)   => { itype.lt(a, b, c); }
            Type::Fractional(ftype) => { ftype.lt(a, b, c); }
        }
    }
}

#[derive(Debug)]
pub enum Constant {
    /// u64 giving the current sample.
    Now,
    /// u64 giving the first sample for which this program was run.
    /// That's to say, when the process thread receives a new program, it will
    /// reset the start value to the current sample.
    /// A program is thereby capable of doing an initialization path, by
    /// checking whether Now == Start.
    Start,
    /// u32 giving the sample rate.
    Rate,
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64)
}

impl Constant {
    pub fn size(&self) -> usize {
        match self {
            Constant::Now    => { UIType::TU64.size() }
            Constant::Start  => { UIType::TU64.size() }
            Constant::Rate   => { UIType::TU32.size() }
            Constant::U8(_)  => { UIType::TU8.size() }
            Constant::U16(_) => { UIType::TU16.size() }
            Constant::U32(_) => { UIType::TU32.size() }
            Constant::U64(_) => { UIType::TU64.size() }
            Constant::I8(_)  => { SIType::TI8.size() }
            Constant::I16(_) => { SIType::TI16.size() }
            Constant::I32(_) => { SIType::TI32.size() }
            Constant::I64(_) => { SIType::TI64.size() }
            Constant::F32(_) => { FType::TF32.size() }
            Constant::F64(_) => { FType::TF64.size() }
        }
    }
}

// TODO improvements to the instruction set?
// - A jump instruction, to avoid the tedium of a branch if equal.
// - For every instruction that takes 2 SVals, let one of them be a literal?
#[derive(Debug)]
pub enum Instruction {

    /// Print the memory at a given address of a given size.
    /// First is the size (U32), second is the address (pointer).
    Trace(Address, Address),

    /// End the program.
    Stop,
    /// Branch to the relative address in the first location if the value in the
    /// second location is 0 as a u8.
    /// Relative address is a signed 32 bit integer giving the number of
    /// instructions to move by. 0 loops, 1 is the same as not branching.
    Branch(Address, Address),
    /// Jump to the relative address in this location (unconditional branch).
    Jump(Address),

    /// Bitwise copy from source to destination.
    /// The addresses are pointers: their values will be treated as u32
    /// addresses. The first is the source, second is destination.
    Copy(Type, Address, Address),
    /// Put a constant into some memory address.
    Set(Constant, Address),
    /// Get input. The first address gives a u8 identifier.
    /// The second gives an offset into that input region (it's a pointer, like
    /// for Copy).
    Read(Type, Address, Address, Address),
    /// Put output. The first address gives a u8 identifier.
    /// The second gives an offset into that output region (it's a pointer, like
    /// for Copy).
    Write(Type, Address, Address, Address),

    Or(UIType, Address, Address, Address),
    And(UIType, Address, Address, Address),
    Xor(UIType, Address, Address, Address),
    Not(UIType, Address, Address),
    /// The second address is for a U8 giving the number of bits to shift.
    Shiftl(UIType, Address, Address, Address),
    /// The second address is for a U8 giving the number of bits to shift.
    Shiftr(UIType, Address, Address, Address),

    Add(Type, Address, Address, Address),
    Sub(Type, Address, Address, Address),
    Mul(Type, Address, Address, Address),
    Div(Type, Address, Address, Address),
    Mod(Type, Address, Address, Address),

    Abs(SIType, Address, Address),
    Absf(FType, Address, Address),

    /// Second is the exponent.
    Pow(FType, Address, Address, Address),
    /// Second is the base.
    Log(FType, Address, Address, Address),
    Sin(FType, Address, Address),
    Exp(FType, Address, Address),
    Ln(FType, Address, Address),

    /// Result is 0b00000001 if the two values are equal as the given
    /// type or 0b00000000 otherwise. For fractional types this is partial
    /// equality; some things are not equal to themselves.
    Eq(Type, Address, Address, Address),
    /// Result is 0b00000001 if the first value is less than the second,
    /// and 0b00000000 otherwise. For fractional types this is a partial
    /// ordering; if something is not less than another thing, it's not
    /// necessarily greater or equal.
    Lt(Type, Address, Address, Address),
    /// See doc for Lt.
    Gt(Type, Address, Address, Address),

    /// Down- or up-cast a fractional type. The result will be the other
    /// fractional size.
    Castf(FType, Address, Address),
    /// Cast a fractional value to a signed integral value.
    Ftoi(FType, SIType, Address, Address),
    /// Cast a signed integral value to a fractional value.
    Itof(SIType, FType, Address, Address)

}

// TODO general form of instructions?
// 1. get 0 or more *const pointers (reads)
// 2. get 1 *mut pointer (write)
// 3. use the type information to act on these
//
// It seems like we can't do this generically because rust doesn't have
// existential types. I'd like to express that, whatever the type indicator is,
// it can always read from a *const u8 that has the proper size.
//
// Add(ty, src1, src2, dst) => {
//     let src1_ptr = region_slice(src1, ..., ty.size(), ...)
//     let src2_ptr = region_slice(src2, ..., ty.size(), ...)
//     let dst_ptr = region_slice_mut(dst, ..., ty.size(), ...)
//     ty.add(src1_ptr, src2_ptr, dst_ptr)
// }
//
// ^ that's one way to do it... define the add function against *const u8 and
// *mut u8 pointers...
//
// Copy(ty, src, dst) => {
//     let src_addr = TU32.read(region_slice(src, ..., 4, ...))
//     let dst_addr = TU32.read(region_slice(dst, ..., 4, ...))
//     let src_ptr = region_slice(src_addr, ..., ty.size(), ...)
//     let dst_ptr = region_slice_mut(dst_addr, ..., ty.size(), ...)
//     // Always a bitwise copy.
//     memcpy(dst_ptr, src_ptr, ty.size())
// }

/// Control flow indicator: program is either done, goes to the next
/// instruction, or jumps to some relative offset in the instruction
/// vector.
pub enum Control {
    Jump(i32),
    Stop,
    Next
}


pub fn execute_instruction<In: IsInputRegion, Out: IsOutputRegion>(
    inst: &Instruction,
    frame_info: &mut FrameInfo<In, Out>,
    input_t: &In::T,
    output_t: &Out::T,
    global_frame: u64,
    local_frame: u64,
    rate: u32) -> Result<Control, AccessError<In::E, Out::E>> {
    // TODO FIXME is it possible to shorten this match? There's lots and lots of
    // redundancy. But sadly, I'm not sufficiently familiar with rust to see a
    // way to do it.
    match inst {
        Instruction::Trace(size, addr) => {
            let size = read_u32(memory_slice(frame_info.memory, *size as usize, 4)?);
            let addr_ptr = read_u32(memory_slice(frame_info.memory, *addr as usize, 4)?);
            let ptr = memory_slice(frame_info.memory, addr_ptr as usize, size as usize)?;
            // TODO do this better. The sink should be a parameter, so that in
            // realtime scenarios like the JACK backend it can be, for instance,
            // a lock-free queue.
            unsafe { println!("{:X?} {:X?}", addr, read_bytes(ptr, size as usize)) };
        }
        Instruction::Stop => {
            return Ok(Control::Stop);
        }
        Instruction::Branch(jmp, chk) => {
            let chk_ptr = memory_slice(frame_info.memory, *chk as usize, UIType::TU8.size())?;
            let jmp_ptr = memory_slice(frame_info.memory, *jmp as usize, SIType::TI32.size())?;
            if read_u8(chk_ptr) == 0x00 {
                return Ok(Control::Jump(read_i32(jmp_ptr)));
            }
        }
        Instruction::Jump(jmp) => {
            let jmp_ptr = memory_slice(frame_info.memory, *jmp as usize, SIType::TI32.size())?;
            return Ok(Control::Jump(read_i32(jmp_ptr)));
        }
        Instruction::Copy(ty, src, dst) => {
            let size = ty.size();
            let src_addr = read_u32(memory_slice(frame_info.memory, *src as usize, UIType::TU32.size())?);
            let dst_addr = read_u32(memory_slice(frame_info.memory, *dst as usize, UIType::TU32.size())?);
            let src_ptr = memory_slice(frame_info.memory, src_addr as usize, size)?;
            let dst_ptr = memory_slice_mut(frame_info.memory, dst_addr as usize, size)?;
            unsafe { std::ptr::copy(src_ptr, dst_ptr, size) };
        }
        Instruction::Read(ty, src_id, src_offset, dst) => {
            let size = ty.size();
            let input_id = read_u8(memory_slice(frame_info.memory, *src_id as usize, UIType::TU8.size())?);
            let offset = read_u32(memory_slice(frame_info.memory, *src_offset as usize, UIType::TU32.size())?);
            let src_ptr = region_slice(&input_id, offset as usize, size, frame_info, input_t)?;
            let dst_ptr = memory_slice_mut(frame_info.memory, *dst as usize, size)?;
            unsafe { std::ptr::copy(src_ptr, dst_ptr, size) };
        }
        // Confusing nomenclature here: the "src" prefix means from where we
        // source the information which will ultimately determine the
        // destination.
        Instruction::Write(ty, src_id, src_offset, src_data) => {
            let size = ty.size();
            let output_id = read_u8(memory_slice(frame_info.memory, *src_id as usize, UIType::TU8.size())?);
            let offset = read_u32(memory_slice(frame_info.memory, *src_offset as usize, UIType::TU32.size())?);
            let src_ptr = memory_slice(frame_info.memory, *src_data as usize, size)?;
            let dst_ptr = region_slice_mut(&output_id, offset as usize, size, frame_info, output_t)?;
            unsafe { std::ptr::copy(src_ptr, dst_ptr, size) };
        }
        Instruction::Set(constant, dst) => {
            let size = constant.size();
            let dst_ptr = memory_slice_mut(frame_info.memory, *dst as usize, size)?;
            match constant {
                Constant::Now    => { write_u64(dst_ptr, global_frame); }
                Constant::Start  => { write_u64(dst_ptr, local_frame); }
                Constant::Rate   => { write_u32(dst_ptr, rate); }
                Constant::U8(x)  => { write_u8(dst_ptr, *x); }
                Constant::U16(x) => { write_u16(dst_ptr, *x); }
                Constant::U32(x) => { write_u32(dst_ptr, *x); }
                Constant::U64(x) => { write_u64(dst_ptr, *x); }
                Constant::I8(x)  => { write_i8(dst_ptr, *x); }
                Constant::I16(x) => { write_i16(dst_ptr, *x); }
                Constant::I32(x) => { write_i32(dst_ptr, *x); }
                Constant::I64(x) => { write_i64(dst_ptr, *x); }
                Constant::F32(x) => { write_f32(dst_ptr, *x); }
                Constant::F64(x) => { write_f64(dst_ptr, *x); }
            }
        }
        Instruction::Or(uity, src1, src2, dst) => {
            let size = uity.size();
            let src1_ptr = memory_slice(frame_info.memory, *src1 as usize, size)?;
            let src2_ptr = memory_slice(frame_info.memory, *src2 as usize, size)?;
            let dst_ptr  = memory_slice_mut(frame_info.memory, *dst as usize, size)?;
            unsafe { uity.or(src1_ptr, src2_ptr, dst_ptr) };
        }
        Instruction::And(uity, src1, src2, dst) => {
            let size = uity.size();
            let src1_ptr = memory_slice(frame_info.memory, *src1 as usize, size)?;
            let src2_ptr = memory_slice(frame_info.memory, *src2 as usize, size)?;
            let dst_ptr  = memory_slice_mut(frame_info.memory, *dst as usize, size)?;
            unsafe { uity.and(src1_ptr, src2_ptr, dst_ptr) };
        }
        Instruction::Xor(uity, src1, src2, dst) => {
            let size = uity.size();
            let src1_ptr = memory_slice(frame_info.memory, *src1 as usize, size)?;
            let src2_ptr = memory_slice(frame_info.memory, *src2 as usize, size)?;
            let dst_ptr  = memory_slice_mut(frame_info.memory, *dst as usize, size)?;
            unsafe { uity.xor(src1_ptr, src2_ptr, dst_ptr) };
        }
        Instruction::Not(uity, src, dst) => {
            let size = uity.size();
            let src_ptr = memory_slice(frame_info.memory, *src as usize, size)?;
            let dst_ptr  = memory_slice_mut(frame_info.memory, *dst as usize, size)?;
            unsafe { uity.not(src_ptr, dst_ptr) };
        }
        Instruction::Shiftl(uity, src1, src2, dst) => {
            let size = uity.size();
            let src1_ptr = memory_slice(frame_info.memory, *src1 as usize, size)?;
            let src2_ptr = memory_slice(frame_info.memory, *src2 as usize, UIType::TU8.size())?;
            let dst_ptr  = memory_slice_mut(frame_info.memory, *dst as usize, size)?;
            unsafe { uity.shiftl(src1_ptr, src2_ptr, dst_ptr) };
        }
        Instruction::Shiftr(uity, src1, src2, dst) => {
            let size = uity.size();
            let src1_ptr = memory_slice(frame_info.memory, *src1 as usize, size)?;
            let src2_ptr = memory_slice(frame_info.memory, *src2 as usize, UIType::TU8.size())?;
            let dst_ptr  = memory_slice_mut(frame_info.memory, *dst as usize, size)?;
            unsafe { uity.shiftr(src1_ptr, src2_ptr, dst_ptr) };
        }
        Instruction::Add(ty, src1, src2, dst) => {
            let size = ty.size();
            let src1_ptr = memory_slice(frame_info.memory, *src1 as usize, size)?;
            let src2_ptr = memory_slice(frame_info.memory, *src2 as usize, size)?;
            let dst_ptr  = memory_slice_mut(frame_info.memory, *dst as usize, size)?;
            unsafe { ty.add(src1_ptr, src2_ptr, dst_ptr) };
        }
        Instruction::Sub(ty, src1, src2, dst) => {
            let size = ty.size();
            let src1_ptr = memory_slice(frame_info.memory, *src1 as usize, size)?;
            let src2_ptr = memory_slice(frame_info.memory, *src2 as usize, size)?;
            let dst_ptr  = memory_slice_mut(frame_info.memory, *dst as usize, size)?;
            unsafe { ty.sub(src1_ptr, src2_ptr, dst_ptr) };
        }
        Instruction::Mul(ty, src1, src2, dst) => {
            let size = ty.size();
            let src1_ptr = memory_slice(frame_info.memory, *src1 as usize, size)?;
            let src2_ptr = memory_slice(frame_info.memory, *src2 as usize, size)?;
            let dst_ptr  = memory_slice_mut(frame_info.memory, *dst as usize, size)?;
            unsafe { ty.mul(src1_ptr, src2_ptr, dst_ptr) };
        }
        Instruction::Div(ty, src1, src2, dst) => {
            let size = ty.size();
            let src1_ptr = memory_slice(frame_info.memory, *src1 as usize, size)?;
            let src2_ptr = memory_slice(frame_info.memory, *src2 as usize, size)?;
            let dst_ptr  = memory_slice_mut(frame_info.memory, *dst as usize, size)?;
            unsafe { ty.div(src1_ptr, src2_ptr, dst_ptr) };
        }
        Instruction::Mod(ty, src1, src2, dst) => {
            let size = ty.size();
            let src1_ptr = memory_slice(frame_info.memory, *src1 as usize, size)?;
            let src2_ptr = memory_slice(frame_info.memory, *src2 as usize, size)?;
            let dst_ptr  = memory_slice_mut(frame_info.memory, *dst as usize, size)?;
            unsafe { ty.modulo(src1_ptr, src2_ptr, dst_ptr) };
        }
        Instruction::Pow(fty, src1, src2, dst) => {
            let size = fty.size();
            let src1_ptr = memory_slice(frame_info.memory, *src1 as usize, size)?;
            let src2_ptr = memory_slice(frame_info.memory, *src2 as usize, size)?;
            let dst_ptr  = memory_slice_mut(frame_info.memory, *dst as usize, size)?;
            unsafe { fty.pow(src1_ptr, src2_ptr, dst_ptr) };
        }
        Instruction::Log(fty, src1, src2, dst) => {
            let size = fty.size();
            let src1_ptr = memory_slice(frame_info.memory, *src1 as usize, size)?;
            let src2_ptr = memory_slice(frame_info.memory, *src2 as usize, size)?;
            let dst_ptr  = memory_slice_mut(frame_info.memory, *dst as usize, size)?;
            unsafe { fty.log(src1_ptr, src2_ptr, dst_ptr) };
        }
        Instruction::Sin(fty, src, dst) => {
            let size = fty.size();
            let src_ptr = memory_slice(frame_info.memory, *src as usize, size)?;
            let dst_ptr = memory_slice_mut(frame_info.memory, *dst as usize, size)?;
            unsafe { fty.sin(src_ptr, dst_ptr) };
        }
        Instruction::Exp(fty, src, dst) => {
            let size = fty.size();
            let src_ptr = memory_slice(frame_info.memory, *src as usize, size)?;
            let dst_ptr = memory_slice_mut(frame_info.memory, *dst as usize, size)?;
            unsafe { fty.exp(src_ptr, dst_ptr) };
        }
        Instruction::Ln(fty, src, dst) => {
            let size = fty.size();
            let src_ptr = memory_slice(frame_info.memory, *src as usize, size)?;
            let dst_ptr = memory_slice_mut(frame_info.memory, *dst as usize, size)?;
            unsafe { fty.ln(src_ptr, dst_ptr) };
        }
        Instruction::Abs(ity, src, dst) => {
            let size = ity.size();
            let src_ptr = memory_slice(frame_info.memory, *src as usize, size)?;
            let dst_ptr = memory_slice_mut(frame_info.memory, *dst as usize, size)?;
            unsafe { ity.abs(src_ptr, dst_ptr) };
        }
        Instruction::Absf(fty, src, dst) => {
            let size = fty.size();
            let src_ptr = memory_slice(frame_info.memory, *src as usize, size)?;
            let dst_ptr = memory_slice_mut(frame_info.memory, *dst as usize, size)?;
            unsafe { fty.abs(src_ptr, dst_ptr) };
        }
        Instruction::Eq(ty, src1, src2, dst) => {
            let size = ty.size();
            let src1_ptr = memory_slice(frame_info.memory, *src1 as usize, size)?;
            let src2_ptr = memory_slice(frame_info.memory, *src2 as usize, size)?;
            let dst_ptr  = memory_slice_mut(frame_info.memory, *dst as usize, UIType::TU8.size())?;
            unsafe { ty.eq(src1_ptr, src2_ptr, dst_ptr) };
        }
        Instruction::Gt(ty, src1, src2, dst) => {
            let size = ty.size();
            let src1_ptr = memory_slice(frame_info.memory, *src1 as usize, size)?;
            let src2_ptr = memory_slice(frame_info.memory, *src2 as usize, size)?;
            let dst_ptr  = memory_slice_mut(frame_info.memory, *dst as usize, UIType::TU8.size())?;
            unsafe { ty.gt(src1_ptr, src2_ptr, dst_ptr) };
        }
        Instruction::Lt(ty, src1, src2, dst) => {
            let size = ty.size();
            let src1_ptr = memory_slice(frame_info.memory, *src1 as usize, size)?;
            let src2_ptr = memory_slice(frame_info.memory, *src2 as usize, size)?;
            let dst_ptr  = memory_slice_mut(frame_info.memory, *dst as usize, UIType::TU8.size())?;
            unsafe { ty.lt(src1_ptr, src2_ptr, dst_ptr) };
        }
        Instruction::Castf(fty, src, dst) => {
            let src_ptr = memory_slice(frame_info.memory, *src as usize, fty.size())?;
            match fty {
                FType::TF32 => {
                    let dst_ptr = memory_slice_mut(frame_info.memory, *dst as usize, FType::TF64.size())?;
                    write_f64(dst_ptr, read_f32(src_ptr) as f64);
                }
                FType::TF64 => {
                    let dst_ptr = memory_slice_mut(frame_info.memory, *dst as usize, FType::TF32.size())?;
                    write_f32(dst_ptr, read_f64(src_ptr) as f32);
                }
            }
        }
        Instruction::Ftoi(fty, sity, src, dst) => {
            let src_ptr = memory_slice(frame_info.memory, *src as usize, fty.size())?;
            let dst_ptr = memory_slice_mut(frame_info.memory, *dst as usize, sity.size())?;
            unsafe { fty.cast_to(sity, src_ptr, dst_ptr) };
        }
        Instruction::Itof(sity, fty, src, dst) => {
            let src_ptr = memory_slice(frame_info.memory, *src as usize, sity.size())?;
            let dst_ptr = memory_slice_mut(frame_info.memory, *dst as usize, fty.size())?;
            unsafe { sity.cast_to(fty, src_ptr, dst_ptr) };
        }
    }
    return Ok(Control::Next);
}


/* Functions for reading and writing supported types from raw pointers.
 * They all use unsafe features, of course.
 * The programmer must ensure that the pointers passed to these are suitable.
 * The functions `region_slice` and `region_slice_mut` do this: they will
 * give an out of bounds error if there's not enough room.
 */

unsafe fn read_bytes<'a>(memory: *const u8, size: usize) -> &'a [u8] {
    return std::slice::from_raw_parts(memory, size)
}

fn read_u8(memory: *const u8) -> u8 {
    unsafe {
        let bytes: [u8; 1] = [*memory];
        return u8::from_le_bytes(bytes);
    }
}

fn write_u8(memory: *mut u8, x: u8) {
    unsafe {
        let bytes: [u8; 1] = x.to_le_bytes();
        std::ptr::copy(&bytes as *const u8, memory, 1);
    }
}

fn read_u16(memory: *const u8) -> u16 {
    unsafe {
        let bytes: [u8; 2] = [*memory, *memory.add(1)];
        return u16::from_le_bytes(bytes);
    }
}

fn write_u16(memory: *mut u8, x: u16) {
    unsafe {
        let bytes: [u8; 2] = x.to_le_bytes();
        std::ptr::copy(&bytes as *const u8, memory, 2);
    }
}

fn read_u32(memory: *const u8) -> u32 {
    unsafe {
        let bytes: [u8; 4] = [
                *memory,
                *memory.add(1),
                *memory.add(2),
                *memory.add(3)
            ];
        return u32::from_le_bytes(bytes);
    }
}

fn write_u32(memory: *mut u8, x: u32) {
    unsafe {
        let bytes: [u8; 4] = x.to_le_bytes();
        std::ptr::copy(&bytes as *const u8, memory, 4);
    }
}

fn read_u64(memory: *const u8) -> u64 {
    unsafe {
        let bytes: [u8; 8] = [
                *memory,
                *memory.add(1),
                *memory.add(2),
                *memory.add(3),
                *memory.add(4),
                *memory.add(5),
                *memory.add(6),
                *memory.add(7)
            ];
        return u64::from_le_bytes(bytes);
    }
}

fn write_u64(memory: *mut u8, x: u64) {
    unsafe {
        let bytes: [u8; 8] = x.to_le_bytes();
        std::ptr::copy(&bytes as *const u8, memory, 8);
    }
}

fn read_i8(memory: *const u8) -> i8 {
    unsafe {
        let bytes: [u8; 1] = [*memory];
        return i8::from_le_bytes(bytes);
    }
}

fn write_i8(memory: *mut u8, x: i8) {
    unsafe {
        let bytes: [u8; 1] = x.to_le_bytes();
        std::ptr::copy(&bytes as *const u8, memory, 1);
    }
}

fn read_i16(memory: *const u8) -> i16 {
    unsafe {
        let bytes: [u8; 2] = [*memory, *memory.add(1)];
        return i16::from_le_bytes(bytes);
    }
}

fn write_i16(memory: *mut u8, x: i16) {
    unsafe {
        let bytes: [u8; 2] = x.to_le_bytes();
        std::ptr::copy(&bytes as *const u8, memory, 2);
    }
}

fn read_i32(memory: *const u8) -> i32 {
    unsafe {
        let bytes: [u8; 4] = [
                *memory,
                *memory.add(1),
                *memory.add(2),
                *memory.add(3)
            ];
        return i32::from_le_bytes(bytes);
    }
}

fn write_i32(memory: *mut u8, x: i32) {
    unsafe {
        let bytes: [u8; 4] = x.to_le_bytes();
        std::ptr::copy(&bytes as *const u8, memory, 4);
    }
}

fn read_i64(memory: *const u8) -> i64 {
    unsafe {
        let bytes: [u8; 8] = [
                *memory,
                *memory.add(1),
                *memory.add(2),
                *memory.add(3),
                *memory.add(4),
                *memory.add(5),
                *memory.add(6),
                *memory.add(7)
            ];
        return i64::from_le_bytes(bytes);
    }
}

fn write_i64(memory: *mut u8, x: i64) {
    unsafe {
        let bytes: [u8; 8] = x.to_le_bytes();
        std::ptr::copy(&bytes as *const u8, memory, 8);
    }
}

fn read_f32(memory: *const u8) -> f32 {
    unsafe {
        let bytes: [u8; 4] = [
                *memory,
                *memory.add(1),
                *memory.add(2),
                *memory.add(3)
            ];
        return f32::from_le_bytes(bytes);
    }
}

fn write_f32(memory: *mut u8, x: f32) {
    unsafe {
        let bytes: [u8; 4] = x.to_le_bytes();
        std::ptr::copy(&bytes as *const u8, memory, 4);
    }
}

fn read_f64(memory: *const u8) -> f64 {
    unsafe {
        let bytes: [u8; 8] = [
                *memory,
                *memory.add(1),
                *memory.add(2),
                *memory.add(3),
                *memory.add(4),
                *memory.add(5),
                *memory.add(6),
                *memory.add(7)
            ];
        return f64::from_le_bytes(bytes);
    }
}

fn write_f64(memory: *mut u8, x: f64) {
    unsafe {
        let bytes: [u8; 8] = x.to_le_bytes();
        std::ptr::copy(&bytes as *const u8, memory, 8);
    }
}

