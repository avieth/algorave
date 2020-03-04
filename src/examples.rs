extern crate algorave;
use algorave::lang::instruction::{*};
use algorave::lang::instruction::Instruction;
use algorave::lang::instruction::Constant;
use algorave::lang::instruction::FType;
use algorave::lang::instruction::Type;
use algorave::lang::instruction::Program;

pub fn program_1() -> Program {
    return vec![
        Instruction::Set(Constant::Rate, 0x00),
        // Previous one fills 8 bytes so that we can treat it like a u64
        // for the upcoming modulo op.
        Instruction::Set(Constant::Now, 0x08),
        // The remainder goes at 0x10 and we treat it like a 32 bit
        // unsigned integer because we know that Rate is at most 2^32.
        Instruction::Mod(
            Type::Integral(IType::TUnsigned(UIType::TU64)),
            0x08,
            0x00,
            0x10
        ),
        // Now we need to divide it by the rate. We approximate the rate
        // and the remainder as 32 bit floats. Before conversion, we treat them
        // as 32 bit signed integers, which is fine because they are positive
        // and almost certainly less than 2^31.
        Instruction::Itof(SIType::TI32, FType::TF32, 0x00, 0x14),
        Instruction::Itof(SIType::TI32, FType::TF32, 0x10, 0x18),
        Instruction::Div(
            Type::Fractional(FType::TF32),
            0x18,
            0x14,
            0x1C
        ),
        // Make it 440Hz
        Instruction::Set(Constant::F32(440.0*2.0*std::f32::consts::PI), 0x20),
        Instruction::Mul(
            Type::Fractional(FType::TF32),
            0x1C,
            0x20,
            0x24
        ),
        Instruction::Sin(FType::TF32, 0x24, 0x28),
        // Attenuate to amplitude of 0.5.
        Instruction::Set(Constant::F32(0.5), 0x2C),
        Instruction::Mul(
            Type::Fractional(FType::TF32),
            0x28,
            0x2C,
            0x30
        ),
        // 0x30 is now the value of the sine wave.

        // Here we do some branching to toggle the byte at 0x100.
        // If we get a MIDI event and it's 0, we set it to 1.
        // If we get a MIDI event and it's 1, we set it to 0.

        // Read into 0x50 the number of MIDI events for this frame (the first
        // byte of the MIDI input region at identifier 0x03.
        // To do so, we put the output identifier (u8) and an address in its
        // memory into registers (u32) 0x40 and 0x41
        Instruction::Set(Constant::U8(0x03), 0x40),
        Instruction::Set(Constant::U32(0x00000000), 0x41),
        Instruction::Read(
            Type::Integral(IType::TUnsigned(UIType::TU8)),
            0x40,
            0x41,
            0x50
        ),
        // If the number of MIDI events is 0, we branch down to skip this
        // step.
        //
        // This is how many instructions to jump.
        Instruction::Set(Constant::I32(0x03), 0x40),
        Instruction::Branch(0x40, 0x50),

        // If we didn't jump, we toggle the value at 0x100.
        Instruction::Set(Constant::U8(0x01), 0xA0),
        Instruction::Xor(
            UIType::TU8,
            0x100,
            0xA0,
            0x100
        ),

        /*
        // In this block, the number of MIDI events is non-zero. Now we
        // branch again: if 0x100 is 0 we set it to 1, otherwise we set it
        // to 0.
        Instruction::Set(Constant::U8(5), Address::MemoryAddress(0xA0)),
        Instruction::Branch(Address::MemoryAddress(0x100), Address::MemoryAddress(0xA0)),
        Instruction::Set(Constant::U8(0x00), Address::MemoryAddress(0x100)),
        // THe next 3 lines make us not reset it to 0x01.
        Instruction::Set(Constant::U8(0x00), Address::MemoryAddress(0xA1)),
        Instruction::Set(Constant::U8(2), Address::MemoryAddress(0xA0)),
        Instruction::Branch(Address::MemoryAddress(0xA1), Address::MemoryAddress(0xA0)),
        Instruction::Set(Constant::U8(0x01), Address::MemoryAddress(0x100)),
        */

        // If 0x100 is 0, do not write the output.
        // We reuse 0x40 as the location of the jump offset.
        Instruction::Set(Constant::I32(0x06), 0x40),
        Instruction::Branch(0x40, 0x100),
        Instruction::Set(Constant::U8(0x00), 0x40),
        Instruction::Set(Constant::U32(0x00000000), 0x41),
        Instruction::Write(
            Type::Fractional(FType::TF32),
            0x40,
            0x41,
            0x30
        ),
        Instruction::Set(Constant::U8(0x01), 0x40),
        Instruction::Write(
            Type::Fractional(FType::TF32),
            0x40,
            0x41,
            0x30
        ),
        Instruction::Stop
    ];
}

/// A MIDI keyboard program. Plays a sine wave at the frequency given by the
/// depressed keys. 16 voice polyphony.
pub fn program_2() -> Program {
    let array_address: u32 = 0x100;
    let array_length:  u8  = 0x10;
    return vec![

        // Get the current time in seconds, for use by a later call to sin.
        Instruction::Set(Constant::Rate, 0x00),
        // Previous one fills 8 bytes so that we can treat it like a u64
        // for the upcoming modulo op.
        Instruction::Set(Constant::Now, 0x08),
        // The remainder goes at 0x10 and we treat it like a 32 bit
        // unsigned integer because we know that Rate is at most 2^32.
        Instruction::Mod(
            Type::Integral(IType::TUnsigned(UIType::TU64)),
            0x08,
            0x00,
            0x10
        ),
        // Now we need to divide it by the rate. We approximate the rate
        // and the remainder as 32 bit floats. Before conversion, we treat them
        // as 32 bit signed integers, which is fine because they are positive
        // and almost certainly less than 2^31.
        Instruction::Itof(SIType::TI32, FType::TF32, 0x00, 0x14),
        Instruction::Itof(SIType::TI32, FType::TF32, 0x10, 0x18),
        Instruction::Div(
            Type::Fractional(FType::TF32),
            0x18,
            0x14,
            0xF8
        ),
        // 0xF8 is the value.

        // Left/right mixes live in 0xF0 and 0xF4.
        // We'll 0 them first. They are always written out at the end.
        Instruction::Set(Constant::F32(0.0), 0xF0),
        Instruction::Set(Constant::F32(0.0), 0xF4),

        // Test data: populate the array with known values.
        //
        // This is to increment a pointer by 1 (byte).
        Instruction::Set(Constant::U32(0x01), 0x04) ,
        // This is the pointer to the array memory.
        Instruction::Set(Constant::U32(array_address), 0x00),

        // TODO next step: do a for loop that populates the array according
        // to MIDI input. The general form?
        // - A pointer to a location from which we can Read in from the
        //   MIDI port.
        // - A pointer to a location in the MIDI buffer, which we can bump
        //   by the number of bytes.
        // -
        // 
        // ptr midi_id = &0x00;
        // ptr midi_in = 0x00000000;
        // ptr n_events;
        // read(u8, midi_id, midi_in, n_events);
        // midi_in += 1;
        // ptr size;
        // for (i = 0; i < n_events; i++) {
        //   read(u8, midi_id, midi_in, size);
        //   if (midi_in != 0) {
        //     // Check the status byte.
        //     // If it's note off, traverse the array and remove it.
        //     // If it's note on, traverse the array and add it if there is
        //     // room.
        //   }
        // }

        // note 64 is on (first bit is 1, the rest is 64 if first bit were 0).
        Instruction::Set(Constant::U8(0b11000000), 0x10),
        Instruction::Set(Constant::U32(0x10), 0x08),
        Instruction::Copy(Type::Integral(IType::TUnsigned(UIType::TU8)), 0x08, 0x00),
        Instruction::Add(Type::Integral(IType::TUnsigned(UIType::TU32)), 0x00, 0x04, 0x00),
        // Note 71 is on
        Instruction::Set(Constant::U16(0b11000111), 0x10),
        Instruction::Set(Constant::U32(0x10), 0x08),
        Instruction::Copy(Type::Integral(IType::TUnsigned(UIType::TU16)), 0x08, 0x00),
        // And note 68 is on, giving a major triad.
        Instruction::Set(Constant::U16(0b11000100), 0x10),
        Instruction::Set(Constant::U32(0x10), 0x08),
        Instruction::Copy(Type::Integral(IType::TUnsigned(UIType::TU16)), 0x08, 0x00),

        // Set the array address back to original. It shall be used and
        // mutated by the loop
        Instruction::Set(Constant::U32(array_address), 0x00),

        // This is the playback loop.
        // It runs through the 16 elements in the array, computes their
        // sine waves (not relative to when the key was pressed) and
        // adds them.
        //
        // 0x08 is our index location.
        Instruction::Set(Constant::U8(0x00), 0x08),
        // We'll need a 1 for incrementing.
        Instruction::Set(Constant::U8(0x01), 0x09),
        // Array length (16 voices).
        Instruction::Set(Constant::U8(array_length), 0x0A),

        // This is the check to exit the loop.
        // If the index (0x00) is less than the length (at 0x02) then we
        // don't branch.
        Instruction::Lt(Type::Integral(IType::TUnsigned(UIType::TU8)), 0x08, 0x0A, 0x0B),
        Instruction::Set(Constant::I32(30), 0xFC),
        Instruction::Branch(0xFC, 0x0B),

        // Loop body: check the array at the given index.
        // We must use a copy instruction to get the next data, since we're
        // dealing with a pointer (at 0x00).
        // NB: the second argument is also a pointer.
        Instruction::Set(Constant::U32(0x0C), 0x10),
        Instruction::Copy(Type::Integral(IType::TUnsigned(UIType::TU8)), 0x00, 0x10),
        // If the first bit is 0, branch so that we don't add the voice.
        Instruction::Set(Constant::U8(0x07), 0x0D),
        Instruction::Shiftr(UIType::TU8, 0x0C, 0x0D, 0x0D),
        // Must skip ahead 12, to land on the instruction which bumps
        // the index and pointer.
        Instruction::Set(Constant::I32(20), 0xFC),
        Instruction::Branch(0xFC, 0x0D),

        //Instruction::Itof(SIType::TI8, FType::TF32, 0x0D, 0x10),
        // 0x10 must contain the desired frequency.
        // To get it:
        // - x := AND with 0b01111111
        // - convert it to a 32-bit float.
        // - subtract 64.0
        // - 2 ^ ( (12*ln(440) + n) / 12) )
        // So we need to be able to do powers of 2... Why not give a
        // Pow instruction for float?
        Instruction::Set(Constant::U8(0b01111111), 0x0D),
        Instruction::And(UIType::TU8, 0x0C, 0x0D, 0x0C),
        Instruction::Itof(SIType::TI8, FType::TF32, 0x0C, 0x10),
        Instruction::Set(Constant::F32(64.0), 0x14),
        Instruction::Sub(Type::Fractional(FType::TF32), 0x10, 0x14, 0x10),
        Instruction::Set(Constant::F32((12.0f32)*((440.0f32).log(2.0))), 0x14),
        Instruction::Add(Type::Fractional(FType::TF32), 0x10, 0x14, 0x10),
        Instruction::Set(Constant::F32(12.0), 0x14),
        Instruction::Div(Type::Fractional(FType::TF32), 0x10, 0x14, 0x10),
        Instruction::Set(Constant::F32(2.0), 0x14),
        Instruction::Pow(FType::TF32, 0x14, 0x10, 0x10),

        Instruction::Set(Constant::F32(2.0*std::f32::consts::PI), 0x14),
        Instruction::Mul(Type::Fractional(FType::TF32), 0x10, 0xF8, 0x10),
        Instruction::Mul(Type::Fractional(FType::TF32), 0x10, 0x14, 0x10),
        Instruction::Sin(FType::TF32, 0x10, 0x10),
        // TODO get the amplitude from the MIDI key velocity.
        Instruction::Set(Constant::F32(0.1), 0x14),
        Instruction::Mul(Type::Fractional(FType::TF32), 0x10, 0x14, 0x10),
        Instruction::Add(Type::Fractional(FType::TF32), 0xF0, 0x10, 0xF0),
        Instruction::Add(Type::Fractional(FType::TF32), 0xF4, 0x10, 0xF4),

        // Bump the array index pointer (by 1, since each element is 8 bits).
        // We're re-using the 1 found at 0x04 from the beginning.
        Instruction::Add(Type::Integral(IType::TUnsigned(UIType::TU32)), 0x00, 0x04, 0x00),
        // Increment the loop index (0x08) by 1 (at 0x09).
        Instruction::Add(Type::Integral(IType::TUnsigned(UIType::TU8)), 0x08, 0x09, 0x08),
        // Loop over: jump back to the start. -20
        Instruction::Set(Constant::I32(-31), 0xFC),
        Instruction::Jump(0xFC),

        // Write from the 0xF0 and 0xF4 locations.
        Instruction::Set(Constant::U8(0x00), 0x14),
        Instruction::Set(Constant::U32(0x00000000), 0x15),
        Instruction::Write(
            Type::Fractional(FType::TF32),
            0x14,
            0x15,
            0xF0
        ),
        Instruction::Set(Constant::U8(0x01), 0x14),
        Instruction::Write(
            Type::Fractional(FType::TF32),
            0x14,
            0x15,
            0xF4
        ),

        Instruction::Stop
    ];
}

/// This program does a bunch of 64-iteration for loops.
/// Shows how quickly performance becomes problematic.
/// But is there really anything we can do? Fact of the matter is, the program
/// _must_ be short and fast enough that it can run to completion 48000 times
/// each second. Looping constructs are obviously problematic, but will they
/// even come up in real programs?
///
/// One case in which loops will be necessary is in reading and writing MIDI
/// data. But in practice these loops will be quite short (< 10).
///
/// Another example of a useful loop is in dealing with polyphony. In the
/// program_2 example we have a 16-element array for 16-voice polyphony, and
/// on the old Athlon 64 X2 we saw ~20% CPU usage just to loop through that
/// array. THAT would seem to be a problem. It ought to be possible to do
/// a 32 voice synth surely...
///
/// I believe signal processing will typically not involve a loop, but rather
/// recording some start point (frame) and computing using the difference
/// between this and the current sample. ADSR, for instance, would I suppose
/// branch based upon comparison of the time elapsed and the ADSR parameters.
///
/// Roughly how long should we expect each instruction in this object language
/// to require? Can we expect it to usually cache hit, since it works on one
/// contiguous piece of memory?
/// Let's suppose everything were in main memory, and it takes 100ns to get
/// something from main memory. At 44.1Khz sample rate we've got roughly
/// 23ms per frame, i.e. 23,000,000 nanoseconds, or the capacity for
/// 230,000 memory accesses.
/// Our experiment is 64 loop iterations of 4 instructions each, or 256
/// instructions. That should be negligible!
/// One thing you may want to try is to make a very simple C program which is
/// a JACK client and does some looping in the process callback. 
///
pub fn program_3() -> Program {
    return vec![
        Instruction::Set(Constant::U32(0x00), 0x00),
        Instruction::Set(Constant::U32(32), 0x04),
        Instruction::Lt(Type::Integral(IType::TUnsigned(UIType::TU32)), 0x00, 0x04, 0x08),
        Instruction::Set(Constant::I32(5), 0xFC),
        Instruction::Branch(0xFC, 0x08),
        Instruction::Set(Constant::U32(0x01), 0x08),
        Instruction::Add(Type::Integral(IType::TUnsigned(UIType::TU32)), 0x00, 0x08, 0x00),
        Instruction::Set(Constant::I32(-6), 0xFC),
        Instruction::Jump(0xFC),
        // Size of thing to trace
        Instruction::Set(Constant::U32(8), 0x00),
        // Address of thing to trace
        Instruction::Set(Constant::U32(0x08), 0x04),
        Instruction::Set(Constant::Now, 0x08),
        Instruction::Trace(0x00, 0x04),
        Instruction::Stop,
    ];
}
