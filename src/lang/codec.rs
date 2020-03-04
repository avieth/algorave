/// Definition of a codec for Instruction.
///
/// - Must be a streaming decoder. The rust program is going to pull in data
///   from stdin and parse it incrementally.
/// - Text or binary? Why not do a binary format to begin with? Simple enough:
///   One byte to indicate the instruction.
///   Then the fields in-order.

use super::instruction::Address;
use super::instruction::Instruction;
use super::instruction::Constant;
use super::instruction::Type;
use super::instruction::UIType;
use super::instruction::SIType;
use super::instruction::IType;
use super::instruction::FType;
use super::instruction;
use super::update::MemoryUpdate;
use super::update::Update;

pub trait Source {
    type Token;
    type Error;
    fn get(&mut self) -> Result<Self::Token, Self::Error>;
}

/*pub trait Sink {
    type Token;
    type Error;
    fn put(&mut self, token: &Self::Token) -> Result<(), Self::Error>;
}*/

#[derive(Debug)]
pub enum EncodeError<NoWrite> {
    NoWrite(NoWrite)
}

#[derive(Debug)]
pub enum DecodeError<NoParse, NoRead> {
    NoParse(NoParse),
    NoRead(NoRead)
}

// Apparently this one is not needed in order for our use of ? to work out...
// cool, because we can't have both of these From traits.
/*
impl<NoParse, NoRead> From<NoParse> for DecodeError<NoParse, NoRead> {
    fn from(no_parse: NoParse) -> DecodeError<NoParse, NoRead> {
        return DecodeError::NoParse(no_parse);
    }
}
*/
impl<NoParse, NoRead> From<NoRead> for DecodeError<NoParse, NoRead> {
    fn from(no_read: NoRead) -> DecodeError<NoParse, NoRead> {
        return DecodeError::NoRead(no_read);
    }
}

#[derive(Debug)]
pub enum InstructionDecodeError {
    UnknownInstructionCode(u8),
    UnknownConstantCode(u8),
    UnknownTypeCode(u8)
}

// Integers and floats are big-endian in this codec.
// They're little-endian in the executable state.

pub fn decode_u16<E, S>(src: &mut S) -> Result<u16, DecodeError<E, S::Error>>
    where S: Source<Token = u8> {
    let a = src.get()?;
    let b = src.get()?;
    return Ok(u16::from_be_bytes([a, b]));
}

pub fn decode_u32<E, S>(src: &mut S) -> Result<u32, DecodeError<E, S::Error>>
    where S: Source<Token = u8> {
    let a = src.get()?;
    let b = src.get()?;
    let c = src.get()?;
    let d = src.get()?;
    return Ok(u32::from_be_bytes([a, b, c, d]));
}

pub fn decode_u64<E, S>(src: &mut S) -> Result<u64, DecodeError<E, S::Error>>
    where S: Source<Token = u8> {
    let a = src.get()?;
    let b = src.get()?;
    let c = src.get()?;
    let d = src.get()?;
    let e = src.get()?;
    let f = src.get()?;
    let g = src.get()?;
    let h = src.get()?;
    return Ok(u64::from_be_bytes([a, b, c, d, e, f, g, h]));
}

pub fn decode_i16<E, S>(src: &mut S) -> Result<i16, DecodeError<E, S::Error>>
    where S: Source<Token = u8> {
    let a = src.get()?;
    let b = src.get()?;
    return Ok(i16::from_be_bytes([a, b]));
}

pub fn decode_i32<E, S>(src: &mut S) -> Result<i32, DecodeError<E, S::Error>>
    where S: Source<Token = u8> {
    let a = src.get()?;
    let b = src.get()?;
    let c = src.get()?;
    let d = src.get()?;
    return Ok(i32::from_be_bytes([a, b, c, d]));
}

pub fn decode_i64<E, S>(src: &mut S) -> Result<i64, DecodeError<E, S::Error>>
    where S: Source<Token = u8> {
    let a = src.get()?;
    let b = src.get()?;
    let c = src.get()?;
    let d = src.get()?;
    let e = src.get()?;
    let f = src.get()?;
    let g = src.get()?;
    let h = src.get()?;
    return Ok(i64::from_be_bytes([a, b, c, d, e, f, g, h]));
}

pub fn decode_f32<E, S>(src: &mut S) -> Result<f32, DecodeError<E, S::Error>>
    where S: Source<Token = u8> {
    let a = src.get()?;
    let b = src.get()?;
    let c = src.get()?;
    let d = src.get()?;
    return Ok(f32::from_be_bytes([a, b, c, d]));
}

pub fn decode_f64<E, S>(src: &mut S) -> Result<f64, DecodeError<E, S::Error>>
    where S: Source<Token = u8> {
    let a = src.get()?;
    let b = src.get()?;
    let c = src.get()?;
    let d = src.get()?;
    let e = src.get()?;
    let f = src.get()?;
    let g = src.get()?;
    let h = src.get()?;
    return Ok(f64::from_be_bytes([a, b, c, d, e, f, g, h]));
}

pub fn decode_constant<S>(src: &mut S) -> Result<Constant, DecodeError<InstructionDecodeError, S::Error>>
    where S: Source<Token = u8> {
    let byte = src.get()?;
    match byte {
        0x00 => {
            let x = src.get()?;
            return Ok(Constant::U8(x));
        }
        0x01 => {
            let x = decode_u16(src)?;
            return Ok(Constant::U16(x));
        }
        0x02 => {
            let x = decode_u32(src)?;
            return Ok(Constant::U32(x));
        }
        0x03 => {
            let x = decode_u64(src)?;
            return Ok(Constant::U64(x));
        }
        0x10 => {
            let x = src.get()?;
            return Ok(Constant::I8(x as i8));
        }
        0x11 => {
            let x = decode_i16(src)?;
            return Ok(Constant::I16(x));
        }
        0x12 => {
            let x = decode_i32(src)?;
            return Ok(Constant::I32(x));
        }
        0x13 => {
            let x = decode_i64(src)?;
            return Ok(Constant::I64(x));
        }
        0x20 => {
            let x = decode_f32(src)?;
            return Ok(Constant::F32(x));
        }
        0x21 => {
            let x = decode_f64(src)?;
            return Ok(Constant::F64(x));
        }
        0xF0 => {
            return Ok(Constant::Now);
        }
        0xF1 => {
            return Ok(Constant::Start);
        }
        0xF2 => {
            return Ok(Constant::Rate);
        }

        _ => {
            return Err(DecodeError::NoParse(InstructionDecodeError::UnknownConstantCode(byte)));
        }
    }
}

// FIXME possible in rust to alias a function declaration?
//
//   decode_address = decode_u32
//
pub fn decode_address<S>(src: &mut S) -> Result<Address, DecodeError<InstructionDecodeError, S::Error>>
    where S: Source<Token = u8> {
    return decode_u32(src);
}

pub fn decode_itype<S>(src: &mut S) -> Result<IType, DecodeError<InstructionDecodeError, S::Error>>
    where S: Source<Token = u8> {
    let byte = src.get()?;
    match byte {
        0x00 => { Ok(IType::TUnsigned(UIType::TU8)) }
        0x01 => { Ok(IType::TUnsigned(UIType::TU16)) }
        0x02 => { Ok(IType::TUnsigned(UIType::TU32)) }
        0x03 => { Ok(IType::TUnsigned(UIType::TU64)) }
        0x10 => { Ok(IType::TSigned(SIType::TI8)) }
        0x11 => { Ok(IType::TSigned(SIType::TI16)) }
        0x12 => { Ok(IType::TSigned(SIType::TI32)) }
        0x13 => { Ok(IType::TSigned(SIType::TI64)) }
        _ =>    { Err(DecodeError::NoParse(InstructionDecodeError::UnknownTypeCode(byte))) }
    }
}

pub fn decode_uitype<S>(src: &mut S) -> Result<UIType, DecodeError<InstructionDecodeError, S::Error>>
    where S: Source<Token = u8> {
    let byte = src.get()?;
    match byte {
        0x00 => { Ok(UIType::TU8) }
        0x01 => { Ok(UIType::TU16) }
        0x02 => { Ok(UIType::TU32) }
        0x03 => { Ok(UIType::TU64) }
        _ =>    { Err(DecodeError::NoParse(InstructionDecodeError::UnknownTypeCode(byte))) }
    }
}

pub fn decode_sitype<S>(src: &mut S) -> Result<SIType, DecodeError<InstructionDecodeError, S::Error>>
    where S: Source<Token = u8> {
    let byte = src.get()?;
    match byte {
        0x10 => { Ok(SIType::TI8) }
        0x11 => { Ok(SIType::TI16) }
        0x12 => { Ok(SIType::TI32) }
        0x13 => { Ok(SIType::TI64) }
        _ =>    { Err(DecodeError::NoParse(InstructionDecodeError::UnknownTypeCode(byte))) }
    }
}

pub fn decode_ftype<S>(src: &mut S) -> Result<FType, DecodeError<InstructionDecodeError, S::Error>>
    where S: Source<Token = u8> {
    let byte = src.get()?;
    match byte {
        0x20 => { Ok(FType::TF32) }
        0x21 => { Ok(FType::TF64) }
        _ =>    { Err(DecodeError::NoParse(InstructionDecodeError::UnknownTypeCode(byte))) }
    }
}

pub fn decode_type<S>(src: &mut S) -> Result<Type, DecodeError<InstructionDecodeError, S::Error>>
    where S: Source<Token = u8> {
    let byte = src.get()?;
    // Codes are the same as their corresponding literals.
    // The first 4 bits determine the kind of number.
    match byte {
        0x00 => { Ok(Type::Integral(IType::TUnsigned(UIType::TU8))) }
        0x01 => { Ok(Type::Integral(IType::TUnsigned(UIType::TU16))) }
        0x02 => { Ok(Type::Integral(IType::TUnsigned(UIType::TU32))) }
        0x03 => { Ok(Type::Integral(IType::TUnsigned(UIType::TU64))) }
        0x10 => { Ok(Type::Integral(IType::TSigned(SIType::TI8))) }
        0x11 => { Ok(Type::Integral(IType::TSigned(SIType::TI16))) }
        0x12 => { Ok(Type::Integral(IType::TSigned(SIType::TI32))) }
        0x13 => { Ok(Type::Integral(IType::TSigned(SIType::TI64))) }
        0x20 => { Ok(Type::Fractional(FType::TF32)) }
        0x21 => { Ok(Type::Fractional(FType::TF64)) }
        _ => { Err(DecodeError::NoParse(InstructionDecodeError::UnknownTypeCode(byte))) }
    }
}

// It's highly desirable that we use the Result type, because then we get the
// magical ? operator which is IMO 100% necessary in order for this language
// to be worthwhile for writing parsers compositionally.
// So we put the "failed to parse" cases in the Ok variant, leaving only the
// "failed to get input" cases in the Err variant.
pub fn decode_instruction<S>(src: &mut S) -> Result<Instruction, DecodeError<InstructionDecodeError, S::Error>>
    where S: Source<Token = u8> {
    let byte = src.get()?;
    match byte {

        /* Control instructions are 0x */
        0x00 => {
            return Ok(Instruction::Stop);
        }
        0x01 => {
            let jmp = decode_address(src)?;
            return Ok(Instruction::Jump(jmp));
        }
        0x02 => {
            let chk = decode_address(src)?;
            let jmp = decode_address(src)?;
            return Ok(Instruction::Branch(chk, jmp));
        }

        /* Move instructions are 1x */
        0x10 => {
            let ty = decode_type(src)?;
            let a = decode_address(src)?;
            let b = decode_address(src)?;
            return Ok(Instruction::Copy(ty, a, b));
        }
        0x11 => {
            let a = decode_constant(src)?;
            let b = decode_address(src)?;
            return Ok(Instruction::Set(a, b));
        }
        0x12 => {
            let ty = decode_type(src)?;
            let a = decode_address(src)?;
            let b = decode_address(src)?;
            let c = decode_address(src)?;
            return Ok(Instruction::Read(ty, a, b, c));
        }
        0x13 => {
            let ty = decode_type(src)?;
            let a = decode_address(src)?;
            let b = decode_address(src)?;
            let c = decode_address(src)?;
            return Ok(Instruction::Write(ty, a, b, c));
        }

        /* Bitwise instructions are 2x */
        0x20 => {
            let uity  = decode_uitype(src)?;
            let sval1 = decode_address(src)?;
            let sval2 = decode_address(src)?;
            let dval  = decode_address(src)?;
            return Ok(Instruction::Or(uity, sval1, sval2, dval));
        }
        0x21 => {
            let uity  = decode_uitype(src)?;
            let sval1 = decode_address(src)?;
            let sval2 = decode_address(src)?;
            let dval  = decode_address(src)?;
            return Ok(Instruction::And(uity, sval1, sval2, dval));
        }
        0x22 => {
            let uity = decode_uitype(src)?;
            let sval1 = decode_address(src)?;
            let sval2 = decode_address(src)?;
            let dval  = decode_address(src)?;
            return Ok(Instruction::Xor(uity, sval1, sval2, dval));
        }
        0x23 => {
            let uity = decode_uitype(src)?;
            let sval = decode_address(src)?;
            let dval = decode_address(src)?;
            return Ok(Instruction::Not(uity, sval, dval));
        }
        0x24 => {
            let uity = decode_uitype(src)?;
            let sval1 = decode_address(src)?;
            let sval2 = decode_address(src)?;
            let dval = decode_address(src)?;
            return Ok(Instruction::Shiftl(uity, sval1, sval2, dval));
        }
        0x25 => {
            let uity = decode_uitype(src)?;
            let sval1 = decode_address(src)?;
            let sval2 = decode_address(src)?;
            let dval = decode_address(src)?;
            return Ok(Instruction::Shiftr(uity, sval1, sval2, dval));
        }

        /* Arithmetic and trig is 3x */
        0x30 => {
            let ty = decode_type(src)?;
            let sval1 = decode_address(src)?;
            let sval2 = decode_address(src)?;
            let dval = decode_address(src)?;
            return Ok(Instruction::Add(ty, sval1, sval2, dval));
        }
        0x31 => {
            let ty = decode_type(src)?;
            let sval1 = decode_address(src)?;
            let sval2 = decode_address(src)?;
            let dval = decode_address(src)?;
            return Ok(Instruction::Sub(ty, sval1, sval2, dval));
        }
        0x32 => {
            let ty = decode_type(src)?;
            let sval1 = decode_address(src)?;
            let sval2 = decode_address(src)?;
            let dval = decode_address(src)?;
            return Ok(Instruction::Mul(ty, sval1, sval2, dval));
        }
        0x33 => {
            let ty = decode_type(src)?;
            let sval1 = decode_address(src)?;
            let sval2 = decode_address(src)?;
            let dval = decode_address(src)?;
            return Ok(Instruction::Div(ty, sval1, sval2, dval));
        }
        0x34 => {
            let ty = decode_type(src)?;
            let sval1 = decode_address(src)?;
            let sval2 = decode_address(src)?;
            let dval = decode_address(src)?;
            return Ok(Instruction::Mod(ty, sval1, sval2, dval));
        }
        0x35 => {
            let sity = decode_sitype(src)?;
            let sval = decode_address(src)?;
            let dval = decode_address(src)?;
            return Ok(Instruction::Abs(sity, sval, dval));
        }
        0x36 => {
            let fty = decode_ftype(src)?;
            let sval = decode_address(src)?;
            let dval = decode_address(src)?;
            return Ok(Instruction::Absf(fty, sval, dval));
        }
        0x37 => {
            let fty = decode_ftype(src)?;
            let sval1 = decode_address(src)?;
            let sval2 = decode_address(src)?;
            let dval = decode_address(src)?;
            return Ok(Instruction::Pow(fty, sval1, sval2, dval));
        }
        0x38 => {
            let fty = decode_ftype(src)?;
            let sval1 = decode_address(src)?;
            let sval2 = decode_address(src)?;
            let dval = decode_address(src)?;
            return Ok(Instruction::Log(fty, sval1, sval2, dval));
        }
        0x39 => {
            let fty = decode_ftype(src)?;
            let sval = decode_address(src)?;
            let dval = decode_address(src)?;
            return Ok(Instruction::Sin(fty, sval, dval));
        }
        0x3A => {
            let fty = decode_ftype(src)?;
            let sval = decode_address(src)?;
            let dval = decode_address(src)?;
            return Ok(Instruction::Exp(fty, sval, dval));
        }
        0x3B => {
            let fty = decode_ftype(src)?;
            let sval = decode_address(src)?;
            let dval = decode_address(src)?;
            return Ok(Instruction::Ln(fty, sval, dval));
        }

        /* Comparison is 4x */

        0x40 => {
            let ty = decode_type(src)?;
            let sval1 = decode_address(src)?;
            let sval2 = decode_address(src)?;
            let dval = decode_address(src)?;
            return Ok(Instruction::Eq(ty, sval1, sval2, dval));
        }
        0x41 => {
            let ty = decode_type(src)?;
            let sval1 = decode_address(src)?;
            let sval2 = decode_address(src)?;
            let dval = decode_address(src)?;
            return Ok(Instruction::Lt(ty, sval1, sval2, dval));
        }
        0x42 => {
            let ty = decode_type(src)?;
            let sval1 = decode_address(src)?;
            let sval2 = decode_address(src)?;
            let dval = decode_address(src)?;
            return Ok(Instruction::Gt(ty, sval1, sval2, dval));
        }

        /* Casts are 5x */

        0x50 => {
            let fty = decode_ftype(src)?;
            let sval = decode_address(src)?;
            let dval = decode_address(src)?;
            return Ok(Instruction::Castf(fty, sval, dval));
        }
        0x51 => {
            let fty = decode_ftype(src)?;
            let sity = decode_sitype(src)?;
            let sval = decode_address(src)?;
            let dval = decode_address(src)?;
            return Ok(Instruction::Ftoi(fty, sity, sval, dval));
        }
        0x52 => {
            let sity = decode_sitype(src)?;
            let fty = decode_ftype(src)?;
            let sval = decode_address(src)?;
            let dval = decode_address(src)?;
            return Ok(Instruction::Itof(sity, fty, sval, dval));
        }

        0xFF => {
            let size_addr = decode_address(src)?;
            let addr = decode_address(src)?;
            return Ok(Instruction::Trace(size_addr, addr));
        }

        _ => {
            return Err(DecodeError::NoParse(InstructionDecodeError::UnknownInstructionCode(byte)));
        }
    }
}

#[derive(Debug)]
pub enum ProgramDecodeError {
    InstructionDecodeError(InstructionDecodeError)
}

impl From<InstructionDecodeError> for ProgramDecodeError {
    fn from(err: InstructionDecodeError) -> ProgramDecodeError {
        return ProgramDecodeError::InstructionDecodeError(err);
    }
}

/// 32-bit big-endian length followed by that many instructions.
pub fn decode_program<S>(src: &mut S) -> Result<instruction::Program, DecodeError<ProgramDecodeError, S::Error>>
    where S: Source<Token = u8> {
    let mut n_instructions = decode_u32(src)?;
    let mut program = Vec::with_capacity(n_instructions as usize);
    while n_instructions > 0 {
        // Can't figure out how to do a polymorphic From trait over DecodeError.
        // It's probably not possible *sigh* no question mark operator here :(
        let instr = decode_instruction(src);
        match instr {
            Ok(x) => {
                program.push(x);
                n_instructions -= 1;
            }
            Err(DecodeError::NoParse(err)) => {
                return Err(DecodeError::NoParse(ProgramDecodeError::InstructionDecodeError(err)));
            }
            Err(DecodeError::NoRead(err)) => {
                return Err(DecodeError::NoRead(err));
            }
        }
    }
    return Ok(program);
}

#[derive(Debug)]
pub enum MemoryUpdateDecodeError {}

/// A memory update is a 32-bit offset followed by a 32-bit size of the data
/// followed by that many bytes.
pub fn decode_memory_update<S>(src: &mut S) -> Result<MemoryUpdate, DecodeError<MemoryUpdateDecodeError, S::Error>>
    where S: Source<Token = u8> {
    let offset = decode_u32(src)?;
    let mut n_bytes = decode_u32(src)?;
    let mut data = Vec::with_capacity(n_bytes as usize);
    while n_bytes > 0 {
        let byte = src.get()?;
        data.push(byte);
        n_bytes -= 1;
    }
    return Ok(MemoryUpdate { offset: offset, data: data });
}

#[derive(Debug)]
pub enum UpdateDecodeError {
    ProgramDecodeError(ProgramDecodeError),
    MemoryUpdateDecodeError(MemoryUpdateDecodeError)
}

/// Program update followed by memory updates.
/// Give a valid encoding of a program. If it's the 0-length program, then
/// the program in the Update will be None.
/// Then give a length-encoded vector of memory updates.
pub fn decode_update<S>(src: &mut S) -> Result<Update, DecodeError<UpdateDecodeError, S::Error>>
    where S: Source<Token = u8> {
    let decoded_program = decode_program(src);
    let program = match decoded_program {
        Ok(program) => {
            if program.len() == 0 { None } else { Some(program) }
        }
        Err(DecodeError::NoParse(err)) => {
            return Err(DecodeError::NoParse(UpdateDecodeError::ProgramDecodeError(err)));
        }
        Err(DecodeError::NoRead(err)) => {
            return Err(DecodeError::NoRead(err));
        }
    };
    let mut n_mem_updates = decode_u32(src)?;
    let mut mem_updates = Vec::with_capacity(n_mem_updates as usize);
    while n_mem_updates > 0 {
        // Can't figure out how to do a polymorphic From trait over DecodeError.
        // It's probably not possible *sigh* no question mark operator here :(
        let update = decode_memory_update(src);
        match update {
            Ok(x) => {
                mem_updates.push(x);
                n_mem_updates -= 1;
            }
            Err(DecodeError::NoParse(err)) => {
                return Err(DecodeError::NoParse(UpdateDecodeError::MemoryUpdateDecodeError(err)));
            }
            Err(DecodeError::NoRead(err)) => {
                return Err(DecodeError::NoRead(err));
            }
        }
    }
    return Ok(Update { program: program, memory: mem_updates });
}
