{-|
Module      : Algorave.Language.Codec
Description : Codec for communication with the rust executable
Copyright   : (c) Alexander Vieth, 2020
Licence     : BSD3
Maintainer  : aovieth@gmail.com
Stability   : experimental
Portability : non-portable (GHC only)
-}

module Algorave.Language.Codec where

import Data.ByteString.Builder (Builder)
import qualified Data.ByteString.Builder as B

import Algorave.Language.Instruction
import Algorave.Language.Program
import Algorave.Language.Update

encode_program :: Program -> Builder
encode_program prog =
  B.word32BE (program_length prog) <> mconcat (fmap encode_instruction (program_instructions prog))

encode_memory_update :: MemoryUpdate -> Builder
encode_memory_update mupdate =
  B.word32BE (mem_update_offset mupdate) <> B.lazyByteString (mem_update_bytes mupdate)

encode_update :: Update -> Builder
encode_update update = encoded_program <> encoded_mem_updates
  where
  encoded_program = case update_program update of
    Nothing -> B.word32BE 0x00
    Just prog -> encode_program prog
  encoded_mem_updates =
       B.word32BE (fromIntegral (length mem_updates))
    <> mconcat (fmap encode_memory_update mem_updates)
  mem_updates = update_mem_updates update

encode_address :: Address -> Builder
encode_address = B.word32BE

encode_type :: Type -> Builder
encode_type ty = case ty of

  Integral (Unsigned TU8)  -> B.word8 0x00
  Integral (Unsigned TU16) -> B.word8 0x01
  Integral (Unsigned TU32) -> B.word8 0x02
  Integral (Unsigned TU64) -> B.word8 0x03

  Integral (Signed TI8)  -> B.word8 0x10
  Integral (Signed TI16) -> B.word8 0x11
  Integral (Signed TI32) -> B.word8 0x12
  Integral (Signed TI64) -> B.word8 0x13

  Fractional TF32 -> B.word8 0x20
  Fractional TF64 -> B.word8 0x21

encode_ftype :: FType -> Builder
encode_ftype ftype = case ftype of
  TF32 -> B.word8 0x20
  TF64 -> B.word8 0x21

encode_itype :: IType -> Builder
encode_itype itype = case itype of

  Unsigned TU8  -> B.word8 0x00
  Unsigned TU16 -> B.word8 0x01
  Unsigned TU32 -> B.word8 0x02
  Unsigned TU64 -> B.word8 0x03

  Signed TI8  -> B.word8 0x10
  Signed TI16 -> B.word8 0x11
  Signed TI32 -> B.word8 0x12
  Signed TI64 -> B.word8 0x13

encode_uitype :: UIType -> Builder
encode_uitype uitype = case uitype of
  TU8  -> B.word8 0x00
  TU16 -> B.word8 0x01
  TU32 -> B.word8 0x02
  TU64 -> B.word8 0x03

encode_sitype :: SIType -> Builder
encode_sitype sitype = case sitype of
  TI8  -> B.word8 0x10
  TI16 -> B.word8 0x11
  TI32 -> B.word8 0x12
  TI64 -> B.word8 0x13

encode_constant :: Constant -> Builder
encode_constant lit = case lit of

  U8  x -> B.word8 0x00 <> B.word8 x
  U16 x -> B.word8 0x01 <> B.word16BE x
  U32 x -> B.word8 0x02 <> B.word32BE x
  U64 x -> B.word8 0x03 <> B.word64BE x

  I8  x -> B.word8 0x10 <> B.int8 x
  I16 x -> B.word8 0x11 <> B.int16BE x
  I32 x -> B.word8 0x12 <> B.int32BE x
  I64 x -> B.word8 0x13 <> B.int64BE x

  F32 x -> B.word8 0x20 <> B.floatBE x
  F64 x -> B.word8 0x21 <> B.doubleBE x

  Now   -> B.word8 0xF0
  Start -> B.word8 0xF1
  Rate  -> B.word8 0xF2

encode_instruction :: Instruction -> Builder
encode_instruction inst = case inst of
  
  Stop -> B.word8 0x00

  Jump jmp -> B.word8 0x01 <> encode_address jmp
  Branch chk jmp -> B.word8 0x02 <> encode_address chk <> encode_address jmp

  Copy ty src dst -> B.word8 0x10 <> encode_type ty <> encode_address src <> encode_address dst
  Set  const dst  -> B.word8 0x11 <> encode_constant const <> encode_address dst
  Read  ty a b c -> B.word8 0x12 <> encode_type ty <> encode_address a <> encode_address b <> encode_address c
  Write ty a b c -> B.word8 0x13 <> encode_type ty <> encode_address a <> encode_address b <> encode_address c

  Or typ src1 src2 dst     -> B.word8 0x20 <> encode_uitype typ <> encode_address src1 <> encode_address src2 <> encode_address dst
  And typ src1 src2 dst    -> B.word8 0x21 <> encode_uitype typ <> encode_address src1 <> encode_address src2 <> encode_address dst
  Xor typ src1 src2 dst    -> B.word8 0x22 <> encode_uitype typ <> encode_address src1 <> encode_address src2 <> encode_address dst
  Not typ src dst          -> B.word8 0x23 <> encode_uitype typ <> encode_address src  <> encode_address dst
  Shiftl typ src1 src2 dst -> B.word8 0x24 <> encode_uitype typ <> encode_address src1 <> encode_address src2 <> encode_address dst
  Shiftr typ src1 src2 dst -> B.word8 0x25 <> encode_uitype typ <> encode_address src1 <> encode_address src2 <> encode_address dst

  Add typ src1 src2 dst -> B.word8 0x30 <> encode_type typ <> encode_address src1 <> encode_address src2 <> encode_address dst
  Sub typ src1 src2 dst -> B.word8 0x31 <> encode_type typ <> encode_address src1 <> encode_address src2 <> encode_address dst
  Mul typ src1 src2 dst -> B.word8 0x32 <> encode_type typ <> encode_address src1 <> encode_address src2 <> encode_address dst
  Div typ src1 src2 dst -> B.word8 0x33 <> encode_type typ <> encode_address src1 <> encode_address src2 <> encode_address dst
  Mod typ src1 src2 dst -> B.word8 0x34 <> encode_type typ <> encode_address src1 <> encode_address src2 <> encode_address dst

  Abs  sity src dst -> B.word8 0x35 <> encode_sitype sity <> encode_address src <> encode_address dst
  Absf fty  src dst -> B.word8 0x36 <> encode_ftype fty   <> encode_address src <> encode_address dst

  Pow  fty src1 src2 dst -> B.word8 0x37 <> encode_ftype fty <> encode_address src1 <> encode_address src2 <> encode_address dst
  Log  fty src1 src2 dst -> B.word8 0x38 <> encode_ftype fty <> encode_address src1 <> encode_address src2 <> encode_address dst
  Sin  fty  src dst -> B.word8 0x39 <> encode_ftype fty   <> encode_address src <> encode_address dst
  Exp  fty  src dst -> B.word8 0x3A <> encode_ftype fty   <> encode_address src <> encode_address dst
  Ln   fty  src dst -> B.word8 0x3B <> encode_ftype fty   <> encode_address src <> encode_address dst

  Eq ty src1 src2 dst -> B.word8 0x40 <> encode_type ty <> encode_address src1 <> encode_address src2 <> encode_address dst
  Lt ty src1 src2 dst -> B.word8 0x41 <> encode_type ty <> encode_address src1 <> encode_address src2 <> encode_address dst
  Gt ty src1 src2 dst -> B.word8 0x42 <> encode_type ty <> encode_address src1 <> encode_address src2 <> encode_address dst

  Castf fty       src dst -> B.word8 0x50 <> encode_ftype fty   <> encode_address src  <> encode_address dst
  Ftoi  fty  sity src dst -> B.word8 0x51 <> encode_ftype fty   <> encode_sitype  sity <> encode_address src <> encode_address dst
  Itof  sity fty  src dst -> B.word8 0x52 <> encode_sitype sity <> encode_ftype   fty  <> encode_address src <> encode_address dst

  Trace size addr -> B.word8 0xFF <> encode_address size <> encode_address addr
