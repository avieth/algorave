{-|
Module      : Algorave.Language.Instruction
Description : Data types corresponding to the rust language definitions
Copyright   : (c) Alexander Vieth, 2020
Licence     : BSD3
Maintainer  : aovieth@gmail.com
Stability   : experimental
Portability : non-portable (GHC only)
-}

{-# LANGUAGE GADTSyntax #-}
{-# LANGUAGE StandaloneDeriving #-}

module Algorave.Language.Instruction where

import Data.Word (Word8, Word16, Word32, Word64)
import Data.Int (Int8, Int16, Int32, Int64)
import GHC.Float (Float, Double)

type Address = Word32

data Type where
  Integral   :: !IType -> Type
  Fractional :: !FType -> Type

deriving instance Show Type
deriving instance Eq Type
deriving instance Ord Type

data IType where
  Unsigned :: !UIType -> IType
  Signed   :: !SIType -> IType

deriving instance Show IType
deriving instance Eq IType
deriving instance Ord IType

data FType where
  TF32 :: FType
  TF64 :: FType

deriving instance Show FType
deriving instance Eq FType
deriving instance Ord FType

data UIType where
  TU8  :: UIType
  TU16 :: UIType
  TU32 :: UIType
  TU64 :: UIType

deriving instance Show UIType
deriving instance Eq UIType
deriving instance Ord UIType

data SIType where
  TI8  :: SIType
  TI16 :: SIType
  TI32 :: SIType
  TI64 :: SIType

deriving instance Show SIType
deriving instance Eq SIType
deriving instance Ord SIType

data Constant where

  Now   :: Constant
  Start :: Constant
  Rate  :: Constant

  U8  :: !Word8 -> Constant
  U16 :: !Word16 -> Constant
  U32 :: !Word32 -> Constant
  U64 :: !Word64 -> Constant

  I8  :: !Int8 -> Constant
  I16 :: !Int16 -> Constant
  I32 :: !Int32 -> Constant
  I64 :: !Int64 -> Constant

  F32 :: !Float -> Constant
  F64 :: !Double -> Constant

deriving instance Show Constant

data Instruction where

  Trace :: !Address -> !Address -> Instruction

  Stop   :: Instruction
  Branch :: !Address -> !Address -> Instruction
  Jump   :: !Address -> Instruction

  Copy :: !Type -> !Address -> !Address -> Instruction
  Set  :: !Constant -> !Address -> Instruction

  Read  :: !Type -> !Address -> !Address -> !Address -> Instruction
  Write :: !Type -> !Address -> !Address -> !Address -> Instruction

  Or     :: !UIType -> !Address -> !Address -> !Address -> Instruction
  And    :: !UIType -> !Address -> !Address -> !Address -> Instruction
  Xor    :: !UIType -> !Address -> !Address -> !Address -> Instruction
  Not    :: !UIType -> !Address -> !Address -> Instruction
  Shiftl :: !UIType -> !Address -> !Address -> !Address -> Instruction
  Shiftr :: !UIType -> !Address -> !Address -> !Address -> Instruction

  Add :: !Type -> !Address -> !Address -> !Address -> Instruction
  Sub :: !Type -> !Address -> !Address -> !Address -> Instruction
  Mul :: !Type -> !Address -> !Address -> !Address -> Instruction
  Div :: !Type -> !Address -> !Address -> !Address -> Instruction
  Mod :: !Type -> !Address -> !Address -> !Address -> Instruction

  Abs  :: !SIType -> !Address -> !Address -> Instruction
  Absf :: !FType -> !Address -> !Address -> Instruction

  Pow :: !FType -> !Address -> !Address -> !Address -> Instruction
  Log :: !FType -> !Address -> !Address -> !Address -> Instruction
  Sin :: !FType -> !Address -> !Address -> Instruction
  Exp :: !FType -> !Address -> !Address -> Instruction
  Ln  :: !FType -> !Address -> !Address -> Instruction

  Eq :: !Type -> !Address -> !Address -> !Address -> Instruction
  Lt :: !Type -> !Address -> !Address -> !Address -> Instruction
  Gt :: !Type -> !Address -> !Address -> !Address -> Instruction

  Castf :: !FType -> !Address -> !Address -> Instruction
  Ftoi  :: !FType -> !SIType -> !Address -> !Address -> Instruction
  Itof  :: !SIType -> !FType -> !Address -> !Address -> Instruction

deriving instance Show Instruction

-- What the instruction type _should_ be.
--
-- 1. It must be possible to source address offsets and input/output identifiers
--    from memory. In particular, to compute an offset using the ALU.
-- 2. It would be nice if some instructions could take literals, as this would
--    save a load and also make programming directly in the machine language a
--    lot more feasible.
-- 3. Decide on the branching system...
--
-- Common ideas we'll want to express:
-- 0. A "register" i.e. some location in main memory (not an I/O region)
--    This is just a Word32.
-- 1. A "register" with an associated type indicating how the bytes at that
--    address will be interpreted.
-- 
-- 1. A literal or an address, with an implied type. In concrete syntax we
--    might write things like
--      3.14f32
--      42u16
--      %0x04i32  # memory address, its value interpreted as 4 byte signed
--                # integer (address itself is always u32)
--      ?0
--
-- Decision 1: non-uniform treatment of memory, input, and output. All that
-- you can do with input is copy from it, and all that you can do with output
-- is copy to it.
--
-- Decision 2: the only instruction which may use an address and read an offset
-- from it is the copy instruction. All other instructions take statically-known
-- offsets.
--
-- Decision 3: the only way in which a constant value may be used is the
-- set instruction, which writes it to a memory location.
--
-- Decision 4: all ALU/FPU instructions will work only against memory
-- locations.
--
--
-- How would we do pointers in this language? These are Word32s. We can do
-- pointer arithmetic as expected. What makes them pointers, though, is the
-- fact that they can be used in Copy to dereference.
--
-- Do we need to be able to "set" to a relative offset? I don't think so: we
-- can set a known offset and then move it to the dynamic offset.
--
-- Decision 5 (rehash of 2): all addresses are static (obviously) it's just
-- that the only instruction which interprets the value at an address as itself
-- an address is "copy". Thus, "pointer dereference" is explicit.
--
-- And now the final decision must surely be about branching.
-- Which conditions should we offer?
-- Surely one single branch if not zero instruction _ought_ to be enough.
-- If we adhere always to the convention that 0x01 is true and 0x00 is false
-- and everything else is nonsense, then we can get any if/then/else
-- predicate by using the logical ALU operations where all arguments are also
-- in {0x00, 0x01}.
-- Maybe the solution really is to not have the {-1, 0, 1} comparison, but
-- instead simple "is greater", "is lesser", and "is equal", which is
-- compatible with floating point as well.
--
-- Decision 6: the only branch operations will be branch if equal and
-- unconditional jump.
--
-- Decision 7: comparison instructions will be 3 separate boolean-valued
-- operations, which work for all numeric types.
--

