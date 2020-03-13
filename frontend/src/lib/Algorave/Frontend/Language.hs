{-|
Module      : Algorave.Language.Frontend.Language
Description : 
Copyright   : (c) Alexander Vieth, 2020
Licence     : BSD3
Maintainer  : aovieth@gmail.com
Stability   : experimental
Portability : non-portable (GHC only)
-}


{-# LANGUAGE GADTSyntax #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE BangPatterns #-}

-- TODO factor this module into
-- - the language itself
-- - assembly to the backend language
-- - parsing of concrete syntax
module Algorave.Frontend.Language where

import Control.Applicative ((<|>))
import Control.Monad (void)
import Data.Attoparsec.Text (Parser)
import qualified Data.Attoparsec.Text as Atto
import qualified Data.Map.Lazy as Lazy (Map)
import qualified Data.Map.Lazy as Lazy.Map
import Data.Bits
import Data.Char (isAlpha, isSpace, ord)
import Data.Int
import Data.Text (Text)
import qualified Data.Text as T
import Data.Word

import Algorave.Language.Instruction hiding (Instruction (..))
import qualified Algorave.Language.Program as L
import qualified Algorave.Language.Instruction as L

-- | Case-insensitive big-endian hex number with optional 0x prefix.
-- Does not use attoparsec's `hexadecimal` function because that one accepts
-- arbitrarily-many hex digits.
parse_hex :: (Integral b, Bits b) => Word -> Parser b
parse_hex n = (prefix *> parse_unprefixed_hex n) <|> parse_unprefixed_hex n
  where
  prefix = Atto.string "0x"

-- | Argument is the maximum number of hex digits.
parse_unprefixed_hex :: (Integral b, Bits b) => Word -> Parser b
parse_unprefixed_hex m = do
  digits <- Atto.takeWhile1 isHexDigit
  if T.length digits > fromIntegral m
  then fail "hex digit too long"
  else pure (T.foldl' step 0 digits)
  where
  isHexDigit c = (c >= '0' && c <= '9') ||
                 (c >= 'a' && c <= 'f') ||
                 (c >= 'A' && c <= 'F')

  step a c | w >= 48 && w <= 57 = (a `shiftL` 4) .|. fromIntegral (w - 48)
           | w >= 97            = (a `shiftL` 4) .|. fromIntegral (w - 87)
           | otherwise          = (a `shiftL` 4) .|. fromIntegral (w - 55)
    where w = ord c

parse_address :: Parser Address
parse_address = parse_hex 8

parse_uitype :: Parser UIType
parse_uitype = Atto.choice
  [ TU8  <$ Atto.string "u8"
  , TU16 <$ Atto.string "u16"
  , TU32 <$ Atto.string "u32"
  , TU64 <$ Atto.string "u64"
  ]

parse_sitype :: Parser SIType
parse_sitype = Atto.choice
  [ TI8  <$ Atto.string "i8"
  , TI16 <$ Atto.string "i16"
  , TI32 <$ Atto.string "i32"
  , TI64 <$ Atto.string "i64"
  ]

parse_ftype :: Parser FType
parse_ftype = Atto.choice
  [ TF32 <$ Atto.string "f32"
  , TF64 <$ Atto.string "f64"
  ]

parse_itype :: Parser IType
parse_itype = Atto.choice
  [ Unsigned <$> parse_uitype
  , Signed   <$> parse_sitype
  ]

parse_type :: Parser Type
parse_type = Atto.choice
  [ Integral   <$> parse_itype
  , Fractional <$> parse_ftype
  ]

-- | Constants are "now", "start", "rate", or a literal prefixed by its type and
-- a colon, as in
--
--   "f32:3.14"
--   "u32:0xFF00AA42"
parse_constant :: Parser Constant
parse_constant = Atto.choice
  [ Now   <$ Atto.string "now"
  , Start <$ Atto.string "start"
  , Rate  <$ Atto.string "rate"
  , literal
  ]
  where
  literal = do
    ty <- parse_type
    Atto.char ':'
    case ty of
      Integral (Unsigned TU8)  -> U8  <$> parse_u8
      Integral (Unsigned TU16) -> U16 <$> parse_u16
      Integral (Unsigned TU32) -> U32 <$> parse_u32
      Integral (Unsigned TU64) -> U64 <$> parse_u64
      Integral (Signed TI8)  -> I8  <$> parse_i8
      Integral (Signed TI16) -> I16 <$> parse_i16
      Integral (Signed TI32) -> I32 <$> parse_i32
      Integral (Signed TI64) -> I64 <$> parse_i64
      Fractional TF32 -> F32 <$> parse_f32
      Fractional TF64 -> F64 <$> parse_f64

parse_u8 :: Parser Word8
parse_u8 = parse_hex 2

parse_u16 :: Parser Word16
parse_u16 = parse_hex 4

parse_u32 :: Parser Word32
parse_u32 = parse_hex 8

parse_u64 :: Parser Word64
parse_u64 = parse_hex 16

-- TODO better parser for signed integers. Shouldn't have to write it as its
-- signed binary rep.

parse_i8 :: Parser Int8
parse_i8 = Atto.choice
  [ fromIntegral <$> parse_u8
  ]

parse_i16 :: Parser Int16
parse_i16 = Atto.choice
  [ fromIntegral <$> parse_u16
  ]

parse_i32 :: Parser Int32
parse_i32 = Atto.choice
  [ fromIntegral <$> parse_u32
  ]

parse_i64 :: Parser Int64
parse_i64 = Atto.choice
  [ fromIntegral <$> parse_u64
  ]

parse_f32 :: Parser Float
parse_f32 = Atto.rational

parse_f64 :: Parser Double
parse_f64 = Atto.double

type Label = Text

parse_label :: Parser Label
parse_label = Atto.takeWhile1 isAlpha

-- | Check for an address first and otherwise try a label.
parse_label_or_address :: Parser (Either Label Address)
parse_label_or_address = Atto.choice
  [ Right <$> parse_address
  , Left  <$> parse_label
  ]

data Instruction where

  Trace  :: !Address -> !Address -> Instruction

  Stop   :: Instruction
  Branch :: !(Either Label Address) -> !Address -> Instruction
  Jump   :: !(Either Label Address) -> Instruction

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
  Absf :: !FType  -> !Address -> !Address -> Instruction

  Pow :: !FType -> !Address -> !Address -> !Address -> Instruction
  Log :: !FType -> !Address -> !Address -> !Address -> Instruction
  Sin :: !FType -> !Address -> !Address -> Instruction
  Exp :: !FType -> !Address -> !Address -> Instruction
  Ln  :: !FType -> !Address -> !Address -> Instruction

  Eq :: !Type -> !Address -> !Address -> !Address -> Instruction
  Lt :: !Type -> !Address -> !Address -> !Address -> Instruction
  Gt :: !Type -> !Address -> !Address -> !Address -> Instruction

  Castf :: !FType  -> !Address -> !Address -> Instruction
  Ftoi  :: !FType  -> !SIType  -> !Address -> !Address -> Instruction
  Itof  :: !SIType -> !FType   -> !Address -> !Address -> Instruction

deriving instance Show Instruction

parse_instruction :: Parser Instruction
parse_instruction = Atto.choice
  [ Trace <$ Atto.string "trace" <* ws <*> parse_address <* ws <*> parse_address

  , Stop   <$ Atto.string "stop"
  , Branch <$ Atto.string "branch" <* ws <*> parse_label_or_address <* ws <*> parse_address
  , Jump   <$ Atto.string "jump"   <* ws <*> parse_label_or_address

  , Copy <$ Atto.string "copy" <* ws <*> parse_type     <* ws <*> parse_address <* ws <*> parse_address
  , Set  <$ Atto.string "set"  <* ws <*> parse_constant <* ws <*> parse_address

  , Read  <$ Atto.string "read"  <* ws <*> parse_type <* ws <*> parse_address <* ws <*> parse_address <* ws <*> parse_address
  , Write <$ Atto.string "write" <* ws <*> parse_type <* ws <*> parse_address <* ws <*> parse_address <* ws <*> parse_address

  , Or     <$ Atto.string "or"     <* ws <*> parse_uitype <* ws <*> parse_address <* ws <*> parse_address <* ws <*> parse_address
  , And    <$ Atto.string "and"    <* ws <*> parse_uitype <* ws <*> parse_address <* ws <*> parse_address <* ws <*> parse_address
  , Xor    <$ Atto.string "xor"    <* ws <*> parse_uitype <* ws <*> parse_address <* ws <*> parse_address <* ws <*> parse_address
  , Not    <$ Atto.string "not"    <* ws <*> parse_uitype <* ws <*> parse_address <* ws <*> parse_address
  , Shiftl <$ Atto.string "shiftl" <* ws <*> parse_uitype <* ws <*> parse_address <* ws <*> parse_address <* ws <*> parse_address
  , Shiftr <$ Atto.string "shiftr" <* ws <*> parse_uitype <* ws <*> parse_address <* ws <*> parse_address <* ws <*> parse_address

  , Add <$ Atto.string "add" <* ws <*> parse_type <* ws <*> parse_address <* ws <*> parse_address <* ws <*> parse_address
  , Sub <$ Atto.string "sub" <* ws <*> parse_type <* ws <*> parse_address <* ws <*> parse_address <* ws <*> parse_address
  , Mul <$ Atto.string "mul" <* ws <*> parse_type <* ws <*> parse_address <* ws <*> parse_address <* ws <*> parse_address
  , Div <$ Atto.string "div" <* ws <*> parse_type <* ws <*> parse_address <* ws <*> parse_address <* ws <*> parse_address
  , Mod <$ Atto.string "mod" <* ws <*> parse_type <* ws <*> parse_address <* ws <*> parse_address <* ws <*> parse_address

  , Abs  <$ Atto.string "abs"  <* ws <*> parse_sitype <* ws <*> parse_address <* ws <*> parse_address
  , Absf <$ Atto.string "absf" <* ws <*> parse_ftype  <* ws <*> parse_address <* ws <*> parse_address

  , Pow <$ Atto.string "pow" <* ws <*> parse_ftype <* ws <*> parse_address <* ws <*> parse_address <* ws <*> parse_address
  , Log <$ Atto.string "log" <* ws <*> parse_ftype <* ws <*> parse_address <* ws <*> parse_address <* ws <*> parse_address
  , Sin <$ Atto.string "sin" <* ws <*> parse_ftype <* ws <*> parse_address <* ws <*> parse_address
  , Exp <$ Atto.string "exp" <* ws <*> parse_ftype <* ws <*> parse_address <* ws <*> parse_address
  , Ln  <$ Atto.string "ln"  <* ws <*> parse_ftype <* ws <*> parse_address <* ws <*> parse_address

  , Eq <$ Atto.string "eq" <* ws <*> parse_type <* ws <*> parse_address <* ws <*> parse_address <* ws <*> parse_address
  , Lt <$ Atto.string "lt" <* ws <*> parse_type <* ws <*> parse_address <* ws <*> parse_address <* ws <*> parse_address
  , Gt <$ Atto.string "gt" <* ws <*> parse_type <* ws <*> parse_address <* ws <*> parse_address <* ws <*> parse_address

  , Castf <$ Atto.string "castf" <* ws <*> parse_ftype  <* ws <*> parse_address <* ws <*> parse_address
  , Ftoi  <$ Atto.string "ftoi"  <* ws <*> parse_ftype  <* ws <*> parse_sitype  <* ws <*> parse_address <* ws <*> parse_address
  , Itof  <$ Atto.string "itof"  <* ws <*> parse_sitype <* ws <*> parse_ftype   <* ws <*> parse_address <* ws <*> parse_address
  ]
  where
  ws :: Parser ()
  ws = void $ Atto.many1 Atto.space

-- Simple comment syntax: # until \n.
parse_comment :: Parser Text
parse_comment = Atto.char '#' *> Atto.takeTill (== '\n')

parse_item :: Parser Item
parse_item = Atto.choice
  [ Label <$> parse_label <* Atto.char ':'
  , Instruction <$> parse_instruction <* Atto.char ';'
  , Comment <$ parse_comment
  ]

-- | Every instruction from the algorave low-level language, with support
-- for labels.
--
-- Comment text is not included here, since there's no need for it at the
-- moment.
data Item where
  Instruction :: Instruction -> Item
  Label       :: Text -> Item
  Comment     :: Item

deriving instance Show Item

newtype Program = Program { getProgram :: [Item] }

deriving instance Show Program

-- | colon-separated items with tabs and spaces acceptable.
-- The final instruction does not need a newline after it.
parse_program :: Parser Program
parse_program = Program <$> Atto.many' (ws *> parse_item)
  where
  ws = void $ Atto.many' Atto.space

-- Next step: compiling to the backend language.
-- How shall it be done? The only challenge is to resolve labels.
-- A map from label to index can be made by a simple fold, which can
-- simultaneously remove the labels. We cannot convert to backend syntax until
-- we know the labels though, but can laziness help us here? I think so.

-- For each label, its resolved offset in the program. Must use a lazy
-- map, for we will use lookups before the map has been completely constructed.
type JumpTable = Lazy.Map Text Word32

-- TODO error handling (unknown labels).
assemble :: Program -> L.Program
assemble (Program items) =
  let (_, _, instructions) = foldl assemble_item initial_state items
  in  L.Program (reverse instructions)
  where
  -- The jump table and the current index (does not increase for label items).
  -- Third component is the assembled program in reverse order (it's a left
  -- fold).
  initial_state :: (JumpTable, Word32, [L.Instruction])
  initial_state = (Lazy.Map.empty, 0, [])


-- TODO assemble errors for unresolved labels.
-- Semantics for duplicate labels? It'll jump to the one that appeared closest
-- before.
assemble_item
  :: (JumpTable, Word32, [L.Instruction])
  -> Item
  -> (JumpTable, Word32, [L.Instruction])
assemble_item (jump_table, !idx, insts) item = case item of
  Instruction inst ->
    -- Essential that we do not force this, because Branch and Jump on labels
    -- use the map which may not have the desired label just yet.
    let inst' = assemble_instruction jump_table inst
    in  (jump_table, idx+1, (inst':insts))
  -- Semantics for duplicate labels
  Label       lbl  ->
    -- May overwrite a prior label, but that's fine. For duplicate labels,
    -- branches and jumps will resolve to the one most recently defined before
    -- that instruction.
    let jump_table' = Lazy.Map.insert lbl idx jump_table
    in  (jump_table', idx, insts)
  Comment -> (jump_table, idx, insts)

-- | Resolve the labels in branch and jump. Everything else is trivial.
assemble_instruction :: JumpTable -> Instruction -> L.Instruction
assemble_instruction jt inst = case inst of

  Branch (Left l) b -> case Lazy.Map.lookup l jt of
    Nothing -> error ("assemble_instruction: unknown label " ++ show l)
    Just a -> L.Branch a b
  Jump (Left l) -> case Lazy.Map.lookup l jt of
    Nothing -> error ("assemble_instruction: unknown label " ++ show l)
    Just a -> L.Jump a

  Trace a b -> L.Trace a b
  Stop -> L.Stop
  Branch (Right a) b -> L.Branch a b
  Jump (Right a) -> L.Jump a
  Copy a b c -> L.Copy a b c
  Set a b -> L.Set a b
  Read a b c d -> L.Read a b c d
  Write a b c d -> L.Write a b c d
  Or a b c d -> L.Or a b c d
  And a b c d -> L.And a b c d
  Xor a b c d -> L.Xor a b c d
  Not a b c -> L.Not a b c
  Shiftl a b c d -> L.Shiftl a b c d
  Shiftr a b c d -> L.Shiftr a b c d

  Add a b c d -> L.Add a b c d
  Sub a b c d -> L.Sub a b c d
  Mul a b c d -> L.Mul a b c d
  Div a b c d -> L.Div a b c d
  Mod a b c d -> L.Mod a b c d

  Abs a b c -> L.Abs a b c
  Absf a b c -> L.Absf a b c

  Pow a b c d -> L.Pow a b c d
  Log a b c d -> L.Log a b c d
  Sin a b c -> L.Sin a b c
  Exp a b c -> L.Exp a b c
  Ln a b c -> L.Ln a b c

  Eq a b c d -> L.Eq a b c d
  Lt a b c d -> L.Lt a b c d
  Gt a b c d -> L.Gt a b c d

  Castf a b c -> L.Castf a b c
  Ftoi a b c d -> L.Ftoi a b c d
  Itof a b c d -> L.Itof a b c d
