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
import Control.Monad (forM_, void)
import Control.Monad.Trans.Except
import Control.Monad.Trans.State.Strict
import Data.Attoparsec.Text (Parser, (<?>))
import qualified Data.Attoparsec.Text as Atto
import Data.Functor.Identity (Identity (..))
import Data.Map (Map)
import qualified Data.Map as Map
import Data.Bits
import Data.Char (isAlpha, isSpace, ord)
import Data.DList (DList)
import qualified Data.DList as DList
import Data.Either (partitionEithers)
import Data.Int
import Data.List.NonEmpty (NonEmpty, nonEmpty)
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
parse_label = Atto.takeWhile1 (\c -> isAlpha c || c == '_' || c == '-')

-- | Check for an address first and otherwise try a label.
parse_label_or_address :: Parser (Either Label Address)
parse_label_or_address = Atto.choice
  [ Right <$  Atto.char '%' <*> parse_address
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
  [ Trace <$ Atto.string "trace" <* ws <*> parse_address <* ws <*> parse_address <?> "trace"

  , Stop   <$ Atto.string "stop" <?> "stop"
  , Branch <$ Atto.string "branch" <* ws <*> parse_label_or_address <* ws <*> parse_address <?> "branch"
  , Jump   <$ Atto.string "jump"   <* ws <*> parse_label_or_address <?> "jump"

  , Copy <$ Atto.string "copy" <* ws <*> parse_type     <* ws <*> parse_address <* ws <*> parse_address <?> "copy"
  , Set  <$ Atto.string "set"  <* ws <*> parse_constant <* ws <*> parse_address <?> "set"

  , Read  <$ Atto.string "read"  <* ws <*> parse_type <* ws <*> parse_address <* ws <*> parse_address <* ws <*> parse_address <?> "read"
  , Write <$ Atto.string "write" <* ws <*> parse_type <* ws <*> parse_address <* ws <*> parse_address <* ws <*> parse_address <?> "write"

  , Or     <$ Atto.string "or"     <* ws <*> parse_uitype <* ws <*> parse_address <* ws <*> parse_address <* ws <*> parse_address <?> "or"
  , And    <$ Atto.string "and"    <* ws <*> parse_uitype <* ws <*> parse_address <* ws <*> parse_address <* ws <*> parse_address <?> "and"
  , Xor    <$ Atto.string "xor"    <* ws <*> parse_uitype <* ws <*> parse_address <* ws <*> parse_address <* ws <*> parse_address <?> "xor"
  , Not    <$ Atto.string "not"    <* ws <*> parse_uitype <* ws <*> parse_address <* ws <*> parse_address <?> "not"
  , Shiftl <$ Atto.string "shiftl" <* ws <*> parse_uitype <* ws <*> parse_address <* ws <*> parse_address <* ws <*> parse_address <?> "shiftl"
  , Shiftr <$ Atto.string "shiftr" <* ws <*> parse_uitype <* ws <*> parse_address <* ws <*> parse_address <* ws <*> parse_address <?> "shiftr"

  , Add <$ Atto.string "add" <* ws <*> parse_type <* ws <*> parse_address <* ws <*> parse_address <* ws <*> parse_address <?> "add"
  , Sub <$ Atto.string "sub" <* ws <*> parse_type <* ws <*> parse_address <* ws <*> parse_address <* ws <*> parse_address <?> "sub"
  , Mul <$ Atto.string "mul" <* ws <*> parse_type <* ws <*> parse_address <* ws <*> parse_address <* ws <*> parse_address <?> "mul"
  , Div <$ Atto.string "div" <* ws <*> parse_type <* ws <*> parse_address <* ws <*> parse_address <* ws <*> parse_address <?> "div"
  , Mod <$ Atto.string "mod" <* ws <*> parse_type <* ws <*> parse_address <* ws <*> parse_address <* ws <*> parse_address <?> "mod"

  , Abs  <$ Atto.string "abs"  <* ws <*> parse_sitype <* ws <*> parse_address <* ws <*> parse_address <?> "abs"
  , Absf <$ Atto.string "absf" <* ws <*> parse_ftype  <* ws <*> parse_address <* ws <*> parse_address <?> "absf"

  , Pow <$ Atto.string "pow" <* ws <*> parse_ftype <* ws <*> parse_address <* ws <*> parse_address <* ws <*> parse_address <?> "pow"
  , Log <$ Atto.string "log" <* ws <*> parse_ftype <* ws <*> parse_address <* ws <*> parse_address <* ws <*> parse_address <?> "log"
  , Sin <$ Atto.string "sin" <* ws <*> parse_ftype <* ws <*> parse_address <* ws <*> parse_address <?> "sin"
  , Exp <$ Atto.string "exp" <* ws <*> parse_ftype <* ws <*> parse_address <* ws <*> parse_address <?> "exp"
  , Ln  <$ Atto.string "ln"  <* ws <*> parse_ftype <* ws <*> parse_address <* ws <*> parse_address <?> "ln"

  , Eq <$ Atto.string "eq" <* ws <*> parse_type <* ws <*> parse_address <* ws <*> parse_address <* ws <*> parse_address <?> "eq"
  , Lt <$ Atto.string "lt" <* ws <*> parse_type <* ws <*> parse_address <* ws <*> parse_address <* ws <*> parse_address <?> "lt"
  , Gt <$ Atto.string "gt" <* ws <*> parse_type <* ws <*> parse_address <* ws <*> parse_address <* ws <*> parse_address <?> "gt"

  , Castf <$ Atto.string "castf" <* ws <*> parse_ftype  <* ws <*> parse_address <* ws <*> parse_address <?> "castf"
  , Ftoi  <$ Atto.string "ftoi"  <* ws <*> parse_ftype  <* ws <*> parse_sitype  <* ws <*> parse_address <* ws <*> parse_address <?> "ftoi"
  , Itof  <$ Atto.string "itof"  <* ws <*> parse_sitype <* ws <*> parse_ftype   <* ws <*> parse_address <* ws <*> parse_address <?> "itof"
  ]
  where
  ws :: Parser ()
  ws = void $ Atto.many1 Atto.space

-- Simple comment syntax: # until \n.
parse_comment :: Parser Text
parse_comment = Atto.char '#' *> Atto.takeTill (== '\n')

parse_item :: Parser Item
parse_item = Atto.choice
  [ (Comment <$ parse_comment) <?> "comment"
  , (Label <$> parse_label <* Atto.char ':') <?> "label"
  , (Instruction <$> parse_instruction <* Atto.char ';') <?> "instruction"
  ] <?> "item"

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
--
-- Problem: we want to be able to run the parser against a known input string
-- and fail with the proper error message if any item fails to parse.
parse_program :: Parser Program
--parse_program = Program <$> Atto.many' (ws *> parse_item)
parse_program = Program <$> Atto.manyTill' (ws *> parse_item) (Atto.endOfInput)
  where
  ws = void $ Atto.many' Atto.space

-- Next step: compiling to the backend language.
-- How shall it be done? The only challenge is to resolve labels.
-- A map from label to index can be made by a simple fold, which can
-- simultaneously remove the labels. We cannot convert to backend syntax until
-- we know the labels though, but can laziness help us here? I think so.

-- For each label, its resolved offset in the program. Must use a lazy
-- map, for we will use lookups before the map has been completely constructed.
type JumpTable = Map Text Word32

data AssembleError where
  UnknownLabel :: Text -> AssembleError

deriving instance Show AssembleError

data AssembleState = AssembleState
  { instruction_counter :: !Word32
  -- The map from labels to the instruction index to which they should resolve.
  -- This is extended by Label items.
  , jump_table          :: JumpTable
  , instructions        :: JumpTable -> Except AssembleError (DList L.Instruction)
  }

-- What we want to do is fold over the list of instructions and compute
--   - the instruction index at each spot
--   - the map of label to instruction index
-- and from this get a _continuation_ which can produce either errors or the
-- assembled program.

type Assemble = State AssembleState

assemble :: Program -> Either AssembleError L.Program
assemble (Program items) =
  let astate = AssembleState
        { instruction_counter = 0
        , jump_table = Map.empty
        , instructions = const (pure DList.empty)
        }
      Identity astate' = execStateT (forM_ items assemble_item) astate
  in  runIdentity . runExceptT . fmap (L.Program . DList.toList) $
        instructions astate' (jump_table astate')
      

assemble_item :: Item -> Assemble ()
assemble_item item = case item of
  Instruction inst -> assemble_instruction inst
  Label lbl -> modify $ \as -> as
    { jump_table = Map.insert lbl (instruction_counter as) (jump_table as)
    }
  Comment -> pure ()

add_simple_instruction :: L.Instruction -> Assemble ()
add_simple_instruction inst = modify $ \as -> as
  { instruction_counter = instruction_counter as + 1
  , instructions = \jt -> do
      insts <- instructions as jt
      pure (DList.snoc insts inst)
  }

-- | Resolve the labels in branch and jump. Everything else is trivial.
-- For branch and jump, they expand to 2 instructions: one to set the target
-- address in location 0x00, and the next to actually do the jump or branch.
assemble_instruction :: Instruction -> Assemble ()
assemble_instruction inst = case inst of

  Branch (Left l) b -> modify $ \as -> as
    { instruction_counter = instruction_counter as + 2
    , instructions = \jt -> do
        insts <- instructions as jt
        case Map.lookup l jt of
          Nothing   -> throwE (UnknownLabel l)
          Just addr ->
            -- Jump/branch addresses are relative. The current index is
            -- instruction_counter as + 1 so we subtract  that from the
            -- desired jump point.
            -- We added 2 to instruction counter above because we want it to
            -- give the _next_ instruction index.
            let raddr :: Int32
                raddr = fromIntegral addr - fromIntegral (instruction_counter as + 1)
            in  pure (DList.append insts (DList.fromList [L.Set (I32 raddr) 0x00, L.Branch 0x00 b]))
          
    }

  Jump (Left l) -> modify $ \as -> as
    { instruction_counter = instruction_counter as + 2
    , instructions = \jt -> do
        insts <- instructions as jt
        case Map.lookup l jt of
          Nothing   -> throwE (UnknownLabel l)
          Just addr ->
            let raddr :: Int32
                raddr = fromIntegral addr - fromIntegral (instruction_counter as + 1)
            in  pure (DList.append insts (DList.fromList [L.Set (I32 raddr) 0x00, L.Jump 0x00]))
          
    }

  Trace a b -> add_simple_instruction (L.Trace a b)
  Stop -> add_simple_instruction (L.Stop)
  Branch (Right a) b -> add_simple_instruction (L.Branch a b)
  Jump (Right a) -> add_simple_instruction (L.Jump a)
  Copy a b c -> add_simple_instruction (L.Copy a b c)
  Set a b -> add_simple_instruction (L.Set a b)
  Read a b c d -> add_simple_instruction (L.Read a b c d)
  Write a b c d -> add_simple_instruction (L.Write a b c d)
  Or a b c d -> add_simple_instruction (L.Or a b c d)
  And a b c d -> add_simple_instruction (L.And a b c d)
  Xor a b c d -> add_simple_instruction (L.Xor a b c d)
  Not a b c -> add_simple_instruction (L.Not a b c)
  Shiftl a b c d -> add_simple_instruction (L.Shiftl a b c d)
  Shiftr a b c d -> add_simple_instruction (L.Shiftr a b c d)

  Add a b c d -> add_simple_instruction (L.Add a b c d)
  Sub a b c d -> add_simple_instruction (L.Sub a b c d)
  Mul a b c d -> add_simple_instruction (L.Mul a b c d)
  Div a b c d -> add_simple_instruction (L.Div a b c d)
  Mod a b c d -> add_simple_instruction (L.Mod a b c d)

  Abs a b c -> add_simple_instruction (L.Abs a b c)
  Absf a b c -> add_simple_instruction (L.Absf a b c)

  Pow a b c d -> add_simple_instruction (L.Pow a b c d)
  Log a b c d -> add_simple_instruction (L.Log a b c d)
  Sin a b c -> add_simple_instruction (L.Sin a b c)
  Exp a b c -> add_simple_instruction (L.Exp a b c)
  Ln a b c -> add_simple_instruction (L.Ln a b c)

  Eq a b c d -> add_simple_instruction (L.Eq a b c d)
  Lt a b c d -> add_simple_instruction (L.Lt a b c d)
  Gt a b c d -> (add_simple_instruction (L.Gt a b c d))

  Castf a b c  -> add_simple_instruction (L.Castf a b c)
  Ftoi a b c d -> add_simple_instruction (L.Ftoi a b c d)
  Itof a b c d -> add_simple_instruction (L.Itof a b c d)
