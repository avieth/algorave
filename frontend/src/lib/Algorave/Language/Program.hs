{-|
Module      : Algorave.Language.Program
Description : 
Copyright   : (c) Alexander Vieth, 2020
Licence     : BSD3
Maintainer  : aovieth@gmail.com
Stability   : experimental
Portability : non-portable (GHC only)
-}

module Algorave.Language.Program where

import Data.Word (Word32)

import Algorave.Language.Instruction (Instruction)

newtype Program = Program { getProgram :: [Instruction] }

program_length :: Program -> Word32
program_length = fromIntegral . length . getProgram

program_instructions :: Program -> [Instruction]
program_instructions = getProgram
