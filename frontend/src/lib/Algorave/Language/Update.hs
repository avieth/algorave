{-|
Module      : Algorave.Language.Update
Description : 
Copyright   : (c) Alexander Vieth, 2020
Licence     : BSD3
Maintainer  : aovieth@gmail.com
Stability   : experimental
Portability : non-portable (GHC only)
-}

module Algorave.Language.Update where

import Data.ByteString.Lazy (ByteString)

import Algorave.Language.Instruction (Address)
import Algorave.Language.Program (Program)

data MemoryUpdate = MemoryUpdate !Address !ByteString

mem_update_offset :: MemoryUpdate -> Address
mem_update_offset (MemoryUpdate offset _) = offset

mem_update_bytes :: MemoryUpdate -> ByteString
mem_update_bytes (MemoryUpdate _ bytes) = bytes

data Update = Update !(Maybe Program) ![MemoryUpdate]

no_update :: Update
no_update = Update Nothing []

update_program :: Update -> Maybe Program
update_program (Update mprog _) = mprog

update_mem_updates :: Update -> [MemoryUpdate]
update_mem_updates (Update _ ms) = ms
