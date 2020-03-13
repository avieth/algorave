{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE GADTSyntax #-}

module Main where

import Control.Applicative ((<|>), many)
import Control.Monad (when)
import qualified Data.Attoparsec.Text as Atto
import Data.Text (Text)
import Data.Text.IO (hGetLine)
import qualified Data.Text as T
import Data.ByteString.Builder (hPutBuilder)
import Data.List.NonEmpty (NonEmpty)
import qualified Data.List.NonEmpty as NE
import Data.Text (Text)
import qualified Data.Text as T (pack)
import qualified Data.Text.IO as T (hPutStr)
import qualified Options.Applicative as Opt
import System.IO (BufferMode (..), hIsEOF, hSetBinaryMode, hSetBuffering,
  stderr, stdin, stdout)

import Algorave.Language.Update (Update (..), no_update)
import Algorave.Language.Codec (encode_update)
import Algorave.Language.Instruction
import Algorave.Language.Program (Program (..))
import qualified Algorave.Frontend.Language as Frontend

data Options = Options
  { -- | Where to search for files when loading binary data sections.
    assets_search_path :: !(NonEmpty FilePath)
  }
  deriving (Show)

options_parser :: Opt.Parser Options
options_parser = Options <$> assets_search_path_parser
  where
  assets_search_path_parser :: Opt.Parser (NonEmpty FilePath)
  assets_search_path_parser =
      fmap (\lst -> maybe (pure "./") id (NE.nonEmpty lst))
    . many
    . Opt.option Opt.str
    . mconcat
    $ [ Opt.long "path"
      , Opt.help "Search path for data section file load, in order of preference"
      ]

get_options :: IO Options
get_options = Opt.customExecParser prefs info
  where
  info :: Opt.ParserInfo Options
  info = Opt.info options_parser $ mconcat
    [ Opt.fullDesc
    , Opt.header "A frontend typechecker and compiler for algorave"
    ]
  prefs = Opt.prefs $ mconcat
    [ Opt.showHelpOnError
    ]

data Frontend state m syntax = Frontend
  { -- | Parse abstract syntax from text.
    parser  :: Atto.Parser syntax
    -- | Include the abstract syntax in the system state. To allow for
    -- incremental feedback to the user (progress report for instance) this
    -- does not give an error indicator in the result, but instead may invoke
    -- an error-reporting / status function.
  , include :: Feedback m -> syntax -> state -> m state
  , commit  :: Feedback m -> state -> m (Update, state)
  , reset   :: Feedback m -> state -> m state
  }

dumb_frontend :: Applicative m => Frontend (Bool, Bool) m (Either Text Text)
dumb_frontend = Frontend
  { parser  = dumb_parser
  , include = \_fb syn (!running, _) -> case syn of
      Left  _ -> pure (running, True)
      Right _ -> pure (running, False)
  , commit = \_fb (running, next) -> case (running, next) of
        (True, False) -> pure (update_off, (next, next))
        (False, True) -> pure (update_on,  (next, next))
        _             -> pure (no_update,  (next, next))
  , reset  = \_fb (running, _) -> pure (running, running)
  }
  where
  update_on  = Update (Just program_on)  []
  update_off = Update (Just program_off) []
  program_on = Program [
      -- Trace the current frame.
      Set (U32 0x08) 0x00
    , Set (U32 0x08) 0x04
    , Set Now        0x08
    , Trace 0x00 0x04
    , Stop
    ]
  program_off = Program [
      Stop
    ]
  -- NB: attoparsec makes the dubious decision to defined "string" as
  -- "all-or-nothing": if it fails, no input is consumed. So this is not at
  -- all suitable for use as a streaming parser! The program which runs it
  -- must, on failure, put the leftovers back into the parser on the next try,
  -- but if a failing parser consumed nothing, then it'll look.
  -- Ridiculous.
  dumb_parser = (Left <$> Atto.string (T.pack "on")) <|> (Right <$> Atto.string (T.pack "off"))

-- State is the current program, but only if it has changed since last commit.
proper_frontend :: Monad m => Frontend (Maybe Frontend.Program) m Frontend.Program
proper_frontend = Frontend
  { parser  = Frontend.parse_program <* Atto.endOfInput
  , include = \fb program _ -> do
      debug fb [ T.pack "including program ", T.pack (show program) ]
      pure (Just program)
  , commit = \_fb state -> case state of
      Nothing -> pure (no_update, Nothing)
      Just prog -> do
        let prog' = Frontend.assemble prog
        pure (Update (Just prog') [], Nothing)
  , reset  = \_fb _ -> pure Nothing
  }

data Command where
  Begin  :: Command
  End    :: Command
  Commit :: Command
  Reset  :: Command

parse_command :: Atto.Parser Command
parse_command = Atto.choice
  [ Begin  <$ Atto.string (T.pack "BEGIN")
  , End    <$ Atto.string (T.pack "END")
  , Commit <$ Atto.string (T.pack "COMMIT")
  , Reset  <$ Atto.string (T.pack "RESET")
  ]

-- | Get a command from stdin by reading in a line of text.
-- Leftovers are discarded; the line must exactly match.
read_command_stdin :: IO (Either Text Command)
read_command_stdin = do
  txtLine <- hGetLine stdin
  case Atto.parseOnly (parse_command <* Atto.endOfInput) txtLine of
    Left  err -> pure (Left (T.pack err))
    Right cmd -> pure (Right cmd)

-- | Get all lines of text before a single line containing END is found.
take_until_end :: IO [Text]
take_until_end = go []
  where
  go !acc = do
    txtLine <- hGetLine stdin
    case Atto.parseOnly (parse_command <* Atto.endOfInput) txtLine of
      Right End -> pure (reverse acc)
      -- Other commands are ok; there's no meaning for nested BEGIN/END
      -- brackets.
      _ -> go (txtLine:acc)

-- | Read from stdin until the parser finishes, include the changes, then
-- write generated code to stdout. User feedback appears on stderr, even for
-- non-errors.
run_frontend_loop
  :: forall state syntax void .
     Frontend state IO syntax
  -> state
  -> IO void
run_frontend_loop frontend state = do
  command <- read_command_stdin
  state' <- case command of
    Left  _errTxt -> do
      err stderr_feedback [ T.pack "failed to read command" ]
      pure state
    Right End    -> do
      err stderr_feedback [ T.pack "unexpected END" ]
      pure state
    Right Commit -> do
      debug stderr_feedback [ T.pack "commit" ]
      (update, state') <- commit frontend stderr_feedback state
      hPutBuilder stdout (encode_update update)
      debug stderr_feedback [ T.pack "wrote update to stdout" ]
      pure state'
    Right Reset  -> do
      debug stderr_feedback [ T.pack "reset" ]
      reset frontend stderr_feedback state
    Right Begin  -> do
      debug stderr_feedback [ T.pack "begin" ]
      -- Take every line of text until END appears. BEGIN, COMMIT, and RESET
      -- within these lines is just fine.
      lines <- take_until_end
      --let n = show (length lines)
      --debug stderr_feedback [ T.pack ("got " ++ n ++ " lines") ]
      let result = Atto.parseOnly (parser frontend) (T.intercalate (T.pack "\n") lines)
      case result of
        Left _err -> do
          err stderr_feedback [ T.pack "failed to parse frontend syntax" ]
          pure state
        Right syntax -> include frontend stderr_feedback syntax state
  run_frontend_loop frontend state'

-- | Print a list of lines under a given severity.
data Feedback m = Feedback
  { err   :: [Text] -> m ()
  , warn  :: [Text] -> m ()
  , info  :: [Text] -> m ()
  , debug :: [Text] -> m ()
  }

stderr_feedback :: Feedback IO
stderr_feedback = Feedback
  { err   = \ms -> T.hPutStr stderr $ T.pack "\ESC[91m[ERROR]   = " <> lines ms <> resetColour
  , warn  = \ms -> T.hPutStr stderr $ T.pack "\ESC[93m[WARNING] = " <> lines ms <> resetColour
  , info  = \ms -> T.hPutStr stderr $ T.pack "\ESC[37m[INFO]    = " <> lines ms <> resetColour
  , debug = \ms -> T.hPutStr stderr $ T.pack "\ESC[94m[DEBUG]   = " <> lines ms <> resetColour
  }
  where

  lines :: [Text] -> Text
  lines []     = T.pack "\n"
  lines [x]    = x <> T.pack "\n"
  lines (x:xs) = mconcat (x : T.pack "\n" : fmap indent xs)

  indent :: Text -> Text
  indent m = mconcat [T.pack "          | ", m, T.pack "\n"]

  resetColour :: Text
  resetColour = T.pack "\ESC[0m"

main :: IO ()
main = do
  !options <- get_options
  -- binary encodings of `Update`s will go on stdout so we want binary mode
  -- there. stdin will read textual concrete syntax, and stderr will show
  -- human-readable text feedback.
  hSetBinaryMode stdout True
  -- Output must go immediately to stdout. This is a human-interactive program.
  hSetBuffering stdout NoBuffering
  -- Input is textual concrete syntax, read by the line.
  hSetBuffering stdin LineBuffering
  run_frontend_loop proper_frontend Nothing
