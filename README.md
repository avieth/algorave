# algorave: programming as performance

This is a rust JACK client which produces audio and MIDI output by running a
program for each frame/sample. That program (the object program) may be changed
while the rust program (the meta program) continues to run, allowing the
user programmer (performer) to reconfigure the system live.

Also included in this repository is a Haskell frontend which reads
assembly-like concrete syntax and outputs "machine code" that the rust
executable will understand.

A rudimentary "live coding" setup can be established using only a shell,
your favourite editor, and the
[intercat and infinicat tools](https://github.com/avieth/intercat/):

```sh
# Run the JACK client, taking input from a fifo
mkfifo fifo_backend
algorave < fifo_backend

# Open another shell

# Run the Haskell frontend, taking input from another fifo, and dumping output
# to algorave via the backend fifo.
# infinicat ensures that stdin does not close when the last fifo writer
# vanishes.
mkfifo fifo_frontend
infinicat fifo_frontend | frontend > fifo_backend

# Open another shell

touch source_file.txt
# Go open an editor somewhere.

# Use the interactive cat tool to control when the source file should be
# written, causing the frontend to update and, if desired, output the new
# system program to the JACK client.
intercat
> add lazy source source_file.txt
> add lazy sink fifo_frontend
> commit
```
