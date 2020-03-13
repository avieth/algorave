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

Here's an example of what you could write to the frontend.

```
BEGIN

# The BEGIN is part of the metalanguage. Everything between it and END is
# an assembly-like concrete syntax that is just the rust program's syntax
# but with labels for jumps and branches.

# This program produces a sine wave at 440Hz.
# It's a good demonstration of the basics of the low-level language.

# The goal is to produce an argument to sin. It'll be
#   2 * PI * 440 * t
# where t is in [0,1] and goes from 0 to 1 every second.
# To get that, we can safely use the 64-bit "current frame" and
# the 32-bit sample rate, like so:
set now 0x00;
set rate 0x08;
# Fits in 32 bits since rate is u32.
mod u64 0x00 0x08 0x10;

# 0x10 is now in [0, sample_rate-1] and has exactly the properties
# we want in t, once we divide it by its maximal value.

# Convert to f32 so we can divide: first the modulus we just
# computed and also the sample rate itself.
itof i32 f32 0x10 0x14;
itof i32 f32 0x08 0x18;

# Now we have our t value here. Overwrite 0x00 because we don't
# need that anymore.
div f32 0x14 0x18 0x00;

# Give 2*PI as a literal.
set f32:6.283185307 0x04;
# 440Hz.
set f32:440.0 0x08;
mul f32 0x00 0x04 0x00;
mul f32 0x00 0x08 0x00;
sin f32 0x00 0x04;

# Write to both outputs. We assume there are 2, with id 0 and 1.
# Writing is done by giving an address which contains the output id
# and one which contains the offset into the region, then an address
# containing the value (0x04 in this case).
set u8:0x00 0x10;
set u32:0x00000000 0x11;
write f32 0x10 0x11 0x04;
# Here we set the output id to 1 and do the same thing again.
set u8:0x01 0x10;
write f32 0x10 0x11 0x04;

# The rust program won't stop unless you tell it to.
# Not writing this would cause xruns for sure.
stop;

# Signal the end of the program, and then tell the frontend to commit the
# changes.
END
COMMIT
```
