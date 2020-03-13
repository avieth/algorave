# Interactive setup

```sh
# Shell 1
cd /common/working/directory
mkfifo fifo_backend
algorave < infinicat fifo_backend
```

```sh
# Shell 2
cd /common/working/directory
mkfifo fifo_frontend
infinicat fifo_frontend | frontend > fifo_backend
```

```sh
# Shell 3
cd /common/working/directory
interface
> add lazy source concrete_source
> add lazy sink fifo_frontend
```

Then open an editor on `concrete_source`.
When changes from that edited file are to be commited, write it and then
use `commit` in the `intercat` shell.
