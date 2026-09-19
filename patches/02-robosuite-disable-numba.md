# Patch: disable numba in robosuite for multiprocess LIBERO eval

Root cause: the client process imports robosuite (numba JIT pool initialized),
then forks simulation workers; the corrupted numba state segfaults workers in C
layer (no Python traceback, no worker error log).

Fix in `<venv>/lib/python3.10/site-packages/robosuite/macros_private.py`:

```diff
-ENABLE_NUMBA = True
-CACHE_NUMBA = True
+ENABLE_NUMBA = False
+CACHE_NUMBA = False
```

Also set `NUMBA_DISABLE_JIT=1` when launching eval (belt and braces).
Reference: robosuite source itself notes numba breaks offscreen rendering
deterministically for some tasks.
