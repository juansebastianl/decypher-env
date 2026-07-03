"""Legacy research toolkit (internals; not part of the RL environment).

This package holds the original "hand-write a solver and run a sampler"
codebase that predates the RL environment: the Python/native sampler engines
(``backends``), the continuous/discrete ``relaxations``, the ``XtsSampler``
(``sampler``), and the command-line entry points (``cli``, ``sample_blocks``).

It is kept working and its native engines double as *reference strategies*
documented under ``examples/solvers/{parallel_tempering,continuous_relaxed,
algebraic_relaxed}``. If you are here to train a model, you do not need any of
it -- start at the top-level README and ``src/decoder/rl``.
"""
