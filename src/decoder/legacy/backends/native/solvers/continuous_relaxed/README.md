# Continuous Relaxed Engine

This directory is reserved for componentized pieces of the continuous-relaxed
native solver. The initial implementation lives in `../continuous_relaxed.cpp`
so the bridge and build wiring can land as one cohesive engine; future hot-path
components should move here as they stabilize.
