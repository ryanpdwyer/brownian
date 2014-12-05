# Testing
- [ ] Integration tests with simulated cantilever data
- [x] Travis CI integration

# Code Structure
- [ ] Move helper functions to their own file (from `__init__.py`)
- [x] High level plotting tool (moved to its own project)

Notes: The high level plotting tool should be some class like `Plotter`, with data being stored in some subclass (just linking to the hdf5 file would probably be fine, although syntactic sugar for common tasks would be nice).

# Features
- [ ] Command line script to fit hdf5 file, output a nice text report.
