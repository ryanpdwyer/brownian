# Testing
- [ ] Integration tests with simulated cantilever data
- [ ] Travis CI integration

# Code Structure
- [ ] Move helper functions to their own file (from `__init__.py`)
- [ ] High level plotting tool

Notes: The high level plotting tool should be some class like `Plotter`, with data being stored in some subclass (just linking to the hdf5 file would probably be fine, although syntactic sugar for common tasks would be nice).

# Data Handling
- [ ] Deal with old file formats?