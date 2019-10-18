# matrixlab

matrixlab is a Rust library for working with both sparse and dense
matricies. The library is fully functional, though a little clunky.
This was a school project and is fairly underdeveloped, but has all the basics
you need to do matrix operations.

There's a smattering of documentation in rustdoc, but there's still more
work to be done here as well. Hopefully a combination of the documentation
and source code is enough to get you started!

Currently solving matricies via GMRES, backsolving, and least squares
is possible. Both sparse and normal matricies are supported and matrix
operations use rayon to do everything in parallel.
