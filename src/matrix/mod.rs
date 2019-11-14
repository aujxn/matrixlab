/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/// Dense matricies
pub mod dense;

/// Sparse matricies
pub mod sparse;

/// Sparse matrix iterators
pub mod sparse_matrix_iter;

/// General Minimal Residual Method (Krylov) solver w/QR decomposition (Arnoldi's)
pub mod gmres;
