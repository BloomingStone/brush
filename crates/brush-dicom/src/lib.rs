//! DICOM header + pixel parsing for X-ray (rotational DSA / XA) reconstruction.
//!
//! Targets the multi-frame XA DICOM produced by
//! `local.properties/refs/convert_tif_to_dicom.py`:
//!
//! - C-arm geometry: `DistanceSourceToDetector`, `DistanceSourceToPatient`,
//!   `ImagerPixelSpacing`, `Rows` / `Columns`
//! - Frame timing: `CineRate` / `RecommendedDisplayFrameRate`,
//!   `FrameTimeVector`
//! - Per-frame positioner angles: `PositionerPrimaryAngle(Increment)` /
//!   `PositionerSecondaryAngle(Increment)`
//! - Per-frame cardiac phase: private tag `(0071, 1010)` (creator
//!   `YOUR_INSTITUTION_PHASE_1.0` at `(0071, 0010)`)
//! - Multi-frame `uint16` MONOCHROME2 pixel data
//!
//! The parser reads from an in-memory byte buffer (so it works with
//! [`brush_vfs::BrushVfs`] sources — paths, URLs or zips). Pixel data is
//! uncompressed and decoded manually from the raw `PixelData` bytes.

pub mod meta;
pub mod parser;
pub mod pixel;

pub use meta::{CArmGeometry, DicomMeta, FrameInfo};
pub use parser::{DicomError, parse_dicom};
pub use pixel::{PixelData, extract_pixel_data, normalize_01};
