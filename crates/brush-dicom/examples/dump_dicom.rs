//! Quick metadata dump for a DICOM file (used to inspect RXA_brain.dcm).
use brush_dicom::{parse_dicom, extract_pixel_data};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: dump_dicom <file.dcm>");
    let bytes = std::fs::read(&path)?;
    let meta = parse_dicom(&bytes)?;
    println!("num_frames: {}", meta.num_frames);
    println!("geometry: {:#?}", meta.geometry);
    println!("fps: {}", meta.fps);
    let n = meta.frames.len();
    println!("frames: {n}");
    if n > 0 {
        println!("frame[0]: {:#?}", meta.frames[0]);
        println!("frame[last]: {:#?}", meta.frames[n - 1]);
        println!(
            "alpha range: [{:.2}, {:.2}] deg",
            meta.frames.iter().map(|f| f.alpha_degree).fold(f64::INFINITY, f64::min),
            meta.frames.iter().map(|f| f.alpha_degree).fold(f64::NEG_INFINITY, f64::max),
        );
        println!(
            "beta range: [{:.2}, {:.2}] deg",
            meta.frames.iter().map(|f| f.beta_degree).fold(f64::INFINITY, f64::min),
            meta.frames.iter().map(|f| f.beta_degree).fold(f64::NEG_INFINITY, f64::max),
        );
        // Phase presence.
        let phases: Vec<f64> = meta.phase_array();
        let distinct = phases.iter().fold(0.0f64, |acc, p| acc + (*p - 0.5).abs());
        println!("phase present: {}, sum|p-0.5| = {:.3}", !phases.is_empty(), distinct);
        if !phases.is_empty() {
            println!("phase[0..8] = {:?}", &phases[..phases.len().min(8)]);
        }
    }
    let pix = extract_pixel_data(&bytes)?;
    println!(
        "pixels: {} frames x {}x{} (f32 bytes {})",
        pix.num_frames,
        pix.height,
        pix.width,
        pix.values.len() * 4
    );
    Ok(())
}
