use ab_glyph::{point, Font, Glyph, Point, PxScale, ScaleFont};
use conv::ValueInto;
use image::{
    Enlargeable, GenericImage, GenericImageView, GrayImage, ImageBuffer, Luma, LumaA, Pixel,
    Primitive, Rgb, Rgba,
};
use num::Num;

use std::cmp::{max, min};

/// Helper for a conversion that we know can't fail.
pub fn cast<T, U>(x: T) -> U
where
    T: ValueInto<U>,
{
    match x.value_into() {
        Ok(y) => y,
        Err(_) => panic!("Failed to convert"),
    }
}

pub type Image<P> = ImageBuffer<P, Vec<<P as Pixel>::Subpixel>>;

/// Pixels which have a named Black value.
pub trait HasBlack {
    /// Returns a black pixel of this type.
    fn black() -> Self;
}

/// Pixels which have a named White value.
pub trait HasWhite {
    /// Returns a white pixel of this type.
    fn white() -> Self;
}

macro_rules! impl_black_white {
    ($for_:ty, $min:expr, $max:expr) => {
        impl HasBlack for $for_ {
            fn black() -> Self {
                $min
            }
        }

        impl HasWhite for $for_ {
            fn white() -> Self {
                $max
            }
        }
    };
}

impl_black_white!(Luma<u8>, Luma([u8::MIN]), Luma([u8::MAX]));
impl_black_white!(Luma<u16>, Luma([u16::MIN]), Luma([u16::MAX]));
impl_black_white!(
    LumaA<u8>,
    LumaA([u8::MIN, u8::MAX]),
    LumaA([u8::MAX, u8::MAX])
);
impl_black_white!(
    LumaA<u16>,
    LumaA([u16::MIN, u16::MAX]),
    LumaA([u16::MAX, u16::MAX])
);
impl_black_white!(Rgb<u8>, Rgb([u8::MIN; 3]), Rgb([u8::MAX; 3]));
impl_black_white!(Rgb<u16>, Rgb([u16::MIN; 3]), Rgb([u16::MAX; 3]));
impl_black_white!(
    Rgba<u8>,
    Rgba([u8::MIN, u8::MIN, u8::MIN, u8::MAX]),
    Rgba([u8::MAX, u8::MAX, u8::MAX, u8::MAX])
);
impl_black_white!(
    Rgba<u16>,
    Rgba([u16::MIN, u16::MIN, u16::MIN, u16::MAX]),
    Rgba([u16::MAX, u16::MAX, u16::MAX, u16::MAX])
);

/// Something with a 2d position.
pub trait Position {
    /// x-coordinate.
    fn x(&self) -> u32;
    /// y-coordinate.
    fn y(&self) -> u32;
}

/// Something with a score.
pub trait Score {
    /// Score of this item.
    fn score(&self) -> f32;
}

/// A type to which we can clamp a value of type T.
/// Implementations are not required to handle `NaN`s gracefully.
pub trait Clamp<T> {
    /// Clamp `x` to a valid value for this type.
    fn clamp(x: T) -> Self;
}

/// Creates an implementation of Clamp<From> for type To.
macro_rules! implement_clamp {
    ($from:ty, $to:ty, $min:expr, $max:expr, $min_from:expr, $max_from:expr) => {
        impl Clamp<$from> for $to {
            fn clamp(x: $from) -> $to {
                if x < $max_from as $from {
                    if x > $min_from as $from {
                        x as $to
                    } else {
                        $min
                    }
                } else {
                    $max
                }
            }
        }
    };
}

/// Implements Clamp<T> for T, for all input types T.
macro_rules! implement_identity_clamp {
    ( $($t:ty),* ) => {
        $(
            impl Clamp<$t> for $t {
                fn clamp(x: $t) -> $t { x }
            }
        )*
    };
}

implement_clamp!(i16, u8, u8::MIN, u8::MAX, u8::MIN as i16, u8::MAX as i16);
implement_clamp!(u16, u8, u8::MIN, u8::MAX, u8::MIN as u16, u8::MAX as u16);
implement_clamp!(i32, u8, u8::MIN, u8::MAX, u8::MIN as i32, u8::MAX as i32);
implement_clamp!(u32, u8, u8::MIN, u8::MAX, u8::MIN as u32, u8::MAX as u32);
implement_clamp!(f32, u8, u8::MIN, u8::MAX, u8::MIN as f32, u8::MAX as f32);
implement_clamp!(f64, u8, u8::MIN, u8::MAX, u8::MIN as f64, u8::MAX as f64);

implement_clamp!(
    i32,
    u16,
    u16::MIN,
    u16::MAX,
    u16::MIN as i32,
    u16::MAX as i32
);
implement_clamp!(
    f32,
    u16,
    u16::MIN,
    u16::MAX,
    u16::MIN as f32,
    u16::MAX as f32
);
implement_clamp!(
    f64,
    u16,
    u16::MIN,
    u16::MAX,
    u16::MIN as f64,
    u16::MAX as f64
);

implement_clamp!(
    i32,
    i16,
    i16::MIN,
    i16::MAX,
    i16::MIN as i32,
    i16::MAX as i32
);

implement_identity_clamp!(u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);

/// Runs the canny edge detection algorithm.
///
/// Returns a binary image where edge pixels have a value of 255
///  and non-edge pixels a value of 0.
///
/// # Params
///
/// - `low_threshold`: Low threshold for the hysteresis procedure.
/// Edges with a strength higher than the low threshold will appear
/// in the output image, if there are strong edges nearby.
/// - `high_threshold`: High threshold for the hysteresis procedure.
/// Edges with a strength higher than the high threshold will always
/// appear as edges in the output image.
///
/// The greatest possible edge strength (and so largest sensible threshold)
/// is`sqrt(5) * 2 * 255`, or approximately 1140.39.
///
/// This odd looking value is the result of using a standard
/// definition of edge strength: the strength of an edge at a point `p` is
/// defined to be `sqrt(dx^2 + dy^2)`, where `dx` and `dy` are the values
/// of the horizontal and vertical Sobel gradients at `p`.
pub fn canny(image: &GrayImage, low_threshold: f32, high_threshold: f32) -> GrayImage {
    assert!(high_threshold >= low_threshold);
    // Heavily based on the implementation proposed by wikipedia.
    // 1. Gaussian blur.
    const SIGMA: f32 = 1.4;
    let blurred = gaussian_blur_f32(image, SIGMA);

    // 2. Intensity of gradients.
    let gx = horizontal_sobel(&blurred);
    let gy = vertical_sobel(&blurred);
    let g: Vec<f32> = gx
        .iter()
        .zip(gy.iter())
        .map(|(h, v)| (*h as f32).hypot(*v as f32))
        .collect::<Vec<f32>>();

    let g = ImageBuffer::from_raw(image.width(), image.height(), g).unwrap();

    // 3. Non-maximum-suppression (Make edges thinner)
    let thinned = non_maximum_suppression(&g, &gx, &gy);

    // 4. Hysteresis to filter out edges based on thresholds.
    hysteresis(&thinned, low_threshold, high_threshold)
}

/// Finds local maxima to make the edges thinner.
fn non_maximum_suppression(
    g: &ImageBuffer<Luma<f32>, Vec<f32>>,
    gx: &ImageBuffer<Luma<i16>, Vec<i16>>,
    gy: &ImageBuffer<Luma<i16>, Vec<i16>>,
) -> ImageBuffer<Luma<f32>, Vec<f32>> {
    const RADIANS_TO_DEGREES: f32 = 180f32 / std::f32::consts::PI;
    let mut out = ImageBuffer::from_pixel(g.width(), g.height(), Luma([0.0]));
    for y in 1..g.height() - 1 {
        for x in 1..g.width() - 1 {
            let x_gradient = gx[(x, y)][0] as f32;
            let y_gradient = gy[(x, y)][0] as f32;
            let mut angle = (y_gradient).atan2(x_gradient) * RADIANS_TO_DEGREES;
            if angle < 0.0 {
                angle += 180.0
            }
            // Clamp angle.
            let clamped_angle = if !(22.5..157.5).contains(&angle) {
                0
            } else if (22.5..67.5).contains(&angle) {
                45
            } else if (67.5..112.5).contains(&angle) {
                90
            } else if (112.5..157.5).contains(&angle) {
                135
            } else {
                unreachable!()
            };

            // Get the two perpendicular neighbors.
            let (cmp1, cmp2) = unsafe {
                match clamped_angle {
                    0 => (g.unsafe_get_pixel(x - 1, y), g.unsafe_get_pixel(x + 1, y)),
                    45 => (
                        g.unsafe_get_pixel(x + 1, y + 1),
                        g.unsafe_get_pixel(x - 1, y - 1),
                    ),
                    90 => (g.unsafe_get_pixel(x, y - 1), g.unsafe_get_pixel(x, y + 1)),
                    135 => (
                        g.unsafe_get_pixel(x - 1, y + 1),
                        g.unsafe_get_pixel(x + 1, y - 1),
                    ),
                    _ => unreachable!(),
                }
            };
            let pixel = *g.get_pixel(x, y);
            // If the pixel is not a local maximum, suppress it.
            if pixel[0] < cmp1[0] || pixel[0] < cmp2[0] {
                out.put_pixel(x, y, Luma([0.0]));
            } else {
                out.put_pixel(x, y, pixel);
            }
        }
    }
    out
}

/// Filter out edges with the thresholds.
/// Non-recursive breadth-first search.
fn hysteresis(
    input: &ImageBuffer<Luma<f32>, Vec<f32>>,
    low_thresh: f32,
    high_thresh: f32,
) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let max_brightness = Luma::white();
    let min_brightness = Luma::black();
    // Init output image as all black.
    let mut out = ImageBuffer::from_pixel(input.width(), input.height(), min_brightness);
    // Stack. Possible optimization: Use previously allocated memory, i.e. gx.
    let mut edges = Vec::with_capacity(((input.width() * input.height()) / 2) as usize);
    for y in 1..input.height() - 1 {
        for x in 1..input.width() - 1 {
            let inp_pix = *input.get_pixel(x, y);
            let out_pix = *out.get_pixel(x, y);
            // If the edge strength is higher than high_thresh, mark it as an edge.
            if inp_pix[0] >= high_thresh && out_pix[0] == 0 {
                out.put_pixel(x, y, max_brightness);
                edges.push((x, y));
                // Track neighbors until no neighbor is >= low_thresh.
                while !edges.is_empty() {
                    let (nx, ny) = edges.pop().unwrap();
                    let neighbor_indices = [
                        (nx + 1, ny),
                        (nx + 1, ny + 1),
                        (nx, ny + 1),
                        (nx - 1, ny - 1),
                        (nx - 1, ny),
                        (nx - 1, ny + 1),
                    ];

                    for neighbor_idx in &neighbor_indices {
                        let in_neighbor = *input.get_pixel(neighbor_idx.0, neighbor_idx.1);
                        let out_neighbor = *out.get_pixel(neighbor_idx.0, neighbor_idx.1);
                        if in_neighbor[0] >= low_thresh && out_neighbor[0] == 0 {
                            out.put_pixel(neighbor_idx.0, neighbor_idx.1, max_brightness);
                            edges.push((neighbor_idx.0, neighbor_idx.1));
                        }
                    }
                }
            }
        }
    }
    out
}

/// Blurs an image using a Gaussian of standard deviation sigma.
/// The kernel used has type f32 and all intermediate calculations are performed
/// at this type.
///
/// # Panics
///
/// Panics if `sigma <= 0.0`.
// TODO: Integer type kernel, approximations via repeated box filter.
pub fn gaussian_blur_f32<P>(image: &Image<P>, sigma: f32) -> Image<P>
where
    P: Pixel + 'static,
    <P as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>,
{
    assert!(sigma > 0.0, "sigma must be > 0.0");
    let kernel = gaussian_kernel_f32(sigma);
    separable_filter_equal(image, &kernel)
}

#[inline]
fn gaussian(x: f32, r: f32) -> f32 {
    ((2.0 * std::f32::consts::PI).sqrt() * r).recip() * (-x.powi(2) / (2.0 * r.powi(2))).exp()
}

/// Construct a one dimensional float-valued kernel for performing a Gaussian blur
/// with standard deviation sigma.
fn gaussian_kernel_f32(sigma: f32) -> Vec<f32> {
    let kernel_radius = (2.0 * sigma).ceil() as usize;
    let mut kernel_data = vec![0.0; 2 * kernel_radius + 1];
    for i in 0..kernel_radius + 1 {
        let value = gaussian(i as f32, sigma);
        kernel_data[kernel_radius + i] = value;
        kernel_data[kernel_radius - i] = value;
    }
    kernel_data
}

/// Returns 2d correlation of an image with the outer product of the 1d
/// kernel filter with itself.
pub fn separable_filter_equal<P, K>(image: &Image<P>, kernel: &[K]) -> Image<P>
where
    P: Pixel + 'static,
    <P as Pixel>::Subpixel: ValueInto<K> + Clamp<K>,
    K: Num + Copy,
{
    separable_filter(image, kernel, kernel)
}

/// Returns 2d correlation of view with the outer product of the 1d
/// kernels `h_kernel` and `v_kernel`.
pub fn separable_filter<P, K>(image: &Image<P>, h_kernel: &[K], v_kernel: &[K]) -> Image<P>
where
    P: Pixel + 'static,
    <P as Pixel>::Subpixel: ValueInto<K> + Clamp<K>,
    K: Num + Copy,
{
    let h = horizontal_filter(image, h_kernel);
    vertical_filter(&h, v_kernel)
}

/// Returns horizontal correlations between an image and a 1d kernel.
/// Pads by continuity. Intermediate calculations are performed at
/// type K.
pub fn horizontal_filter<P, K>(image: &Image<P>, kernel: &[K]) -> Image<P>
where
    P: Pixel + 'static,
    <P as Pixel>::Subpixel: ValueInto<K> + Clamp<K>,
    K: Num + Copy,
{
    // Don't replace this with a call to Kernel::filter without
    // checking the benchmark results. At the time of writing this
    // specialised implementation is faster.
    let (width, height) = image.dimensions();
    let mut out = Image::<P>::new(width, height);
    let zero = K::zero();
    let mut acc = vec![zero; P::CHANNEL_COUNT as usize];
    let k_width = kernel.len() as i32;

    // Typically the image side will be much larger than the kernel length.
    // In that case we can remove a lot of bounds checks for most pixels.
    if k_width >= width as i32 {
        for y in 0..height {
            for x in 0..width {
                for (i, k) in kernel.iter().enumerate() {
                    let x_unchecked = (x as i32) + i as i32 - k_width / 2;
                    let x_p = max(0, min(x_unchecked, width as i32 - 1)) as u32;
                    let p = unsafe { image.unsafe_get_pixel(x_p, y) };
                    accumulate(&mut acc, &p, *k);
                }

                let out_channels = out.get_pixel_mut(x, y).channels_mut();
                for (a, c) in acc.iter_mut().zip(out_channels.iter_mut()) {
                    *c = <P as Pixel>::Subpixel::clamp(*a);
                    *a = zero;
                }
            }
        }

        return out;
    }

    let half_k = k_width / 2;

    for y in 0..height {
        // Left margin - need to check lower bound only
        for x in 0..half_k {
            for (i, k) in kernel.iter().enumerate() {
                let x_unchecked = (x as i32) + i as i32 - k_width / 2;
                let x_p = max(0, x_unchecked) as u32;
                let p = unsafe { image.unsafe_get_pixel(x_p, y) };
                accumulate(&mut acc, &p, *k);
            }

            let out_channels = out.get_pixel_mut(x as u32, y).channels_mut();
            for (a, c) in acc.iter_mut().zip(out_channels.iter_mut()) {
                *c = <P as Pixel>::Subpixel::clamp(*a);
                *a = zero;
            }
        }

        // Neither margin - don't need bounds check on either side
        for x in half_k..(width as i32 - half_k) {
            for (i, k) in kernel.iter().enumerate() {
                let x_unchecked = (x as i32) + i as i32 - k_width / 2;
                let x_p = x_unchecked as u32;
                let p = unsafe { image.unsafe_get_pixel(x_p, y) };
                accumulate(&mut acc, &p, *k);
            }

            let out_channels = out.get_pixel_mut(x as u32, y).channels_mut();
            for (a, c) in acc.iter_mut().zip(out_channels.iter_mut()) {
                *c = <P as Pixel>::Subpixel::clamp(*a);
                *a = zero;
            }
        }

        // Right margin - need to check upper bound only
        for x in (width as i32 - half_k)..(width as i32) {
            for (i, k) in kernel.iter().enumerate() {
                let x_unchecked = (x as i32) + i as i32 - k_width / 2;
                let x_p = min(x_unchecked, width as i32 - 1) as u32;
                let p = unsafe { image.unsafe_get_pixel(x_p, y) };
                accumulate(&mut acc, &p, *k);
            }

            let out_channels = out.get_pixel_mut(x as u32, y).channels_mut();
            for (a, c) in acc.iter_mut().zip(out_channels.iter_mut()) {
                *c = <P as Pixel>::Subpixel::clamp(*a);
                *a = zero;
            }
        }
    }

    out
}

/// Returns horizontal correlations between an image and a 1d kernel.
/// Pads by continuity.
pub fn vertical_filter<P, K>(image: &Image<P>, kernel: &[K]) -> Image<P>
where
    P: Pixel + 'static,
    <P as Pixel>::Subpixel: ValueInto<K> + Clamp<K>,
    K: Num + Copy,
{
    // Don't replace this with a call to Kernel::filter without
    // checking the benchmark results. At the time of writing this
    // specialised implementation is faster.
    let (width, height) = image.dimensions();
    let mut out = Image::<P>::new(width, height);
    let zero = K::zero();
    let mut acc = vec![zero; P::CHANNEL_COUNT as usize];
    let k_height = kernel.len() as i32;

    // Typically the image side will be much larger than the kernel length.
    // In that case we can remove a lot of bounds checks for most pixels.
    if k_height >= height as i32 {
        for y in 0..height {
            for x in 0..width {
                for (i, k) in kernel.iter().enumerate() {
                    let y_unchecked = (y as i32) + i as i32 - k_height / 2;
                    let y_p = max(0, min(y_unchecked, height as i32 - 1)) as u32;
                    let p = unsafe { image.unsafe_get_pixel(x, y_p) };
                    accumulate(&mut acc, &p, *k);
                }

                let out_channels = out.get_pixel_mut(x, y).channels_mut();
                for (a, c) in acc.iter_mut().zip(out_channels.iter_mut()) {
                    *c = <P as Pixel>::Subpixel::clamp(*a);
                    *a = zero;
                }
            }
        }

        return out;
    }

    let half_k = k_height / 2;

    // Top margin - need to check lower bound only
    for y in 0..half_k {
        for x in 0..width {
            for (i, k) in kernel.iter().enumerate() {
                let y_unchecked = (y as i32) + i as i32 - k_height / 2;
                let y_p = max(0, y_unchecked) as u32;
                let p = unsafe { image.unsafe_get_pixel(x, y_p) };
                accumulate(&mut acc, &p, *k);
            }

            let out_channels = out.get_pixel_mut(x, y as u32).channels_mut();
            for (a, c) in acc.iter_mut().zip(out_channels.iter_mut()) {
                *c = <P as Pixel>::Subpixel::clamp(*a);
                *a = zero;
            }
        }
    }

    // Neither margin - don't need bounds check on either side
    for y in half_k..(height as i32 - half_k) {
        for x in 0..width {
            for (i, k) in kernel.iter().enumerate() {
                let y_unchecked = (y as i32) + i as i32 - k_height / 2;
                let y_p = y_unchecked as u32;
                let p = unsafe { image.unsafe_get_pixel(x, y_p) };
                accumulate(&mut acc, &p, *k);
            }

            let out_channels = out.get_pixel_mut(x, y as u32).channels_mut();
            for (a, c) in acc.iter_mut().zip(out_channels.iter_mut()) {
                *c = <P as Pixel>::Subpixel::clamp(*a);
                *a = zero;
            }
        }
    }

    // Right margin - need to check upper bound only
    for y in (height as i32 - half_k)..(height as i32) {
        for x in 0..width {
            for (i, k) in kernel.iter().enumerate() {
                let y_unchecked = (y as i32) + i as i32 - k_height / 2;
                let y_p = min(y_unchecked, height as i32 - 1) as u32;
                let p = unsafe { image.unsafe_get_pixel(x, y_p) };
                accumulate(&mut acc, &p, *k);
            }

            let out_channels = out.get_pixel_mut(x, y as u32).channels_mut();
            for (a, c) in acc.iter_mut().zip(out_channels.iter_mut()) {
                *c = <P as Pixel>::Subpixel::clamp(*a);
                *a = zero;
            }
        }
    }

    out
}

fn accumulate<P, K>(acc: &mut [K], pixel: &P, weight: K)
where
    P: Pixel,
    <P as Pixel>::Subpixel: ValueInto<K>,
    K: Num + Copy,
{
    for i in 0..(P::CHANNEL_COUNT as usize) {
        acc[i as usize] = acc[i as usize] + cast(pixel.channels()[i]) * weight;
    }
}

/// A 2D kernel, used to filter images via convolution.
pub struct Kernel<'a, K> {
    data: &'a [K],
    width: u32,
    height: u32,
}

impl<'a, K: Num + Copy + 'a> Kernel<'a, K> {
    /// Construct a kernel from a slice and its dimensions. The input slice is
    /// in row-major form.
    pub fn new(data: &'a [K], width: u32, height: u32) -> Kernel<'a, K> {
        assert!(width > 0 && height > 0, "width and height must be non-zero");
        assert!(
            width * height == data.len() as u32,
            "Invalid kernel len: expecting {}, found {}",
            width * height,
            data.len()
        );
        Kernel {
            data,
            width,
            height,
        }
    }

    /// Returns 2d correlation of an image. Intermediate calculations are performed
    /// at type K, and the results converted to pixel Q via f. Pads by continuity.
    pub fn filter<P, F, Q>(&self, image: &Image<P>, mut f: F) -> Image<Q>
    where
        P: Pixel + 'static,
        <P as Pixel>::Subpixel: ValueInto<K>,
        Q: Pixel + 'static,
        F: FnMut(&mut Q::Subpixel, K),
    {
        let (width, height) = image.dimensions();
        let mut out = Image::<Q>::new(width, height);
        let num_channels = P::CHANNEL_COUNT as usize;
        let zero = K::zero();
        let mut acc = vec![zero; num_channels];
        let (k_width, k_height) = (self.width as i64, self.height as i64);
        let (width, height) = (width as i64, height as i64);

        for y in 0..height {
            for x in 0..width {
                for k_y in 0..k_height {
                    let y_p = min(height - 1, max(0, y + k_y - k_height / 2)) as u32;
                    for k_x in 0..k_width {
                        let x_p = min(width - 1, max(0, x + k_x - k_width / 2)) as u32;
                        accumulate(
                            &mut acc,
                            unsafe { &image.unsafe_get_pixel(x_p, y_p) },
                            unsafe { *self.data.get_unchecked((k_y * k_width + k_x) as usize) },
                        );
                    }
                }
                let out_channels = out.get_pixel_mut(x as u32, y as u32).channels_mut();
                for (a, c) in acc.iter_mut().zip(out_channels.iter_mut()) {
                    f(c, *a);
                    *a = zero;
                }
            }
        }

        out
    }
}

/// Sobel filter for detecting vertical gradients.
///
/// Used by the [`vertical_sobel`](fn.vertical_sobel.html) function.
#[rustfmt::skip]
pub static VERTICAL_SOBEL: [i32; 9] = [
    -1, -2, -1,
     0,  0,  0,
     1,  2,  1];

/// Sobel filter for detecting horizontal gradients.
///
/// Used by the [`horizontal_sobel`](fn.horizontal_sobel.html) function.
#[rustfmt::skip]
pub static HORIZONTAL_SOBEL: [i32; 9] = [
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1];

/// Convolves an image with the [`HORIZONTAL_SOBEL`](static.HORIZONTAL_SOBEL.html)
/// kernel to detect horizontal gradients.
pub fn horizontal_sobel(image: &GrayImage) -> Image<Luma<i16>> {
    filter3x3(image, &HORIZONTAL_SOBEL)
}

/// Convolves an image with the [`VERTICAL_SOBEL`](static.VERTICAL_SOBEL.html)
/// kernel to detect vertical gradients.
pub fn vertical_sobel(image: &GrayImage) -> Image<Luma<i16>> {
    filter3x3(image, &VERTICAL_SOBEL)
}

/// Returns 2d correlation of an image with a 3x3 row-major kernel. Intermediate calculations are
/// performed at type K, and the results clamped to subpixel type S. Pads by continuity.
pub fn filter3x3<P, K, S>(image: &Image<P>, kernel: &[K]) -> Image<ChannelMap<P, S>>
where
    P::Subpixel: ValueInto<K>,
    S: Clamp<K> + Primitive + 'static,
    P: WithChannel<S> + 'static,
    K: Num + Copy,
{
    let kernel = Kernel::new(kernel, 3, 3);
    kernel.filter(image, |channel, acc| *channel = S::clamp(acc))
}

/// The type obtained by replacing the channel type of a given `Pixel` type.
/// The output type must have the same name of channels as the input type, or
/// several algorithms will produce incorrect results or panic.
pub trait WithChannel<C: Primitive>: Pixel {
    /// The new pixel type.
    type Pixel: Pixel<Subpixel = C> + 'static;
}

/// Alias to make uses of `WithChannel` less syntactically noisy.
pub type ChannelMap<Pix, Sub> = <Pix as WithChannel<Sub>>::Pixel;

impl<T, U> WithChannel<U> for Rgb<T>
where
    T: Primitive + Enlargeable + 'static,
    U: Primitive + Enlargeable + 'static,
{
    type Pixel = Rgb<U>;
}

impl<T, U> WithChannel<U> for Rgba<T>
where
    T: Primitive + Enlargeable + 'static,
    U: Primitive + Enlargeable + 'static,
{
    type Pixel = Rgba<U>;
}

impl<T, U> WithChannel<U> for Luma<T>
where
    T: Primitive + 'static,
    U: Primitive + 'static,
{
    type Pixel = Luma<U>;
}

impl<T, U> WithChannel<U> for LumaA<T>
where
    T: Primitive + 'static,
    U: Primitive + 'static,
{
    type Pixel = LumaA<U>;
}

/// A surface for drawing on - many drawing functions in this
/// library are generic over a `Canvas` to allow the user to
/// configure e.g. whether to use blending.
///
/// All instances of `GenericImage` implement `Canvas`, with
/// the behaviour of `draw_pixel` being equivalent to calling
/// `set_pixel` with the same arguments.
///
/// See [`Blend`](struct.Blend.html) for another example implementation
/// of this trait - its implementation of `draw_pixel` alpha-blends
/// the input value with the pixel's current value.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use image::{Pixel, Rgba, RgbaImage};
/// use imageproc::drawing::{Canvas, Blend};
///
/// // A trivial function which draws on a Canvas
/// fn write_a_pixel<C: Canvas>(canvas: &mut C, c: C::Pixel) {
///     canvas.draw_pixel(0, 0, c);
/// }
///
/// // Background color
/// let solid_blue = Rgba([0u8, 0u8, 255u8, 255u8]);
///
/// // Drawing color
/// let translucent_red = Rgba([255u8, 0u8, 0u8, 127u8]);
///
/// // Blended combination of background and drawing colors
/// let mut alpha_blended = solid_blue;
/// alpha_blended.blend(&translucent_red);
///
/// // The implementation of Canvas for GenericImage overwrites existing pixels
/// let mut image = RgbaImage::from_pixel(1, 1, solid_blue);
/// write_a_pixel(&mut image, translucent_red);
/// assert_eq!(*image.get_pixel(0, 0), translucent_red);
///
/// // This behaviour can be customised by using a different Canvas type
/// let mut image = Blend(RgbaImage::from_pixel(1, 1, solid_blue));
/// write_a_pixel(&mut image, translucent_red);
/// assert_eq!(*image.0.get_pixel(0, 0), alpha_blended);
/// # }
/// ```
pub trait Canvas {
    /// The type of `Pixel` that can be drawn on this canvas.
    type Pixel: Pixel;

    /// The width and height of this canvas.
    fn dimensions(&self) -> (u32, u32);

    /// The width of this canvas.
    fn width(&self) -> u32 {
        self.dimensions().0
    }

    /// The height of this canvas.
    fn height(&self) -> u32 {
        self.dimensions().1
    }

    /// Returns the pixel located at (x, y).
    fn get_pixel(&self, x: u32, y: u32) -> Self::Pixel;

    /// Draw a pixel at the given coordinates. `x` and `y`
    /// should be within `dimensions` - if not then panicking
    /// is a valid implementation behaviour.
    fn draw_pixel(&mut self, x: u32, y: u32, color: Self::Pixel);
}

impl<I> Canvas for I
where
    I: GenericImage,
{
    type Pixel = I::Pixel;

    fn dimensions(&self) -> (u32, u32) {
        <I as GenericImageView>::dimensions(self)
    }

    fn get_pixel(&self, x: u32, y: u32) -> Self::Pixel {
        self.get_pixel(x, y)
    }

    fn draw_pixel(&mut self, x: u32, y: u32, color: Self::Pixel) {
        self.put_pixel(x, y, color)
    }
}

/// A canvas that blends pixels when drawing.
///
/// See the documentation for [`Canvas`](trait.Canvas.html)
/// for an example using this type.
pub struct Blend<I>(pub I);

impl<I: GenericImage> Canvas for Blend<I> {
    type Pixel = I::Pixel;

    fn dimensions(&self) -> (u32, u32) {
        self.0.dimensions()
    }

    fn get_pixel(&self, x: u32, y: u32) -> Self::Pixel {
        self.0.get_pixel(x, y)
    }

    fn draw_pixel(&mut self, x: u32, y: u32, color: Self::Pixel) {
        let mut pixel = self.0.get_pixel(x, y);
        pixel.blend(&color);
        unsafe { self.0.unsafe_put_pixel(x, y, pixel) }
    }
}

/// Adds pixels with the given weights. Results are clamped to prevent arithmetical overflows.
///
/// # Examples
/// ```
/// # extern crate image;
/// # extern crate imageproc;
/// # fn main() {
/// use image::Rgb;
/// use imageproc::pixelops::weighted_sum;
///
/// let left = Rgb([10u8, 20u8, 30u8]);
/// let right = Rgb([100u8, 80u8, 60u8]);
///
/// let sum = weighted_sum(left, right, 0.7, 0.3);
/// assert_eq!(sum, Rgb([37, 38, 39]));
/// # }
/// ```
pub fn weighted_sum<P: Pixel>(left: P, right: P, left_weight: f32, right_weight: f32) -> P
where
    P::Subpixel: ValueInto<f32> + Clamp<f32>,
{
    left.map2(&right, |p, q| {
        weighted_channel_sum(p, q, left_weight, right_weight)
    })
}

#[inline(always)]
fn weighted_channel_sum<C>(left: C, right: C, left_weight: f32, right_weight: f32) -> C
where
    C: ValueInto<f32> + Clamp<f32>,
{
    Clamp::clamp(cast(left) * left_weight + cast(right) * right_weight)
}

/// Simple paragraph layout for glyphs into `target`.
/// Taken from https://github.com/alexheretic/ab-glyph/blob/master/dev/src/layout.rs
pub fn layout_paragraph<F, SF>(font: SF, position: Point, text: &str, target: &mut Vec<Glyph>)
where
    F: Font,
    SF: ScaleFont<F>,
{
    let v_advance = font.height() + font.line_gap();
    let mut caret = position + point(0.0, font.ascent());
    let mut last_glyph: Option<Glyph> = None;
    for c in text.chars() {
        if c.is_control() {
            if c == '\n' {
                caret = point(position.x, caret.y + v_advance);
                last_glyph = None;
            }
            continue;
        }
        let mut glyph = font.scaled_glyph(c);
        if let Some(previous) = last_glyph.take() {
            caret.x += font.kern(previous.id, glyph.id);
        }
        glyph.position = caret;

        last_glyph = Some(glyph.clone());
        caret.x += font.h_advance(glyph.id);

        target.push(glyph);
    }
}

/// Draws colored text on an image in place. `scale` is augmented font scaling on both the x and y axis (in pixels). Note that this function *does not* support newlines, you must do this manually
pub fn draw_text_mut<'a, C, F>(
    canvas: &'a mut C,
    color: C::Pixel,
    x: i32,
    y: i32,
    scale: PxScale,
    max_width: u32,
    font: F,
    text: &'a str,
) where
    C: Canvas,
    <C::Pixel as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>,
    F: Font,
{
    let scaled_font = font.as_scaled(scale);
    let mut glyphs = Vec::new();
    let position = point(x as f32, y as f32);
    layout_paragraph(scaled_font, position, text, &mut glyphs);

    let last_glyph = &glyphs[glyphs.len() - 1];
    let actual_width = last_glyph.position.x - (x as f32) + last_glyph.scale.x;
    if actual_width > max_width as f32 {
        let shrink_factor = actual_width / (max_width as f32);
        let new_scale = PxScale {
            x: scale.x / shrink_factor,
            y: scale.y,
        };
        glyphs.clear();
        let rescaled_font = font.as_scaled(new_scale);
        layout_paragraph(rescaled_font, position, text, &mut glyphs);
    }

    // Loop through the glyphs in the text, positing each one on a line
    for glyph in glyphs {
        if let Some(outlined) = scaled_font.outline_glyph(glyph) {
            let bounds = outlined.px_bounds();
            // Draw the glyph into the image per-pixel by using the draw closure
            outlined.draw(|x, y, v| {
                // Offset the position by the glyph bounding box
                let pixel_x = x + bounds.min.x as u32;
                let pixel_y = y + bounds.min.y as u32;
                let px = canvas.get_pixel(pixel_x, pixel_y);
                let weighted_color = weighted_sum(px, color, 1.0 - v, v);
                // Turn the coverage into an alpha value (blended with any previous)
                canvas.draw_pixel(pixel_x, pixel_y, weighted_color);
            });
        }
    }
}

/// Draws colored text on an image in place. `scale` is augmented font scaling on both the x and y axis (in pixels). Note that this function *does not* support newlines, you must do this manually
pub fn draw_text<'a, I, F>(
    image: &'a mut I,
    color: I::Pixel,
    x: i32,
    y: i32,
    scale: PxScale,
    max_width: u32,
    font: F,
    text: &'a str,
) -> Image<I::Pixel>
where
    I: GenericImage,
    <I::Pixel as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>,
    I::Pixel: 'static,
    F: Font,
{
    let mut out = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0).unwrap();
    draw_text_mut(&mut out, color, x, y, scale, max_width, font, text);
    out
}
