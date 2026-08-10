use super::nms;

pub(super) const NUM_CLASSES: usize = 80;
pub(super) const REG_MAX: usize = 7;
pub(super) const BINS: usize = REG_MAX + 1;
pub(super) const CHANNELS: usize = NUM_CLASSES + 4 * BINS;

const NANODET_STRIDES: &[usize] = &[8, 16, 32];
const NANODET_PLUS_STRIDES: &[usize] = &[8, 16, 32, 64];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Contract {
    pub(super) name: &'static str,
    pub(super) input_size: usize,
    pub(super) points: usize,
    pub(super) strides: &'static [usize],
}

impl Contract {
    #[must_use]
    pub(super) const fn dims(self) -> [usize; 3] {
        [1, self.points, CHANNELS]
    }

    #[must_use]
    pub(super) const fn elements(self) -> usize {
        self.points * CHANNELS
    }
}

pub(super) const CONTRACTS: [Contract; 4] = [
    Contract {
        name: "nanodet-m-320",
        input_size: 320,
        points: 2_100,
        strides: NANODET_STRIDES,
    },
    Contract {
        name: "nanodet-plus-320",
        input_size: 320,
        points: 2_125,
        strides: NANODET_PLUS_STRIDES,
    },
    Contract {
        name: "nanodet-m-416",
        input_size: 416,
        points: 3_549,
        strides: NANODET_STRIDES,
    },
    Contract {
        name: "nanodet-plus-416",
        input_size: 416,
        points: 3_598,
        strides: NANODET_PLUS_STRIDES,
    },
];

#[must_use]
pub(super) fn contract_for_dims(dims: &[usize]) -> Option<Contract> {
    CONTRACTS
        .iter()
        .copied()
        .find(|contract| dims == contract.dims())
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Detection {
    pub(super) class: u8,
    pub(super) score: f32,
    pub(super) x1: f32,
    pub(super) y1: f32,
    pub(super) x2: f32,
    pub(super) y2: f32,
    pub(super) ordinal: u16,
}

fn distribution_expectation<T: Copy>(
    values: &[T],
    to_f32: impl Fn(T) -> f32 + Copy,
) -> Option<f32> {
    let mut max = f32::NEG_INFINITY;
    for value in values {
        let value = to_f32(*value);
        if !value.is_finite() {
            return None;
        }
        max = max.max(value);
    }

    let mut denominator = 0.0_f32;
    let mut numerator = 0.0_f32;
    for (index, value) in values.iter().enumerate() {
        let probability = (to_f32(*value) - max).exp();
        #[expect(
            clippy::cast_precision_loss,
            reason = "distribution bin indices are at most seven and exactly representable"
        )]
        let index = index as f32;
        denominator += probability;
        numerator += index * probability;
    }
    (denominator.is_finite() && denominator > 0.0).then_some(numerator / denominator)
}

fn best_class<T: Copy>(scores: &[T], to_f32: impl Fn(T) -> f32 + Copy) -> Option<(usize, f32)> {
    let mut best = None;
    for (class, value) in scores.iter().copied().enumerate() {
        let score = to_f32(value);
        if !score.is_finite() {
            return None;
        }
        if best.is_none_or(|(_, best_score)| !score.total_cmp(&best_score).is_lt()) {
            best = Some((class, score));
        }
    }
    best
}

pub(super) fn decode(
    output: &[f32],
    contract: Contract,
    score_threshold: f32,
    iou_threshold: f32,
    max_detections: usize,
    candidates: &mut Vec<Detection>,
) -> Result<(), String> {
    decode_values(
        output,
        contract,
        score_threshold,
        iou_threshold,
        max_detections,
        std::convert::identity,
        candidates,
    )
}

pub(super) fn decode_float16(
    output: &[u16],
    contract: Contract,
    score_threshold: f32,
    iou_threshold: f32,
    max_detections: usize,
    candidates: &mut Vec<Detection>,
) -> Result<(), String> {
    decode_values(
        output,
        contract,
        score_threshold,
        iou_threshold,
        max_detections,
        |bits| half::f16::from_bits(bits).to_f32(),
        candidates,
    )
}

fn decode_values<T: Copy>(
    output: &[T],
    contract: Contract,
    score_threshold: f32,
    iou_threshold: f32,
    max_detections: usize,
    to_f32: impl Fn(T) -> f32 + Copy,
    candidates: &mut Vec<Detection>,
) -> Result<(), String> {
    if output.len() != contract.elements() {
        return Err(format!(
            "NanoDet output has {} elements; {} expects {} ({}x{CHANNELS})",
            output.len(),
            contract.name,
            contract.elements(),
            contract.points
        ));
    }

    candidates.clear();
    let initial_capacity = max_detections.min(contract.points);
    if candidates.capacity() < initial_capacity {
        candidates.reserve(initial_capacity);
    }
    let mut rows = output.chunks_exact(CHANNELS);
    let mut point = 0usize;
    #[expect(
        clippy::cast_precision_loss,
        reason = "supported input sizes are exactly representable"
    )]
    let input_size = contract.input_size as f32;
    for &stride in contract.strides {
        let grid = contract.input_size.div_ceil(stride);
        #[expect(
            clippy::cast_precision_loss,
            reason = "supported strides are exactly representable"
        )]
        let stride_f32 = stride as f32;
        for grid_y in 0..grid {
            for grid_x in 0..grid {
                let Some(row) = rows.next() else {
                    return Err(format!("NanoDet output row {point} is unavailable"));
                };
                let ordinal = point;
                point += 1;

                let Some(scores) = row.get(..NUM_CLASSES) else {
                    continue;
                };
                let Some((class, score)) = best_class(scores, to_f32) else {
                    continue;
                };
                if score < score_threshold {
                    continue;
                }

                let mut distances = [0.0_f32; 4];
                let mut valid = true;
                for (side, distance) in distances.iter_mut().enumerate() {
                    let start = NUM_CLASSES + side * BINS;
                    let Some(bins) = row.get(start..start + BINS) else {
                        valid = false;
                        break;
                    };
                    let Some(value) = distribution_expectation(bins, to_f32) else {
                        valid = false;
                        break;
                    };
                    *distance = value * stride_f32;
                }
                if !valid {
                    continue;
                }

                #[expect(
                    clippy::cast_precision_loss,
                    reason = "supported grid coordinates and input sizes are exactly representable"
                )]
                let (center_x, center_y) = (grid_x as f32 * stride_f32, grid_y as f32 * stride_f32);
                let (x1, y1, x2, y2) = (
                    (center_x - distances[0]).clamp(0.0, input_size),
                    (center_y - distances[1]).clamp(0.0, input_size),
                    (center_x + distances[2]).clamp(0.0, input_size),
                    (center_y + distances[3]).clamp(0.0, input_size),
                );
                if x2 <= x1 || y2 <= y1 {
                    continue;
                }
                let class = u8::try_from(class)
                    .map_err(|_error| format!("NanoDet class {class} exceeds u8"))?;
                let ordinal = u16::try_from(ordinal)
                    .map_err(|_error| format!("NanoDet point {ordinal} exceeds u16"))?;
                candidates.push(Detection {
                    class,
                    score,
                    x1,
                    y1,
                    x2,
                    y2,
                    ordinal,
                });
            }
        }
    }

    if point != contract.points {
        return Err(format!(
            "{} grid produces {point} points; expected {}",
            contract.name, contract.points
        ));
    }

    nms::suppress(candidates, iou_threshold, max_detections);
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::hint::black_box;
    use std::time::Instant;

    use super::*;

    #[test]
    fn distribution_expectation_is_stable_and_rejects_non_finite_logits() {
        let uniform = distribution_expectation(&[0.0; BINS], std::convert::identity)
            .expect("finite distribution");
        assert!((uniform - 3.5).abs() < f32::EPSILON);

        let dominant = distribution_expectation(
            &[
                -10_000.0, -10_000.0, -10_000.0, 10_000.0, -10_000.0, -10_000.0, -10_000.0,
                -10_000.0,
            ],
            std::convert::identity,
        )
        .expect("stable extreme distribution");
        assert!((dominant - 3.0).abs() < f32::EPSILON);
        assert_eq!(distribution_expectation(&[], std::convert::identity), None);
        assert_eq!(
            distribution_expectation(&[f32::NAN; BINS], std::convert::identity),
            None
        );
        assert_eq!(
            distribution_expectation(&[f32::INFINITY; BINS], std::convert::identity),
            None
        );
        assert_eq!(
            distribution_expectation(&[f32::NEG_INFINITY; BINS], std::convert::identity),
            None
        );
    }

    fn output(contract: Contract) -> Vec<f32> {
        vec![0.0; contract.elements()]
    }

    fn set_candidate(
        output: &mut [f32],
        point: usize,
        class: usize,
        score: f32,
        distance_bin: usize,
    ) {
        let start = point * CHANNELS;
        if let Some(value) = output.get_mut(start + class) {
            *value = score;
        }
        for side in 0..4 {
            for bin in 0..BINS {
                if let Some(value) = output.get_mut(start + NUM_CLASSES + side * BINS + bin) {
                    *value = if bin == distance_bin { 20.0 } else { 0.0 };
                }
            }
        }
    }

    fn decoded(
        output: &[f32],
        contract: Contract,
        score_threshold: f32,
        iou_threshold: f32,
        max_detections: usize,
    ) -> Vec<Detection> {
        let mut detections = Vec::new();
        decode(
            output,
            contract,
            score_threshold,
            iou_threshold,
            max_detections,
            &mut detections,
        )
        .expect("valid tensor");
        detections
    }

    fn env_count(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|value| *value > 0)
            .unwrap_or(default)
    }

    fn report_benchmark<T: Copy>(
        contract: Contract,
        tensor_type: &str,
        density: &str,
        values: &[T],
        warmup: usize,
        iterations: usize,
        decode: impl Fn(&[T], &mut Vec<Detection>) -> Result<(), String>,
    ) {
        let mut detections = Vec::new();
        for _ in 0..warmup {
            decode(black_box(values), &mut detections).expect("benchmark tensor decodes");
        }

        let start = Instant::now();
        for _ in 0..iterations {
            decode(black_box(values), &mut detections).expect("benchmark tensor decodes");
            black_box(&detections);
        }
        let elapsed = start.elapsed();
        let sample_count = u32::try_from(iterations).expect("iteration count fits u32");
        let average = elapsed / sample_count;
        println!(
            "model={} type={tensor_type} density={density} points={} detections={} warmup={warmup} iterations={iterations} average={average:?} average_ns={} throughput_fps={:.1}",
            contract.name,
            contract.points,
            detections.len(),
            average.as_nanos(),
            1.0 / average.as_secs_f64()
        );
    }

    #[test]
    #[ignore = "run explicitly in release mode with GSTSMITH_BENCH_WARMUP and GSTSMITH_BENCH_ITERATIONS"]
    fn benchmark_decoder_reports_model_type_and_candidate_density() {
        let warmup = env_count("GSTSMITH_BENCH_WARMUP", 10);
        let iterations = env_count("GSTSMITH_BENCH_ITERATIONS", 100);

        for contract in CONTRACTS {
            for (density, candidate_count) in [("sparse", 8), ("dense", contract.points)] {
                let mut float32 = output(contract);
                for candidate in 0..candidate_count {
                    let point = candidate * contract.points / candidate_count;
                    set_candidate(&mut float32, point, candidate % NUM_CLASSES, 0.9, 1);
                }
                let float16 = float32
                    .iter()
                    .map(|value| half::f16::from_f32(*value).to_bits())
                    .collect::<Vec<_>>();

                report_benchmark(
                    contract,
                    "float32",
                    density,
                    &float32,
                    warmup,
                    iterations,
                    |values, detections| decode(values, contract, 0.3, 0.6, 100, detections),
                );
                report_benchmark(
                    contract,
                    "float16",
                    density,
                    &float16,
                    warmup,
                    iterations,
                    |values, detections| {
                        decode_float16(values, contract, 0.3, 0.6, 100, detections)
                    },
                );
            }
        }
    }

    #[test]
    fn all_contracts_match_their_stride_geometry() {
        for contract in CONTRACTS {
            let points = contract
                .strides
                .iter()
                .map(|stride| contract.input_size.div_ceil(*stride).pow(2))
                .sum::<usize>();
            assert_eq!(points, contract.points, "{}", contract.name);
            assert_eq!(contract_for_dims(&contract.dims()), Some(contract));
        }
    }

    #[test]
    fn decodes_a_known_point_and_clamps_boundaries_for_each_contract() {
        for contract in CONTRACTS {
            let mut values = output(contract);
            let first_grid = contract.input_size.div_ceil(contract.strides[0]);
            set_candidate(&mut values, first_grid + 1, 7, 0.9, 1);
            let detections = decoded(&values, contract, 0.3, 0.6, 100);
            assert_eq!(detections.len(), 1, "{}", contract.name);
            let detection = detections.first().expect("one detection");
            assert_eq!(detection.class, 7);
            assert!((detection.score - 0.9).abs() < 1e-6);
            assert!(detection.x1 >= 0.0 && detection.y1 >= 0.0);
            #[expect(
                clippy::cast_precision_loss,
                reason = "supported input sizes are exactly representable"
            )]
            let input_size = contract.input_size as f32;
            assert!(detection.x2 <= input_size);
            assert!(detection.y2 <= input_size);
            assert!((detection.x1 - 0.0).abs() < 0.01);
            assert!((detection.y1 - 0.0).abs() < 0.01);
            assert!((detection.x2 - 16.0).abs() < 0.01);
            assert!((detection.y2 - 16.0).abs() < 0.01);
        }
    }

    #[test]
    fn class_aware_nms_suppresses_only_the_same_class() {
        let contract = CONTRACTS[1];
        let mut values = output(contract);
        set_candidate(&mut values, 41, 2, 0.9, 2);
        set_candidate(&mut values, 42, 2, 0.8, 2);
        set_candidate(&mut values, 43, 3, 0.7, 2);
        let detections = decoded(&values, contract, 0.3, 0.3, 100);
        assert_eq!(detections.len(), 2);
        assert_eq!(detections[0].class, 2);
        assert_eq!(detections[1].class, 3);
    }

    #[test]
    fn ordering_is_deterministic_and_maximum_is_enforced() {
        let contract = CONTRACTS[1];
        let mut values = output(contract);
        set_candidate(&mut values, 82, 4, 0.8, 1);
        set_candidate(&mut values, 164, 3, 0.9, 1);
        set_candidate(&mut values, 246, 2, 0.7, 1);
        let detections = decoded(&values, contract, 0.3, 1.0, 2);
        assert_eq!(detections.len(), 2);
        assert_eq!(detections[0].class, 3);
        assert_eq!(detections[1].class, 4);
    }

    #[test]
    fn skips_non_finite_rows_and_below_threshold_scores() {
        let contract = CONTRACTS[1];
        let mut values = output(contract);
        set_candidate(&mut values, 82, 4, 0.2, 1);
        set_candidate(&mut values, 164, 3, 0.9, 1);
        if let Some(value) = values.get_mut(164 * CHANNELS + NUM_CLASSES) {
            *value = f32::NAN;
        }
        assert!(decoded(&values, contract, 0.3, 0.6, 100).is_empty());
    }

    #[test]
    fn rejects_the_wrong_element_count() {
        let contract = CONTRACTS[1];
        let error =
            decode(&[0.0], contract, 0.3, 0.6, 100, &mut Vec::new()).expect_err("wrong length");
        assert!(error.contains("expects 238000"));
    }
}
