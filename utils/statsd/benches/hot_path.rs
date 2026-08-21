#![expect(
    dead_code,
    clippy::expect_used,
    reason = "the benchmark includes the full private store to measure its exact recurring method"
)]

use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use gst::prelude::*;

#[path = "../src/metrics.rs"]
mod metrics;

fn new_metrics(max_pad_series: usize) -> metrics::MetricsSlot {
    let slot = metrics::MetricsSlot::default();
    let (collector, _retired) = metrics::Metrics::new(None, None, max_pad_series);
    assert!(slot.install(collector));
    slot
}

fn benchmark_hot_path(criterion: &mut Criterion) {
    gst::init().expect("initializing GStreamer for benchmarks");
    let element = gst::ElementFactory::make("identity")
        .build()
        .expect("constructing benchmark element");
    let pad = element.static_pad("src").expect("benchmark source pad");

    let tracked = new_metrics(1);
    tracked.record_push(&pad, 1, 1_024);
    criterion.bench_function("tracer slot cached tracked pad update", |bencher| {
        bencher.iter(|| tracked.record_push(&pad, 1, 1_024));
    });

    let ignored = new_metrics(1);
    let first = gst::ElementFactory::make("identity")
        .build()
        .expect("constructing capped element");
    let first_pad = first.static_pad("src").expect("first source pad");
    ignored.record_push(&first_pad, 1, 1);
    ignored.record_push(&pad, 1, 1_024);
    criterion.bench_function("tracer slot cached ignored pad update", |bencher| {
        bencher.iter(|| ignored.record_push(&pad, 1, 1_024));
    });
}

fn benchmark_contention(criterion: &mut Criterion) {
    gst::init().expect("initializing GStreamer for benchmarks");
    let mut group = criterion.benchmark_group("cached tracked pad contention");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(3));

    for thread_count in [1_usize, 2, 4, 8] {
        let collector = new_metrics(thread_count);
        let elements = (0..thread_count)
            .map(|_| {
                gst::ElementFactory::make("identity")
                    .build()
                    .expect("constructing benchmark element")
            })
            .collect::<Vec<_>>();
        let pads = elements
            .iter()
            .map(|element| element.static_pad("src").expect("benchmark source pad"))
            .collect::<Vec<_>>();
        for pad in &pads {
            collector.record_push(pad, 1, 1_024);
        }
        group.throughput(Throughput::Elements(
            u64::try_from(thread_count).expect("thread count fits in u64"),
        ));
        group.bench_with_input(
            BenchmarkId::new("different pads", thread_count),
            &thread_count,
            |bencher, _| {
                bencher.iter_custom(|iterations| {
                    let started = Instant::now();
                    std::thread::scope(|scope| {
                        for pad in &pads {
                            let collector = &collector;
                            scope.spawn(move || {
                                for _ in 0..iterations {
                                    collector.record_push(pad, 1, 1_024);
                                }
                            });
                        }
                    });
                    started.elapsed()
                });
            },
        );
        let shared_pad = pads.first().expect("at least one benchmark pad");
        group.bench_with_input(
            BenchmarkId::new("same pad", thread_count),
            &thread_count,
            |bencher, &thread_count| {
                bencher.iter_custom(|iterations| {
                    let started = Instant::now();
                    std::thread::scope(|scope| {
                        for _ in 0..thread_count {
                            let collector = &collector;
                            scope.spawn(move || {
                                for _ in 0..iterations {
                                    collector.record_push(shared_pad, 1, 1_024);
                                }
                            });
                        }
                    });
                    started.elapsed()
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, benchmark_hot_path, benchmark_contention);
criterion_main!(benches);
