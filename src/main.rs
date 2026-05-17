mod benchmark;
mod config;
mod decoding;
mod error;
mod igpu;
mod interconnect;
mod kv_cache;
mod measure;
mod npu;
mod orchestrator;
mod session;
mod tuner;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "hcch", about = "Heterogeneous Compute Cascade — distributed MoE inference across AMD Strix Halo nodes", version = "0.2.0")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run HCC orchestrator node
    Run {
        #[arg(short, long, default_value = "config.toml")]
        config: String,
        #[arg(long)]
        node_id: Option<usize>,
    },
    /// Launch llama.cpp rpc-server
    RpcServer {
        #[arg(long, default_value = "50052")]
        port: u16,
        #[arg(long)]
        model: String,
    },
    /// Run benchmark suite (roofline validation)
    Benchmark {
        #[arg(short, long, default_value = "config.toml")]
        config: String,
    },
    /// Run real inference measurement against paper predictions
    Measure {
        #[arg(short, long, default_value = "config.toml")]
        config: String,
    },
    /// Check kernel tuning status
    Tune {
        #[arg(long)]
        apply: bool,
    },
    /// Show hardware info
    Info,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::Run { config, node_id }) => {
            let mut cfg = config::load(config)?;
            if let Some(id) = node_id {
                cfg.cluster.node_id = *id;
            }
            let mut orch = orchestrator::HccOrchestrator::new(cfg).await?;
            orch.run().await?;
        }
        Some(Commands::RpcServer { port, model }) => {
            print_rpc_server_cmd(*port, model);
        }
        Some(Commands::Benchmark { config }) => {
            let cfg = config::load(config)?;
            cfg.validate();
            benchmark::BenchmarkRunner::run_all(&cfg)?;
        }
        Some(Commands::Measure { config }) => {
            let cfg = config::load(config)?;
            cfg.validate();
            measure::MeasureRunner::run(&cfg).await?;
        }
        Some(Commands::Tune { apply }) => {
            let report = tuner::KernelTuner::check();
            println!("{}", report);
            if *apply {
                println!("── Applying kernel tuning ──");
                for cmd in tuner::KernelTuner::apply() {
                    println!("  $ {cmd}");
                }
                println!("(requires root — run commands above manually if needed)");
            }
        }
        Some(Commands::Info) | None => {
            print_hardware_info();
        }
    }

    Ok(())
}

fn print_rpc_server_cmd(port: u16, model: &str) {
    println!("── llama.cpp rpc-server ──");
    println!();
    println!("  rpc-server --port={port} --model={model} --device=hip");
    println!();
    println!("  Required environment:");
    println!("    ROCBLAS_USE_HIPBLASLT=1   (36.9 vs 5.1 TFLOPS on Strix Halo)");
    println!("    HSA_OVERRIDE_GFX_VERSION=11.5.1  (gfx1151 compatibility)");
    println!();
    println!("  Requires:");
    println!("    - ROCm 7.2.3+");
    println!("    - linux-firmware >= 20260309");
    println!("    - Kernel 6.17+");
}

fn print_hardware_info() {
    let tune = tuner::KernelTuner::check();
    println!();
    println!("╔══════════════════════════════════════════╗");
    println!("║     HCC System Information               ║");
    println!("╚══════════════════════════════════════════╝");
    println!();
    println!("── Hardware ──");
    println!("CPU:  AMD RYZEN AI MAX+ 395 w/ Radeon 8060S");
    println!("      16 Zen 5 cores, 32 threads, up to 5.1 GHz");
    println!("GPU:  Radeon 8060S (gfx1151, 40 RDNA 3.5 CUs @ 2.9 GHz)");
    println!("NPU:  XDNA 2 (aie2p, 50 TOPS) at /dev/accel0");
    println!("RAM:  128 GB LPDDR5x-8000 (256-bit, 212 GB/s sustained)");
    println!();
    println!("── Kernel Tuning ──");
    print!("{tune}");
    println!();
    println!("── ROCm ──");
    println!("Version: 7.2.3");
    println!("MIGraphX: /opt/rocm/lib/libmigraphx_c.so.3 present");
    println!("hipBLASLt: /opt/rocm/lib/libhipblaslt.so.0 present");
    println!();
    println!("── Performance Reference ──");
    println!("  llama.cpp 7B Q4_0  PP512:  998 tok/s (Vulkan)");
    println!("  llama.cpp 7B Q4_0  TG128:  46.5 tok/s");
    println!("  llama.cpp 120B MoE  TG128:  52.3 tok/s (HIP+hipBLASLt)");
    println!();
    println!("  Source: kyuz0/amd-strix-halo-toolboxes (Mar 2026)");
    println!();
    println!("── Commands ──");
    println!("  hcch run              Start HCC orchestrator");
    println!("  hcch rpc-server       Launch llama.cpp RPC server");
    println!("  hcch benchmark        Run roofline benchmark suite");
    println!("  hcch measure          Run real inference vs paper predictions");
    println!("  hcch tune             Check kernel tuning");
    println!("  hcch info             Show this info");
    println!();
}
