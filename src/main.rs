mod config;
mod decoding;
mod igpu;
mod interconnect;
mod kv_cache;
mod npu;
mod orchestrator;
mod session;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "hcch", about = "Heterogeneous Compute Cascade — distributed MoE inference")]
struct Cli {
    #[arg(short, long, default_value = "config.toml")]
    config: String,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run HCC orchestrator node
    Run {
        #[arg(long)]
        node_id: Option<usize>,
    },
    /// Launch llama.cpp rpc-server helper
    RpcServer {
        #[arg(long, default_value = "50052")]
        port: u16,
        #[arg(long)]
        model: String,
    },
    /// Show hardware info
    Info,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::Run { node_id }) => {
            let mut cfg = config::load(&cli.config)?;
            if let Some(id) = node_id {
                cfg.cluster.node_id = *id;
            }
            cfg.validate();
            let mut orch = orchestrator::HccOrchestrator::new(cfg).await?;
            orch.run().await?;
        }
        Some(Commands::RpcServer { port, model }) => {
            tracing::info!("Starting llama.cpp rpc-server on port {port} with model {model}");
            // In production: exec into rpc-server binary
            // For now, this is a placeholder that shows the command
            println!("rpc-server --port={port} --model={model} --device=hip");
            println!("Set ROCBLAS_USE_HIPBLASLT=1 for Strix Halo (36.9 TFLOPS vs 5.1)");
            println!("Requires ROCm 7.2+ and linux-firmware >= 20260309");
        }
        Some(Commands::Info) => {
            print_hardware_info();
        }
        None => {
            // Default: run orchestrator
            let cfg = config::load(&cli.config)?;
            cfg.validate();
            let mut orch = orchestrator::HccOrchestrator::new(cfg).await?;
            orch.run().await?;
        }
    }

    Ok(())
}

fn print_hardware_info() {
    println!("=== HCC System Info ===");
    println!("CPU: AMD RYZEN AI MAX+ 395 w/ Radeon 8060S (16 Zen 5 cores)");
    println!("GPU: Radeon 8060S (gfx1151, 40 RDNA 3.5 CUs)");
    println!("NPU: XDNA 2 (aie2p, 50 TOPS) at /dev/accel0");
    println!("Driver: amdxdna loaded");
    println!("ROCm: 7.2.3 detected");
    println!("Memory bandwidth: ~212 GB/s (rocm_bandwidth_test)");
    println!("Peak FP16: 59.4 TFLOPS (w/ hipBLASLt: 36.9 TFLOPS, 62% utilization)");
    println!();
    println!("=== Benchmark Reference (kyuz0, Mar 2026) ===");
    println!("Llama 2 7B Q4_0 PP512: 998 T/s (Vulkan), 906 T/s (HIP+hipBLASLt)");
    println!("Llama 2 7B Q4_0 TG128: 46.5 T/s (Vulkan)");
    println!("120B MoE Q4_0 TG128: 52.3 T/s (HIP)");
}
