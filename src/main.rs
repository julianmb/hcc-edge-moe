mod config;
mod decoding;
mod igpu;
mod interconnect;
mod kv_cache;
mod npu;
mod orchestrator;
mod session;

use clap::Parser;
use tracing_subscriber::{prelude::*, EnvFilter};

#[derive(Parser)]
#[command(name = "hcch", about = "Heterogeneous Compute Cascade — distributed MoE inference")]
struct Cli {
    #[arg(short, long, default_value = "config.toml")]
    config: String,

    #[arg(short, long)]
    role: Option<String>,

    #[arg(long)]
    node_id: Option<usize>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(EnvFilter::from_default_env())
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();
    let cfg: config::HccConfig = config::load(&cli.config)?;

    if let Some(role) = cli.role {
        tracing::info!("starting HCC node in role: {role}");
    }

    let mut orchestrator = orchestrator::HccOrchestrator::new(cfg).await?;
    orchestrator.run().await?;

    Ok(())
}
