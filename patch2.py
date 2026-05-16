with open("Heterogeneous_Compute_Cascade.tex", "r") as f:
    content = f.read()

start_idx = content.find("\\section{Post-Publication Architectural Upgrades")
end_idx = content.find("\\section{Conclusion}")

new_section = r"""\section{Post-Publication Architectural Upgrades (v0.7.0)}
\label{sec:v070}

Since the initial preprint, significant architectural enhancements have been integrated into the HCC codebase to push the boundaries of the Strix Halo platform:

\begin{itemize}
    \item \textbf{Agentic CPU Orchestration (v0.7.0):} Acknowledging the shifting CPU-GPU balance in Agentic AI, we implemented a dedicated CPU orchestration layer. The 16 Zen 5 cores now continuously parse the draft token stream, performing real-time JSON schema validation and speculatively pre-warming tool execution environments (e.g., DNS resolution) before the iGPU completes its verification forward pass.
    \item \textbf{Speculative Tree Attention:} We replaced linear draft sequences with branching draft trees. By utilizing a 2D Tree Attention Mask, the iGPU evaluates multiple candidate branches simultaneously, exponentially increasing $\mathbb{E}[k]$ per USB4 network crossing.
    \item \textbf{Native XRT FFI:} Draft generation now bypasses the local HTTP stack entirely. By dynamically linking to \texttt{libxrt\_core.so} and allocating zero-copy Buffer Objects (\texttt{xrt::bo}) directly in the LPDDR5x UMA, NPU dispatch latency is reduced to the sub-millisecond regime.
    \item \textbf{AF\_XDP Kernel-Bypass:} To overcome the remaining 17 $\mu$s TCP/IP overhead over \texttt{thunderbolt-net}, we implemented an \texttt{AF\_XDP} zero-copy socket interface. By mapping DMA-BUFs directly into the UMEM ring buffers, HCC achieves true kernel-bypass, driving USB4 RTT into the single-digit microsecond range.
    \item \textbf{Custom $d_{kv}=576$ HIP Kernel:} Generic quantization frameworks require power-of-2 dimensions, which forces GLM-5.1's $d_{kv}=576$ to be padded to 1024---wasting 44\% of the KV cache memory footprint. We wrote and integrated a custom ROCm HIP kernel that explicitly factors 576 into $512 + 64$ for the Fast Walsh-Hadamard Transform, eliminating all padding waste.
\end{itemize}

"""

if start_idx != -1 and end_idx != -1:
    content = content[:start_idx] + new_section + content[end_idx:]
    with open("Heterogeneous_Compute_Cascade.tex", "w") as f:
        f.write(content)
