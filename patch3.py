with open("Heterogeneous_Compute_Cascade.tex", "r") as f:
    content = f.read()

start_idx = content.find("\\section{Post-Publication Architectural Upgrades (v0.7.0)}")
end_idx = content.find("\\section{Conclusion}")

new_section = r"""\section{Post-Publication Architectural Upgrades (v0.7.1)}
\label{sec:v071}

Since the initial preprint, significant architectural enhancements have been integrated into the HCC codebase to push the boundaries of the Strix Halo platform:

\begin{itemize}
    \item \textbf{NPU-Offloaded MoE Routing:} To eliminate iGPU pipeline bubbles caused by evaluating small, dense gating networks, HCC now offloads the MoE router to the XDNA 2 NPU. The NPU asynchronously computes Top-K gating probabilities and writes the expert indices directly to a shared UMA buffer, allowing the iGPU to execute expert layers without interruption.
    \item \textbf{Continuous Speculative KV-Correction (CSKVC):} To combat draft-model drift over long context windows, the target iGPU now computes the residual difference between its hidden states and the draft's hidden states. This residual is 1-bit quantized (re-using our QJL math) and appended to the \texttt{AF\_XDP} verification packet. The NPU continuously applies this micro-correction, artificially aligning the draft state with the target to sustain high acceptance rates ($\alpha$).
    \item \textbf{Agentic DirectStorage (\texttt{io\_uring}):} The CPU Agentic Orchestrator now uses Linux \texttt{io\_uring} to issue zero-copy DMA reads straight from the system's Gen5 NVMe SSD into the iGPU's LPDDR5x pool. This completely bypasses CPU bounce buffers when pre-fetching massive contexts for speculatively executed tools (like local Vector DB queries).
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
