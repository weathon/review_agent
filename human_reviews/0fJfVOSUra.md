# ThunderKittens: Simple, Fast, and $\textit{Adorable}$ Kernels

- Avg Score: 7.50
- Decision: Accept (Spotlight)
- Scores: 6, 8, 8, 8

## Abstract
The challenge of mapping AI architectures to GPU hardware is creating a critical bottleneck in AI progress. Despite substantial efforts, hand-written custom kernels fail to meet their theoretical performance thresholds, even on well-established operations like linear attention. The diverse capabilities of GPUs suggests we might we need a wide variety of techniques to achieve high performance. However, our work explores if a small number of key abstractions can drastically simplify the process. We present ThunderKittens (TK), a framework for writing performant AI kernels while remaining easy to use. Our abstractions map to the three levels of the GPU hierarchy: (1) at the warp-level, we provide 16x16 matrix tiles as basic data structures and PyTorch-like operations, (2) at the thread-block level, we provide templates for asynchronously overlapping operations, and (3) at the grid-level, TK helps hide block launch, tear-down, and memory costs. We show the value of TK by providing simple & diverse kernels that match or outperform prior art. We match CuBLAS and FlashAttention-3 on GEMM and attention inference performance and outperform the strongest baselines by $10-40$\% on attention backwards, $8\times$ on state space models, and $14\times$ on linear attention.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces THUNDERKITTENS (TK), a framework that simplifies writing AI kernels for GPUs while still allowing for high performance. Using a few key abstractions, TK provides tools for developers to create efficient kernels without deep expertise in GPU programming. Through benchmarking, the authors show that TK performs on par with or better than other leading frameworks like CuBLAS and FlashAttention-3 for various AI tasks. TK’s accessible design, inspired by PyTorch and NumPy, aims to make high-performance kernel development more straightforward and accessible to a wider audience.

### Strengths
The paper offers a fresh and practical approach to GPU kernel programming, using only a handful of essential abstractions to make high-performance kernel writing accessible to a wider range of developers. This simplicity-oriented approach can reduce the complexity typically associated with GPU development, which could be particularly valuable for those without extensive CUDA experience. In terms of performance, THUNDERKITTENS shows impressive results, even surpassing established libraries like CuBLAS and FlashAttention-3 in several tasks, especially in backward pass operations for attention mechanisms and linear attention. The results strongly suggest that TK’s design strikes a good balance between simplicity and performance optimization. Furthermore, by aligning its design with PyTorch and NumPy, TK makes it easier for non-specialists to adopt, potentially expanding the accessibility of efficient GPU programming.

### Weaknesses
1- While the minimalistic design is a key strength, it may also limit TK’s flexibility for more specialized AI tasks that require tailored optimization strategies. As demands grow for handling complex and emerging AI workloads, the current set of abstractions could potentially fall short.

2- The focus on NVIDIA’s H100 GPUs raises questions about how well TK can transfer to other platforms, such as AMD or Apple GPUs. Expanding on cross-platform compatibility would provide more clarity about TK’s broader usability.

3- Though the paper demonstrates strong performance on medium-sized data, it is less clear how TK handles scalability with very large datasets or highly parallelized scenarios. Addressing its limitations in these settings could further support TK’s value in real-world applications.

### Questions
Could the authors elaborate on the potential for cross-platform compatibility? Given the focus on NVIDIA’s H100 GPUs, it would be helpful to understand whether TK’s abstractions could be adapted to other GPU architectures, like AMD or Apple, and what challenges might arise.

The paper demonstrates TK’s strong performance on medium-sized data blocks, but could the authors provide more insights into how well TK scales with very large datasets? Are there specific limitations to consider for applications requiring high parallelization or extensive data handling?

Could the authors expand on their design choice to limit TK to a few key abstractions? Are there specific reasons why additional templates or adaptive features were not incorporated, and would doing so have risked undermining the framework’s simplicity?

In scenarios with high memory demands, how does TK manage the balance between memory overhead and computational efficiency? Further detail on this balance could clarify TK’s suitability for applications with varied memory and compute requirements.

Lastly, could the authors clarify TK’s debugging process, especially for users who may not be familiar with GPU optimization? Since GPU kernel errors can be challenging to diagnose, any insights into how TK might support error handling and debugging would be valuable for potential adopters.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper presents ThunderKittens (TK), a C++ embedded library for writing high-performance CUDA kernels for NVIDIA GPUs. It introduces warp-, thread-block-, and grid-level abstractions to facilitate mapping of kernels to the GPU hierarchy. Experimental results indicate that TK can outperform strong industrial baselines, achieving superior performance for GEMM and attention kernels.

### Strengths
1. The TK library provides a useful abstraction for writing high-performance asynchronous kernels on GPU.
2. The presentation is clear and accessible, especially the introductory sections on GPU architecture, which provide a helpful overview for ML researchers who may lack in-depth experience with GPU programming.
3. The experimental results are compelling, showing performance on par or better than highly optimized kernels, such as FlashAttention3. The paper also demonstrates significant speedups across different kernel types compared to state-of-the-art frameworks like Triton and PyTorch.

### Weaknesses
1. The TK library is still too low-level with too many details, which requires users to manage synchronization carefully and does not simplify the programming burden.
2. The novelty and advantages of TK over CUTLASS are unclear. Many functionalities seem achievable with CUTLASS as well. The authors mention that TK addresses bank conflicts, but the evidence presented is minimal. There appear to be no inherent limitations in CUTLASS that would prevent it from avoiding bank conflicts.
3. Similarly, the benefits of TK over Triton are not well established. Triton, embedded in Python with a PyTorch-like API, may offer a more accessible interface. By contrast, TK, embedded in C++, still requires explicit handling of communication with mbarrier operations like expect and arrive. No user study or lines of code comparisons are provided to demonstrate that TK improves programmer productivity.
4. Experimental results are good, but still missing comparisons in some important cases like quantized kernels and causal attention.
5. The work reads more like a system paper, with limited ML-focused insights, raising questions about its fit for ICLR.

Minor:
- P4: "Since the frameworks are not C++ embedded, it can be challenging to use specialized hardware instructions" This statement is inaccurate; TVM provides mechanisms to incorporate low-level TensorCore instructions, and Triton also has [inline](https://triton-lang.org/main/python-api/triton.language.html#inline-assembly) operation to include PTX code.
- Section 2 does not discuss the Tensor Memory Accelerator (TMA) on Hopper, which is essential for asynchronous optimizations mentioned in Contribution 2.
- Appendix B labels appear broken (??).

### Questions
1. What are the fundamental challenges preventing CUTLASS from avoiding bank conflicts? Could it be that the FlashAttention3 kernel simply did not select the optimal layout?
2. CUTLASS has implemented both ping-pong and cooperative kernel variants for GEMM, with varying performance across different scenarios. How does TK support ping-pong and cooperative kernels, and could you include a comparison with CUTLASS in Figure 7’s GEMM kernel results?
3. TK appears designed specifically for the Hopper architecture with asynchronous features. Is it also compatible with Ampere or other GPU generations? How does TK’s performance on an A100 compare to Triton?
4. Following Q3, if Blackwell GPUs were released, would TK’s abstractions remain applicable? How do you plan to ensure extensibility across GPU generations?
5. What's the usage of the cost model in Section 2.2? This formula is highly simplified and does not guide any optimization or automatic search later.
6. Section 3.1 discusses various layouts — do users need to manually manage data organization and specify layouts in TK?
7. Figure 5 is just some wrappers of mbarriers. Any insights here?
8. Can TK effectively handle quantized kernels, where data layout is crucial for efficient transfers from TMA and WGMMA computation? How does it perform on FP8 GEMM and FlashAttention kernels?
9. What is TK's performance on causal attention kernels?
10. Please provide detailed experimental configurations in the Appendix. For example, which versions of PyTorch and Triton were used? Was `torch.compile` employed to optimize those network layers? For cuBLAS, was the latest [cuBLASLt](https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/) autotuning enabled? Since PyTorch also uses Triton as a backend, what distinguishes the two baselines in Figure 8?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a framework to facilitate easy writing of efficient CUDA kernels. The authors leverage the asynchronous compute capabilities of the Hopper series GPUs by following a producer-consumer paradigm, to efficiently overlap different kernel operations. Additionally, the authors investigate the impact of various memory ordering strategies, demonstrating that relatively simple strided patterns offer the best tradeoffs. Lastly, the authors demonstrate performance that is comparable to or exceeds existing methods, including Triton.

Overall, the work provides a significant contribution to improving computational efficiency for common operations, though the application appears limited in scope. Additionally, minor technical and structural errors impact readability. These issues could be addressed in a revision, at which point I would be inclined to raise my score.

### Strengths
The authors demonstrate significant improvements to computational efficiency within a clearly defined framework that appears relatively straightforward to adapt. Their framework also provides functionality for more complex resource management, which is often challenging to manage directly in CUDA.  Additionally, the authors demonstrate the impact of varying hyperparameters for several key kernel operations, most of which match or exceed standard baselines. Lastly, the results show a surprising contrast with Triton implementations, positioning their approach within the CUDA domain while achieving a similar level of complexity to Triton.

### Weaknesses
-	The application appears limited in scope, which should be explicitly addressed. For example, is the framework limited to Hopper GPUs and above? And the focus on 16x16 register blocks may limit extensibility to other common cases such as GEMV and sparse computations.
-	The paper contains many issues with presentation, including caption errors, grammatical and awkward wording, and typos, all of which impair readability. 
-	The paper overlooks relevant computer architecture literature regarding performance modeling, specifically in the context of balancing compute and memory (e.g. roofline analysis). Many of the findings presented in the paper are expected from the existing literature.

### Questions
1)	Is your framework limited to the Hopper series? Can it be applied to A100s, or other GPUs such as the A40/L40?
2)	You focus on the 16x16 register block level, but how can your framework be extended to smaller blocks, such as with GEMV, sparse operations, and masked operations (e.g. non-power-of-two dimensions and strided masking, such as in Natten).
3)	Throughout the paper, you focus on BF16 precision (with the exception of softmax); have you considered other data types, such as integer types or floating-point formats like FP8?
4)	How could your framework be extended to handle multi-GPU operations, such as Fully Sharded Data Parallel (FSDP) for split operations? This seems like a natural extension of the producer-consumer model.
5)	You compare yourself against Triton, which also supports AMD GPUs. Can you address this as a potential tradeoff in the paper? Alternatively, if your framework can be trivially extended to ROCm, this should be included in the paper with a demonstration, otherwise it represents a tradeoff between efficiency and portability.
6)	Your cost model in Section 2.2 is effectively a Roofline model; could you contextualize this in the existing literature? The results in Table 3 are expected, as reordering increases the arithmetic intensity (FLOPs/Byte) of the inner loops.
7)	Throughout the paper, the emphasis on industry versus academic adoption (including the use by undergraduates) feels extraneous and detracts from the main narrative. The paper’s contributions should stand on their own without reliance on external endorsements or applications.
8)	Figures 2 and 5 present a simplified sketch for softmax, whereas the true implementation is significantly more complex, potentially leading to a misleading comparison with PyTorch.  Furthermore, Figure 2 led me to question why you are using C at all for the API, when the listing could easily have been captured by a python trace (e.g. Triton). This design choice is only clarified upon reviewing the implementation details provided in the appendix and supplementary material.

To build on these questions, the feedback below addresses specific technical details and aims to enhance overall clarity. While this paper presents a strong contribution toward improving kernel efficiency, addressing these points will better showcase the authors’ contributions.

Minor Technical Errors:

-	044: The H100 datasheet shows a 7.4x ratio between TCs and ALUs, not 16x. Additionally, my understanding is that the TCs necessarily require bubbles as the Register path cannot keep up with the TC I/O for full throughput. 
-	136: This should be "can load" or "may load" instead of "loads." In general, a kernel does not necessarily need to load data from memory. Kernels can rely solely on arguments (loaded into registers at startup) to generate new data. For example, a kernel might generate a pseudo-random noise tensor without accessing memory.
-	148: The 32 threads must be within the same quadrant, where “consecutive” or “adjacent” would be more appropriate than “nearby”.
-	150: In Ampere, a warp cannot simultaneously occupy different functional units, though separate warps can. For accuracy, please verify this claim against the Hopper documentation or micro-benchmarking paper, otherwise consider omitting if verification is unavailable.
-	167: Excess registers spill over into Global Memory, not L1. They can appear in L1 due to the memory hierarchy, but this is at the discretion of the hardware cache manager.
-	171: Multiple thread blocks can only schedule on the same SM if there is sufficient space (e.g. SMem), otherwise they would clobber each other.
-	173: This statement should be more precise to mention “all thread blocks” and that the L2 is hardware managed, making it distinct from the software managed SMem.
-	179: The tail-effect cost mentioned only applies to singular kernels. Ideally the GPU should have multiple kernels in flight, which can run concurrently.
-	It would also be relevant to mention that kernels which contain too many instructions can cause slowdown as they will incur ICache misses.

Presentation Issues:

-	The abstract should be revised for clarity, with suggested improvements like “creates a”, “suggest that”, and “resembling PyTorch.”
-	The paper could benefit from clarity revisions in several sections, where phrasing and word choice could make technical details easier to follow. Lines: 073, 170, 178, 205, 278, 299, 301, 328, 370, 397
-	325: You should not use "[1]" and "[2]" to enumerate concepts as they are easily confused with reference indicators. 
-	Table 2 and Table 3 should probably be Figures like Figure 6. It is also unclear why these stop at 4 stages, K=1024, and what K is. (MxN)x(NxK)? 
-	Figure 7 and 8 should use subfig captions rather than plot titles. If parameters are common among subfigures, then they should be stated in the figure caption, otherwise in the subfig caption. The fontsize for the axis and labels is too small. Finally, the batch size does not match with the titles and caption.
-	The table in Section 4.2 is missing a caption and column (TK is listed twice).
-	The reference links are broken in Appendix B.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a new programming library for implementing efficient CUDA kernels. tThe paper contains the three ideas at three different levels for CUDA kernel implementation: (1) At warp-level, the author proposes to organize the tile as the multiple of 16; (2) at thread-block level, the author devises template libraries to overlap between different asynchronous warps; (3) at grid-level, the author proposes methods for managing kernel launch kernel launching overheads. As a result, the proposed library can achieve a performance on par with the existing state-of-the-art implementations.

### Strengths
* The paper proposes methods at different levels that simplify CUDA kernel implementations
* The paper can achieve a similar performance compared to the state-of-the-art implementation

### Weaknesses
* The paper has not discussed the tunning overhead with the proposed techniques.

### Questions
Thanks for submitting the excellent paper to ICLR. While in general I enjoyed reading the paper, I have a few thoughts on the extension of the paper. Specifically, this paper proposes a new CUDA abstraction that allows users to write new kernels. However, it seems that it is built on top of the fact that all the dimensions should be a multiple of 16. This could be problematic in the context of dynamic shapes where the dimension does not divide 16. Could you please elaborate on how could the proposed technique be extended to such cases?

Besides, the paper uses auto-tuning for further adjust the hyperparameters for a better performance. Could you elaborate how much the tunning overhead is?

### Soundness
3

### Presentation
3

### Contribution
3
