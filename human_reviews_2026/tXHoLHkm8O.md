# CCKS: Cooperative CPU-GPU Scheduling for Fused Kernels on Coherent Architectures

- Decision: Reject
- Scores: 6, 4, 6, 0

## Abstract
Executing modern ML workloads as sequences of discrete GPU kernels leads to significant hardware underutilization because of kernel launch, data movement, and CPU-GPU synchronization overheads. Recent advancements in kernel fusion reduce small kernel launch overhead by consolidating many small kernels into a single, persistent kernel. However, existing fusion techniques delegate complex scheduling logic to the GPU itself—a task for which its architecture is ill-suited. This on-GPU scheduling creates critical inefficiencies, as its control-intensive, synchronization-heavy logic is fundamentally mismatched with the GPU's parallel microarchitecture, and leads to stalled threads during synchronization, and high-overhead collection of global state.

We propose CCKS (Cooperative Coherent Kernel Scheduler), a novel framework that leverages tightly-integrated, cache-coherent CPU-GPU architectures such as the NVIDIA Grace Hopper Superchip for fused kernel scheduling. CCKS offloads the scheduling of fused kernels to the host CPU, treating it as a dedicated co-processor. In our design, the GPU's role is simplified to that of an efficient information provider and decision executor. This division of labor is enabled by a near-zero overhead, cache-coherent interface that exposes GPU runtime state and allows the CPU to make and propagate scheduling decisions asynchronously concurrently. To facilitate our approach, we introduce an innovative programming framework that automatically generates the requisite CPU scheduler and GPU code from a high-level description. Our evaluation shows that CCKS achieves up to 77% performance improvement over state-of-the-art kernel fusion frameworks on representative ML workloads.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents CCKS, a framework for fused kernel scheduling on tightly coupled CPU/GPU systems, like Grace Hopper.

### Strengths
+ Impressive engineering, and great results on a challenging problem to improve performance on shared-memory systems.

### Weaknesses
- Can this technique generalize beyond Nvidia Grace Hopper?
- How much of the benefit is truly due to CCKS, versus the underlying shared-memory paradigm of Grace Hopper?

### Questions
At a high level, I like this idea, and I believe the authors did a solid job explaining the challenges and presenting their solution. My main question is, "Isn't something like this supposed to already exist on Grace Hopper?" The fact that it doesn't (if it doesn't) highlights that this is important work. However, under the hood, I'm struggling to understand if the speed up is indeed due to CCKS's improved scheduler, or the fact that Grace Hopper is the first, real, shared memory system between CPU/GPU? By virtue of having such direct shared memory, how much benefit is CCKS providing to the end user, versus the CUDA compiler eventually enabling CCKS's ideas?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper leverages the cache coherence in modern CPU–GPU systems such as the NVIDIA Grace Hopper Superchip to develop a cache-coherent kernel scheduling framework (CCKS). The framework introduces three techniques: speculative enqueue, batch commit, and CPU bypass, to improve scheduling efficiency. CCKS is integrated into two existing LLM inference systems, Pod-Attention and Mirage.

### Strengths
The motivation is clearly articulated and grounded in practical limitations of existing on-GPU scheduling.

The paper is generally well written and easy to follow.

The proposed optimisation techniques are simple yet effective. For instance, in speculative enqueue, when the speculation is incorrect, the committed task can be simply overwritten by the correct task.

### Weaknesses
- Grace Hopper is no longer NVIDIA’s latest GPU architecture. It would be good to discuss whether CCKS remains applicable to newer generations such as Blackwell, and what hardware assumptions (e.g., cache coherence model or interconnect behaviour) are required.

- The integration details of CCKS within Pod-Attention and Mirage are limited. More implementation specifics on how speculative enqueue, batch commit, and CPU bypass are realized within these frameworks would improve clarity.

- The paper does not explicitly discuss characteristics of ML workloads that make them particularly suited to CCKS. The current design appears general to GPU workloads. Please elaborate on why ML inference workloads especially benefit from the proposed mechanisms, or what properties (e.g., kernel granularity, dependency patterns) motivate this focus.

- From a venue-fit perspective, this paper is primarily a systems contribution aimed at improving runtime efficiency rather than a study of representation learning. Its relevance to ICLR may therefore be a question.

### Questions
The paper states that “traditional inference systems for LLMs execute a sequence of discrete GPU kernels to perform computation.” Could the authors provide a more concrete example illustrating how LLM inference generates a sequence of small kernels, which makes the associated launch and teardown overheads become non-negligible?

When mentioning that “each kernel launch incurs a setup and teardown cost,” it may strengthen the presentation to explicitly link the setup phase to the enqueueing process on the GPU command queue, which is one of the key motivations for the proposed speculative enqueue mechanism.

In speculative enqueue, when a speculation is incorrect, the committed task is overwritten by the correct one. Are there any negative side effects (e.g., wasted memory traffic, resource contention, or timing delays) from frequent mis-speculations? A discussion or quantitative measurement would be helpful.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a new scheduling framework, CCKS, for fused persistent GPU kernels that leverages emerging cache-coherent CPU–GPU architectures such as NVIDIA’s Grace Hopper with low latency high bandwidth links. Recent fused kernel systems execute scheduling logic on the GPU that cause inefficiency due to control-heavy, serial scheduling running on SIMT hardware. CCKS instead offloads this scheduling to the CPU, with optimizations such as speculations and scheduling bypass to make this more efficient.

### Strengths
* Tackles an important and challenging problem 
* An interesting un-intuitive approach to offload scheduling to the CPU 
* Reasonable approach that uses speculation and CPU bypass when needed
* The paper is well motivated
* Significant speedups over baseline approaches

### Weaknesses
* The approach relies on a ultra-low latency, high-bandwidth, cache-coherent interconnect between the CPU and GPU. It would be good to see what the impact of latency is and when this approach becomes feasible. 
* While the proposed approach is quite interesting, it does add a lot of scheduling complexity and non-determinism in performance to the scheduling pipeline
* Somewhat narrow in applicability, as this would be only useful when using fused persistent kernels. 
* Some prior works missing, e.g., "ACE: Efficient GPU Kernel Concurrency for Input-Dependent Irregular Computational Graphs", Durvasula et al., PACT 2024
* There are many typos and grammatical errors in the paper. Please fix them.

### Questions
Please comment on/address weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper proposes a framework for scheduling work on the GPU from the CPU exploiting the tight coupling on NVIDIA's Hopper for this purpose.

### Strengths
The paper is relatively easy to read.

### Weaknesses
The paper appears to me to not accurately describe the key related works it sets out to build upon.  To me it appears neither Wu et al., 2025 or Spector et al., 2024 employ on GPU scheduling (which makes little sense).  My understanding is the scheduling in those papers done offline by the compiler when generating the kernels before they are run.  Also, I didn't see mention of persistent kernels in the OSDI paper from Wu et al., 2025.   Similarly, the statement "The core innovation enabling kernel fusion is an on-GPU scheduler" seems inaccurate or needs some clarification.  Kernel fusion can be done statically before running the code.

The optimizations in Figure 4 and 5 are not explained in nearly sufficient detail to understand where the supposed benefits are coming from.   Partly, that can be blamed on the format of ICLR which has a very limited page budget.  More details could have been provided in an appendix.  Or better, yet, submit to a systems conference where you get twice as much space in the main text.  At a systems conference I would expect a more thorough explanation (with data) of the source of the problem being tackled. 

Line 303: "Speculative enqueue: The CPU scheduler speculatively prepares task data structures and copies
them into GPU memory queues in advance, even when prior GPU tasks are still running."  -- GPUs already do this kind of thing since the first CUDA enable GPUs.   The whole point of async memcpy and streams is to allow the CPU to load up work into a ring buffer queue of tasks that are read in by the GPU as the GPU completes work.  GPUs work this way for graphics as well (not just compute).  

I don't see mention of CUDA graphs, which seems related.

Typos: "imporve fused kernel", "tighyly integrated"

### Questions
Is code available?  I didn't see any supplemental materials or links.  

This is a systems paper, which seems a bit outside the normal scope for ICLR.  To my judgment this paper would likely get rejected at flagship systems conferences that are all interested in work on ML, so why should it be published at ICLR?  

Where in Wu et al. (2025) [Mirage paper at OSDI 2025] is there a description that matches the following text from this submission "This model designates one or more SMs to act exclusively as schedulers Wu et al. (2025)."?  The words "designate" and "scheduler" do not seem to appear in the OSDI 2025 paper.

### Soundness
1

### Presentation
2

### Contribution
1
