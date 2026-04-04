# FSA: AN ALTERNATIVE EFFICIENT IMPLEMENTATION OF NATIVE SPARSE ATTENTION KERNEL


**Ran Yan** [1] _[∗]_ **, Youhe Jiang** [1] _[∗]_ **, Zhuoming Chen** [2] **, Haohui Mai** [1] **, Beidi Chen** [2] **, Binhang Yuan** [1]
1. The Hong Kong University of Science and Technology
2. Carnegie Mellon University
ryanaf@connect.ust.hk, youhejiang@gmail.com
_{_ zhuominc, beidic _}_ @andrew.cmu.edu, _{_ haohui, biyuan _}_ @ust.hk


ABSTRACT


Recent advances in sparse attention mechanisms have demonstrated strong potential for reducing the computational cost of long-context training and inference
in large language models (LLMs). Native Sparse Attention (NSA), one state-ofthe-art approach, introduces natively trainable, hardware-aligned sparse attention
that delivers substantial system-level performance boosts while maintaining accuracy comparable to full attention. However, the kernel implementation of NSA
forces a loop order that is only efficient with a relatively large number of query
heads in each Grouped Query Attention (GQA) group, whereas existing LLMs
widely adopt a much smaller number of query heads in each GQA group — such
an inconsistency significantly limits the applicability of this sparse algorithmic
advance. In this work, we propose **Flash** **Sparse** **Attention** **(FSA)**, an alternative kernel implementation that enables efficient NSA computation across a wide
range of popular LLMs with a varied, smaller number of heads in each GQA
group on modern GPUs. Compared to vanilla NSA kernel implementation, our
empirical evaluation demonstrates that FSA achieves (i) up to 3.5 _×_ and on average 1.6 _×_ kernel-level latency reduction, (ii) up to 1.25 _×_ and 1.09 _×_ on average
end-to-end training speedup on state-of-the-art LLMs, and (iii) up to 1.36 _×_ and
1.11 _×_ on average for prefill-phase speedup in LLM generative inference. The
[source code is open-sourced and publicly available at https://github.com/](https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention)
[Relaxed-System-Lab/Flash-Sparse-Attention.](https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention)


1 INTRODUCTION


Large Language Models (LLMs) with long context windows (OpenAI, 2024; Anthropic, 2024;
Young et al., 2024; Dubey et al., 2024) face prohibitive computational costs due to the full attention mechanism’s quadratic time and memory complexity. As sequence length increases, attention
computation becomes a critical bottleneck — for instance, attention can account for 70–80% of total decoding latency at a 64k token context (Yuan et al., 2025). In extreme cases, processing a 1
million-token prompt with an 8B model can take up to 30 minutes on a single GPU (Jiang et al.,
2024a). These observations underscore the urgent need for more efficient attention mechanisms in
long-context LLM training and inference. A recent promising direction is to exploit sparse attention,
whereby the query of each token only interacts with a subset of key and value, dramatically reducing
the computation load and HBM I/O volumes. However, implementing efficient sparse attention at
scale is non-trivial — In fact, the challenge of implementing high-performance kernels has become a
major obstacle to deploying state-of-the-art sparse attention techniques in practice. In this paper, we
want to explore: _Can we design and implement an efficient sparse attention kernel for a wide range_
_of current LLMs to fully unleash the potential of this algorithmic advance over modern GPUs?_


Addressing the above question is crucial because adopting sparse attention in long-context LLMs
could mitigate the quadratic cost and enable new applications (Xu et al., 2025a; Chen et al., 2024;
Acharya et al., 2024; Wang et al., 2024). By leveraging the inherent sparsity of attention patterns,
one can significantly cut down computation and memory overhead. Among such methods, one


_∗_ represents equal contribution. Correspondence to Binhang Yuan.


1


promising example is Natively Sparse Attention (NSA) (Yuan et al., 2025), a recently proposed
sparse attention framework, which organizes keys/values into blocks and processes them via three
parallel attention modules - compressed coarse-grained tokens, selected fine-grained tokens, and
sliding local windows. By learning which tokens to compress or drop, NSA achieves long-context
efficiency without a predefined pattern, making it a natural choice for long-context LLM training.


Nevertheless, implementing an efficient sparse attention kernel, i.e, NSA, is challenging. The core
difficulty lies in implementing the sparse mechanism in NSA (i.e., computing attention score based
on selectively retained fine-grained tokens), where the query of each token needs to dynamically
select a different set of keys and values. Such computation results in irregular HBM access patterns
on modern GPUs, where each query processes distinct selected keys/values, potentially requiring
unnecessary padding for query tiles before executing warp-/warpgroup- level matrix multiply-andaccumulate instructions (e.g., wmma or wgmma), and leading to the underutilization of tensor cores.


This scattered access pattern conflicts with the GPU hardware-efficient design principle: GPUs
achieve their peak mathematical throughput when the warps execute dense (no-padded) matrix multiply and accumulation instructions. Thus, current sparse attention implementations fail to translate
the theoretical floating-point operations (FLOPs) reduction into wall-clock speedups.


Vanilla NSA kernel implements a two-level loop: In the outer loop, NSA kernel loads one token
and batches query attention heads that share the same key and value heads; in the inner loop, NSA
kernel loads selected KV block iteratively and performs attention computation. This strategy reaches
kernel efficiency only when each Grouped Query Attention (GQA) (Ainslie et al., 2023) group
has sufficient number of query heads, so that no-padding is required to execute PTX instructions
(e.g., wmma or wgmma) on modern GPUs. [1] However, such an assumption may not hold for a wide
range of popular LLMs so that the original NSA kernel efficiency could drop considerably. With an
insufficient number of query heads in each GQA group, batching query heads is inefficient to satisfy
this hardware requirement. Thus, the original NSA kernel implementation must pad query attention
heads to meet instruction requirements, resulting in unnecessary data loading and computations.


To resolve this issue, we propose FSA, which implements optimized kernels efficient for NSA under
various GQA group settings. We make the following concrete contributions:


- **Contribution 1:** We propose an alternative implementation for the NSA kernel, which exchanges
the two-level loop order in NSA implementation — FSA loops over KV blocks in the outer loop
and loops over query tokens in the inner loop to accelerate this system bottleneck. Since the
number of query tokens that attend to a given KV block is usually much larger than the hardware
required value, FSA introduces no padding, significantly reducing unnecessary kernel memory
access and FLOPs, thereby facilitating faster token selection kernel execution.

- **Contribution 2:** We analyze the trade-off between vanilla NSA and FSA implementation in terms
of kernel efficiency and memory accessing paradigm, which illustrates the effective design and implementation of FSA. To maximize the performance benefits of FSA kernel design, we implement
dedicated optimizations for query token memory access, which is accessed in the inner loop of
FSA kernel, and employ separate optimized kernels for attention result reduction.

- **Contribution 3:** We conduct empirical studies to compare FSA with vanilla NSA and full attention. Concretely, we benchmark kernel execution latencies, end-to-end training and inference
prefill phase latencies for state-of-the-art LLMs. Compared to NSA, results show that FSA delivers (i) up to 3.5 _×_ and on average 1.6 _×_ kernel-level latency reduction, (ii) up to 1.25 _×_ and 1.09 _×_
on average end-to-end training speedup, and (iii) up to 1.36 _×_ and 1.11 _×_ on average inference
prefill-phase speedup. Compared to full attention, the performance boost is further amplified.


2 PRELIMINARIES AND RELATED WORK


2.1 GPU KERNEL IMPLEMENTATION


**Parallelization** **in** **modern** **GPUs.** Modern GPUs utilize massive threads to execute kernels concurrently. Optimized kernel implementations typically employ two-level parallelism: (i) Thread


1Concretely, performance is downgraded due to hardware requirements on matrix shapes for warp/warpgroup- level matrix multiply-and-accumulate instructions (e.g., wmma or wgmma) (NVIDIA, 2025), where
each dimension of a matrix tile must be larger than specified value (e.g., at least 8 on Hopper GPUs).


2


block-level parallelism: Optimized implementations partition input matrices into multiple tiles, assign them to thread blocks, and execute computations for each thread block in parallel. Common
paradigm within a single thread block follows three key steps: Load matrix tiles into the GPU’s
shared memory; perform computations using the loaded tiles; and store computed results to the output tensor. (ii) Warp-level parallelism: Within each thread block, optimized kernels further partition
assigned matrix tiles to multiple warps — each containing 32 threads on NVIDIA GPUs (NVIDIA,
2024d) - to enable fine-grained parallel execution. Warp-level parallelism maximizes hardware
efficiency through coalesced memory access and implicit synchronization within warps.


**Efficient** **kernel** **implementation.** Modern GPU architectures impose strict requirements on the
shapes of matrix tiles used in low-level computations. Specifically, PTX warp-level matrix multiplyaccumulate instructions (NVIDIA, 2025) require that for matrix multiplication _C_ = _AB_, where
_A_ _∈_ R _[m][×][k]_ and _B_ _∈_ R _[k][×][n]_, the dimensions _m_, _n_, and _k_ must satisfy minimum size requirements
for single-warp processing. On NVIDIA Hopper GPUs, _m_, _n_, _k_ must be at least 8. To achieve
higher efficiency, a thread block typically utilizes multiple warps for sufficient warp-level parallelism. Additionally, modern GPUs perform optimally with coalesced and contiguous data loading
and storing; non-contiguous memory access leads to a lower L2 cache hit rate, thereby reducing
effective memory bandwidth and degrading overall kernel efficiency.


2.2 ATTENTION MECHANISMS


**Full attention.** Full attention with causality (Vaswani et al., 2017; Ainslie et al., 2023)—where each
query token attends to all previous KV tokens—is standard in LLM training and inference. Formally,
given sequence length _N_, query/key head dimension _dK_, value head dimension _dV_, _h_ query heads,
and _hK_ KV heads, attention computation involves query/key/value tensor **Q** _∈_ R _[N]_ _[×][d][K]_ _[×][h]_, **K** _∈_
R _[N]_ _[×][d][K]_ _[×][h][K]_, **V** _∈_ R _[N]_ _[×][d][V][ ×][h][K]_ . For _j_ -th ( _j_ _∈{_ 0 _,_ 1 _, ..., h−_ 1 _}_ ) query head, _⌊j/hK⌋_ -th (ranging from
0 to _hK_ -1) key and value head, denote involved matrices as **Q** _[j]_ _,_ **K** _[⌊][j/h][K]_ _[⌋]_ _∈_ R _[N]_ _[×][d][K]_ _,_ **V** _[⌊][j/h][K]_ _[⌋]_ _∈_
R _[N]_ _[×][d][V]_ . Full attention computation can be formalized as:


    - **Q** _j_ ( **K** _⌊j/hK_ _⌋_ ) _T_
**O** _[j]_ = Softmax ~~_√_~~
_dK_


**V** _[⌊][j/h][K]_ _[⌋]_ (1)


On the system side, recent research (Dao, 2023; Kwon et al., 2023) has optimized full attention from
various perspectives. Notably, Flash Attention (Dao, 2023) optimizes full attention with a two-level
loop: Each thread block loads a block of query tokens and, while KV tokens remain, iteratively processes a block of KV tokens and accumulates intermediate results with online softmax (Milakov &
Gimelshein, 2018). Results are finally written to the output tensor. This design minimizes redundant
memory accesses for query and output tensors, thereby reducing attention execution latency.


**Sparse attention.** Recent efforts in sparse attention algorithms (Yuan et al., 2025; Lu et al., 2025;
Lee et al., 2023; Tay et al., 2020; Zhao et al., 2019; Tang et al., 2024; Xiao et al., 2024b; Zhu
et al., 2024; Lai et al., 2025; Xu et al., 2025b; Zhang et al., 2023) and system side optimizations efforts (Zhang et al., 2024b;a; 2025b) represent an emerging trend aimed at reducing attention computation costs in long-context LLM training and inference, where standard attention performs poorly due to its quadratic complexity with respect to sequence length. The most notable
efforts in sparse attention include Native Sparse Attention (NSA) (Yuan et al., 2025). Formally,
in NSA, for _j_ -th query head, each query token **q** _[j]_ _t_ _[∈]_ [R][1] _[×][d][K]_ _[, t]_ _[∈{]_ [0] _[,]_ [ 1] _[, ..., N]_ _[−]_ [1] _[}]_ [attends] [to]
_N_ ˜ _≪_ _N_ KV tokens via three attention mechanisms _c_ _∈C_, where _C_ = _{_ cmp _,_ sel _,_ win _}_, representing compression, selection, and sliding window for keys and values. We denote KV tokens
as **K** **[˜]** _[⌊]_ _c_ _[j/h][K]_ _[⌋]_ _∈_ R _N_ [˜] _×dK_ _,_ **˜V** _c_ _[⌊][j/h][K]_ _[⌋]_ _∈_ R _N_ [˜] _×dV_, which contains _⌊j/hK⌋_ -th KV head and a subset of
KV tokens of attention mechanism _c_ . Given trainable gating scores _τt_ _[c]_ _[∈]_ [[0] _[,]_ [ 1]] [for] [three] [attention]
modules, NSA combines the three attention mechanisms as follows:


**o** _[j]_ _t_ [=] - _τt_ _[c]_ _[·]_ [ Softmax]

_c∈C_


- **q** _[j]_ _t_ [(] **[ ˜K]** ~~_√_~~ _⌊cj/hK_ _⌋_ ) _[T]_
_dK_


**V** ˜ _c_ _[⌊][j/h][K]_ _[⌋]_ (2)


Notably, the NSA kernel that selectively retains fine-grained tokens is a major system bottleneck
across three attention mechanisms. This point is validated in §4.4. The NSA kernel allows each


3


query token across query heads that share the same KV heads to attend to distinct _T_ KV blocks,
each with _BK_ contiguous KV tokens. Distinct KV block selection imposes challenges on effectively batching query tokens and performing computation with KV blocks within one thread block.
Therefore, it is crucial to optimize the batching strategy for efficient NSA kernel execution.


3 FLASH SPARSE ATTENTION


We present FSA design and compare with vanilla NSA (§3.1), then introduce FSA implementation
and optimizations (§3.2). Finally, we provide a thorough analysis of FSA performance (§3.3).


3.1 FSA KERNEL DESIGN


An efficient sparse attention kernel must translate theoretical FLOPs reduction into concrete savings
in memory access and computation during GPU execution. Vanilla NSA kernel is insufficient in
achieving this goal. As illustrated in Figure 1 (left), NSA kernel processes query tokens one by one
in the outer loop and KV blocks in the inner loop, while batching query heads. However, if the
number of query heads is insufficient, this method requires padding to meet the hardware’s matrix
multiplication shape requirements, leading to wasteful memory access and computation.


To achieve higher kernel efficiency, FSA exchanges NSA kernel loop order and processes query
heads one by one, looping over KV blocks in the outer loop and batches of query tokens in the inner
loop. Since the number of such tokens is typically large enough to meet hardware requirements, this
strategy requires no padding and eliminates the overhead of processing padded data.


However, due to inversion of kernel loop order, FSA encounters new challenges:


- **Non-contiguous** **memory** **access** **for** **query** **batches.** Due to the sparse nature of NSA token
selection, for one KV block, only a subset of total query tokens is involved for attention computation, and query token indices are typically non-contiguous. When processing query tokens in
FSA inner loop, it is critical to minimize the negative impact of non-contiguous memory access.


- **Online** **softmax** **statistics** **and** **attention** **results** **accumulation.** Online softmax and attention
results reduction for each query token across distinct KV blocks adds another layer of complexity.
In the NSA token selection logic, computing the final output for a query token requires accumulating partial attention results from its distinct selected KV blocks. Since the NSA kernel’s outer
loop iterates over query tokens, this accumulation process can be handled within one thread block.
In contrast, FSA’s inverted loop order means that partial results for a single query are computed
across different thread blocks, each processing a different KV block. This design necessitates a
proper management strategy for accumulating attention results distributed across thread blocks.


3.2 FSA KERNEL IMPLEMENTATION AND OPTIMIZATION


To implement an efficient FSA kernel, we employ an optimized token selection kernel that minimizes the negative impact of non-contiguous memory access. Additionally, an online softmax and
reduction kernel are designed to efficiently handle online softmax and attention result reduction.


**FSA** **token selection kernel.** FSA _mitigates the impact of non-contiguous memory access by em-_
_ploying index tensors to orchestrate data movement._ During forward pass, as illustrated in Figure 1
(right), each thread block in FSA kernel is assigned a single (Query Head, KV Block) pair. The KV
block is loaded from main memory once per thread block. The kernel then iterates through batches
of non-contiguous query tokens, which are loaded and stored using index tensors _Ii_ and _Oi_ for
_i_ _∈{_ 1 _,_ 2 _, ..., b}_, where _b_ is the total number of KV blocks. These index tensors are pre-computed
from the NSA sparse selection tensor **T** _∈_ R _[h][K]_ _[×][N]_ _[×][T]_, which records selected KV block indices for
each query token. Due to the sparse nature of token selection, each KV block is attended by a subset
of _N_ query tokens. Consequently, index tensor _Ii_, which contains query token indices attending to
current KV block, typically holds fewer than _N_ valid indices, i.e., _N_ valid = _|Ii|_ _≤_ _N_ . To minimize
the impact of non-contiguous memory access, a thread block terminates early once it has processed
all valid query indices in _Ii_, avoiding further memory access or computation. Concurrently, index
mapping tensor _Oi_ facilitates contiguous storage of intermediate results. Note that outputs from
FSA token selection kernel are not final attention scores; they are partial results that are reduced


4


**K**


**K**
_N_ × 𝑑! ×ℎ!


**Inner Loop** **Q**


**Q**
_N_ × 𝑑! × _h_


**Output**
_N_ × _h_ × 𝑑"


**Inner Loop**


**V**
_N_ × 𝑑" × ℎ!


𝐎𝐛𝐮𝐟
𝑁+,-./ × _b_ × 𝑑" × _h_

**Grid Loop**


Figure 1: Left: Illustration of NSA kernel (Yuan et al., 2025), which iterates query tokens in the outer loop, and
processes KV blocks in the inner loop. Right: Illustration of FSA kernel, which alternatively iterate KV blocks
in the outer loop, and processes query tokens in the inner loop — partial attention results are stored in output
buffer **O** buf for accumulation (see §3.2 for more details).


for each query across different KV blocks in a separate reduction kernel, which we introduce next.
In the backward pass, FSA kernel follows a similar logic, loading query tokens non-contiguously
to compute gradients and storing intermediate gradients to buffers. The primary difference is that
index tensors _Ii_ and _Oi_, computed during the forward pass, are retrieved from cache.


FSA _handles query attention results and gradients reduction in separate kernels._ In forward pass,
FSA parallel computation of attention scores - where a single query token’s results are reduced
across multiple KV blocks - requires a careful implementation of online softmax and reduction
logic to ensure numerical correctness. In the backward pass, a similar reduction challenge exists for
gradients of query tokens. FSA achieves efficient and correct accumulation in two kernels:


**FSA reduction kernel.** Since a query’s attention scores or gradients are computed across multiple
thread blocks (each processing a different KV block in FSA token selection kernel), direct reduction
into the output tensor in FSA kernel necessitates atomic additions (NVIDIA, 2024a) to prevent race
conditions. Given the prohibitive overhead of atomic operations, FSA decouples computation from
accumulation. It adopts a two-stage process:


- (i): FSA token selection kernel (see Figure 1 (right)) computes partial query attention results or
gradients without reduction with online softmax and writes them to an intermediate buffer.

- (ii): A dedicated reduction kernel efficiently accumulates these partial results into a final output
tensor with online softmax scaling, which we introduce next.


This two-stage arrangement effectively eliminates atomic operations and achieves efficient attention
result accumulation. However, HBM memory overhead is increased due to intermediate buffers. To
minimize memory overhead, we allocate a buffer sized only for _N_ valid query tokens relevant to each
KV block, rather than for all _N_ tokens. Index mapping tensor _Oi_ facilitates contiguous I/O into this
compact buffer, thereby avoiding the significant overhead of allocating a full-sized buffer for each
KV block. We present a detailed analysis of FSA buffer HBM memory overhead in Appendix E.


**FSA** **online** **softmax** **kernel** . In the forward pass, to ensure numerical correctness, FSA needs to
include online softmax statistics in two aspects:


- (i): In the FSA token selection kernel, computation results between each query token and key
block must be scaled with _historical_ running maximum (Milakov & Gimelshein, 2018)).

- (ii): In the reduction kernel, partial attention outputs of query tokens regarding selected KV blocks
stored in the output buffer must be scaled with online softmax statistics. Additionally, final output
for a query token must be scaled with log-sum exponentials (Milakov & Gimelshein, 2018).


Computing online softmax statistics within the FSA token selection kernel produces incorrect attention results. When multiple thread blocks process the same query token, each block computes only


5


_partial_ statistics, leading to incorrect maximum values and attention outputs. To address this challenge, FSA introduces a separate online softmax kernel that pre-computes online softmax statistics
using query and key tensor **Q** and key tensor **K** and stores them in a buffer.


3.3 FSA PERFORMANCE ANALYSIS


kernel performance:


_computation overhead?_


To answer this question, we conduct detailed
memory footprint and computation load analysis and derive the following theorem:


Figure 2: Comparison on memory access and FLOPs,
block size is 64, top-k is 16. FSA’s memory volume or
FLOPs are normalized to 1.


**Theorem:** _Across_ _popular_ _GQA_ _group_ _settings,_ _where_ _each_ _GQA_ _group_ _contains_ _g_ _∈{_ 1 _,_ 2 _,_ 4 _,_ 8 _}_
_query heads, aggregate memory access volume and FLOPs of_ FSA _token selection, online softmax,_
_and_ _reduction_ _kernel_ _are_ _lower_ _than_ _vanilla_ _NSA_ _kernel._ Comparisons are presented in Figure 2.


ing analysis of empirical results:


_mance_ _of_ FSA _._ Optimized FSA outperforms under block size _BK_ = 64, and top-k value _T_ = 16.
vanilla NSA across popular GPU architectures FSA latency is normalized to 1.
and GQA group settings, despite being compromised by non-contiguous memory access, and reducing attention results in a separate kernel. When
each GQA group contains fewer than 8 query heads, FSA usually demonstrates superior performance to NSA. These empirical results demonstrate that FSA kernel’s performance gains from
overall reduced unnecessary memory access and FLOPs more than compensate for the overhead of
non-contiguous memory access and executing multiple kernels.


under block size _BK_ = 64, and top-k value _T_ = 16.
FSA latency is normalized to 1.


4 EVALUATION


This section presents a comprehensive evaluation of FSA across various NSA configurations. We
aim to investigate the following research questions:


- _Q1:_ _What is the kernel-level performance of_ FSA _compared with NSA and full attention across_
_diverse NSA algorithmic configurations?_


- _Q2:_ _What is the impact of_ FSA _on end-to-end training and inference performance in practice?_


- _Q3:_ _What is the breakdown performance of_ FSA _, and how effective is each part of_ FSA _?_


6


3446

2585

1723

862


0

|Col1|Col2|Col3|6×|Col5|Col6|
|---|---|---|---|---|---|
||||6×|||
|4×|1×|.2×|×3.|||
|1.7×0.|2.3×1.|2.6×2|2.3|||

8 16 32 64


|Col1|Col2|Col3|×|Col5|Col6|
|---|---|---|---|---|---|
||||×|||
|5×|2×|.5×|4.6|||
|1.2×0.|1.6×1.|1.7×2|1.7×|||


8 16 32 64


|Col1|Col2|Col3|×|Col5|Col6|
|---|---|---|---|---|---|
||||×|||
|7×|6×|.3×|5.5|||
|1.0×0.|1.2×1.|1.3×3|1.3×|||


8 16 32 64


|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|1×|1×|.8×|6.1×|||
|1.0×1.|1.0×2.|1.0×3|1.0×|||


8 16 32 64


3446

2585

1723

862


3446

2585

1723

862


3446

2585

1723

862


3446

2585

1723

862


GQA=1, (128, 8), H20


GQA=2, (128, 8), H20


GQA=4, (128, 8), H20


GQA=8, (128, 8), H20


0

|Col1|Col2|Col3|×|Col5|Col6|
|---|---|---|---|---|---|
||||×|||
|5×|1×|.4×|×3.8|||
|2.4×0.|3.0×1.|3.5×2|3.0|||

8 16 32 64


0

|Col1|Col2|Col3|×|Col5|Col6|
|---|---|---|---|---|---|
||||×|||
|5×|3×|.6×|4.9|||
|1.5×0.|2.1×1.|2.2×2|2.2×|||

8 16 32 64


0

|Col1|Col2|Col3|×|Col5|Col6|
|---|---|---|---|---|---|
||||×|||
|6×|7×|.4×|5.8|||
|1.1×0.|1.5×1.|1.7×3|1.6×|||

8 16 32 64


0

|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|9×|9×|.6×|6.4×|||
|1.0×0.|1.0×1.|1.1×3|1.2×|||

8 16 32 64


1315

986

657

329


1315

986

657

329


1315

986

657

329


1315

986

657

329


GQA=1, (64, 16), H200


GQA=2, (64, 16), H200


GQA=4, (64, 16), H200


GQA=8, (64, 16), H200


0

|Col1|Col2|Col3|×|Col5|Col6|
|---|---|---|---|---|---|
||||2.4×|||
||||2.4×|||
|.2×|.8×|1.4×|.1×|||
|1.2×0|2.6×0|2.4×|2|||

8 16 32 64


0

|Col1|Col2|Col3|5×|Col5|Col6|
|---|---|---|---|---|---|
||||5×|||
|.3×|.9×|2.0×|×3.|||
|1.0×0|1.3×0|1.6×|1.6|||

8 16 32 64


0

|Col1|Col2|Col3|×|Col5|Col6|
|---|---|---|---|---|---|
||||×|||
|6×|.8×|.5×|×4.3|||
|1.0×0.|1.0×1|1.2×2|1.2|||

8 16 32 64


0

|Col1|Col2|Col3|×|Col5|Col6|
|---|---|---|---|---|---|
||||×|||
|9×|.7×|.2×|5.1|||
|1.0×0.|1.0×1|1.0×3|1.0×|||

8 16 32 64


GQA=1, (128, 8), H200


|Col1|Col2|Col3|.7×|Col5|Col6|
|---|---|---|---|---|---|
||||.7×|||
|.2×|.6×|1.5×|6×2|||
|1.5×0|1.1×0|2.9×|2.|||


|Col1|Col2|Col3|×|Col5|Col6|
|---|---|---|---|---|---|
||||×|||
|3×|.9×|.3×|×3.8|||
|1.2×0.|1.7×0|2.3×2|2.1|||


|Col1|Col2|Col3|×|Col5|Col6|
|---|---|---|---|---|---|
||||×|||
|5×|0×|.7×|4.6|||
|1.0×0.|1.2×1.|1.6×2|1.5×|||


|Col1|Col2|Col3|×|Col5|Col6|
|---|---|---|---|---|---|
||||×|||
|7×|4×|.9×|4.9|||
|1.0×0.|1.0×1.|1.1×2|1.1×|||


GQA=2, (128, 8), H200


GQA=4, (128, 8), H200


GQA=8, (128, 8), H200


kernels under block sizes and top-k values of ( _BK_, _T_ ) equals to (64, 16) and (128, 8).


4.1 EXPERIMENTAL SETUP


**Experimental** **setups.** We use two GPU types for evaluations: NVIDIA H20 GPUs (NVIDIA,
2024b), which provide 148 TFLOPS tensor core computational power and 4 TB/s memory bandwidth; and NVIDIA H200 GPUs (NVIDIA, 2024c), which deliver 989 TFLOPS tensor core computational power and 4.8 TB/s memory bandwidth. For end-to-end training and inference evaluations,
GPUs are interconnected via NVLink, providing 450 GB/s inter-GPU bandwidth. In our evaluations,
we use BF16 for training and FP16 for inference.


**Baselines.** We compare FSA with two baselines:


- **NSA (Native Sparse Attention) (Yuan et al., 2025).** Our primary baseline is vanilla NSA implementation, which introduces natively hardware-aligned trainable sparse attention. NSA maintains
algorithmic performance comparable to full attention while substantially reducing computational
complexity. We utilize Triton-based NSA kernel (Organization, 2024) for evaluation.


- **Full attention (Flash Attention) (Dao, 2023).** Due to limited hardware resource utilization, theoretical FLOPs reductions achieved by NSA or FSA may not translate to proportional performance
gains. Therefore, the full attention baseline (with causality), which has no sparsity constraints, is
essential to demonstrate the practical effectiveness of both NSA and FSA. We utilize an efficient
Triton-based Flash Attention kernel(Triton, 2024) for fair comparison.


**Experimental** **configurations.** To ensure comprehensive evaluation, we systematically test FSA
and two baselines under varying NSA configurations: (i) GQA settings _g_ _∈{_ 1 _,_ 2 _,_ 4 _,_ 8 _}_, where _g_ is
number of query heads in one GQA group; (ii) NSA hyperparameter block size _BK_ and top-k _T_
combinations of ( _BK, T_ ) _∈{_ (64 _,_ 16) _,_ (128 _,_ 8) _}_ ; and (iii) sequence lengths of _{_ 8K _,_ 16K _,_ 32K _,_ 64K _}_
tokens. [2] For end-to-end training and inference evaluations, we evaluate performance using Llama38B (Dubey et al., 2024), Qwen3-14B (Yang et al., 2025), and Qwen2.5-32B (Team, 2024) with


2More experiments on ultra long sequence lengths, i.e., 128K and 256K sequence lengths, are presented in
Appendix F.


7


sequence lengths of 32K and 64K. When the entire model is too large to fit on a single GPU for
training, we use pipeline parallelism (Shoeybi et al., 2019) to distribute model across multiple GPUs.


**Evaluation metrics.** Following established practices in prior research (Yuan et al., 2025; Lu et al.,
2025; Dao, 2023), we employ two metrics to evaluate system efficiency: (i) Kernel execution latency,
which measures computational time required for attention operations, and (ii) training and inference
latency, which measures end-to-end time required to process a single batch of data during model
training and inference. These metrics directly assess FSA’s computational efficiency.


4.2 FSA KERNEL BENCHMARKING RESULTS (Q1)


**FSA kernel performance.** We evaluate the kernel performance of FSA across both H20 and H200
GPUs under various configurations. In this section, we evaluate FSA on one single GPU, while we
present distributed evaluations of FSA in Appendix I. As shown in Figure 4, the evaluation results
demonstrate that FSA outperforms both NSA and full attention across most of the tested scenarios:


- **Comparison with NSA.** FSA outperforms NSA with significantly lowered memory access volume and FLOPs in NSA token selection module, despite introducing non-contiguous memory
access and auxiliary kernels (see details in §3). FSA achieves up to 3.5 _×_ speedup and on average
1.8 _×_ lower kernel latency on H20 GPUs, and up to 2.9 _×_ speedup and on average 1.4 _×_ lower kernel latency on H200 GPUs compared to NSA. Performance gap between FSA and NSA widens
with smaller GQA group settings ( _g_ _∈{_ 1 _,_ 2 _}_ ) and longer sequence lengths (32K and 64K tokens),
with peak performance improvement of 3.5 _×_ observed at _g_ = 1 (one query head in one GQA
group) and sequence length of 32K tokens. Furthermore, FSA maintains consistent performance
improvements across different NSA algorithmic configurations, e.g., where ( _BK, T_ ) = (64 _,_ 16)
and ( _BK, T_ ) = (128 _,_ 8), demonstrating robust efficiency gains across diverse parameter settings.


- **Comparison with full attention.** For long sequences, FSA outperforms full attention with an efficient NSA algorithm and even more efficient token selection. FSA achieves up to 6.4 _×_ speedup
and on average 2.4 _×_ lower kernel latency on H20 GPUs, and up to 4.9 _×_ speedup and on average
2.3 _×_ lower kernel latency on H200 GPUs compared to full attention. Performance gap between
FSA and full attention increases dramatically with a larger number of query heads in each GQA
group, with the most substantial improvement of 6.4 _×_ observed at _g_ = 8 (8 query heads in one
GQA group) and sequence length of 64K tokens. Similarly, FSA maintains superior efficiency
across ( _BK, T_ ) _∈{_ (64 _,_ 16) _,_ (128 _,_ 8) _}_ settings, demonstrating consistent and substantial performance advantages over full attention. On the other hand, vanilla NSA lags behind full attention
in many tested cases, even with its sparse attention mechanism. For example, when the sequence
length is 32K, one GQA group contains one query head, NSA consistently falls short of full attention, while FSA demonstrates superior performance than full attention.


4.3 END-TO-END PERFORMANCE COMPARISON (Q2)


**End-to-end** **training** **performance.** We benchmark end-to-end training performance of FSA
against NSA and full attention across various models and hardware setups. As shown in Figure
5, results demonstrate that FSA consistently reduces training latency across all evaluated cases.
Specifically, FSA achieves up to 1.25 _×_ speedup and on average 1.09 _×_ speedup compared to NSA,
and delivers up to 2.47 _×_ speedup and an average of 1.86 _×_ speedup compared to full attention. These
efficiency gains are pronounced with longer sequences and on higher-performance hardware like the
H200, demonstrating FSA’s effectiveness in accelerating computation-intensive training scenarios.


**Inference** **performance.** For prefill latency, we benchmark FSA against NSA and full attention
across various models and hardware setups. As shown in Figure 6, our results demonstrate that FSA
achieves lower prefill latency across most evaluated configurations. Specifically, FSA achieves up
to 1.36 _×_ speedup and on average 1.11 _×_ speedup compared to NSA. FSA performance advantages
are even more significant when compared to full attention, where FSA delivers up to 1.69 _×_ speedup
and an average of 1.39 _×_ speedup. Taken together, these results underscore FSA’s efficacy in accelerating the prefill phase of LLM inference. [3] In terms of decoding latency, FSA matches that of
NSA, which reduces memory access of the decoding phase by only loading a sparse subset com

3We present more detailed decoding evaluations in Appendix G.


8


FSA NSA Full Attention


FSA NSA Full Attention


85.5
64.1

0.0

|H20,|, Llama3- (64,16)|Col3|127% -8B|
|---|---|---|---|
||||127%|
||~~8%~~|~~8%~~||
|~~8%~~|68%|||

32 64


28.3
21.2

0.0

|H200|0, Llama3 (64,16)|Col3|135% 3-8B|
|---|---|---|---|
||||135%|
||<br>~~10~~|<br>~~10~~|~~%~~|
|~~6% ~~|44%|||

32 64


85.5
64.1

0.0

|H20,|, Llama3- (128,8)|Col3|131% -8B|
|---|---|---|---|
||~~17~~|~~17~~|131%|
|~~16%~~|69%|||

32 64


|H20,|, Qwen3-1 (64,16)|Col3|115% 14B|
|---|---|---|---|
||||115%|
||61%<br>~~4%~~|61%<br>~~4%~~||
|~~3%~~||||


|H20, Q|Qwen2.5-3 (64,16)|Col3|Col4|89% 32B|
|---|---|---|---|---|
|||||89%|
||37%<br>~~4%~~|37%<br>~~4%~~|37%<br>~~4%~~||
|~~3% ~~|||||


|H20,|Col2|, Llama3- (64,16)|Col4|63% -8B|
|---|---|---|---|---|
|||||63%|
|||48%<br>7%|48%<br>7%||
||9%||||


|H20,|Col2|, Qwen3-1 (64,16)|Col4|58% 14B|
|---|---|---|---|---|
|||||58%|
|||43%<br>4%|43%<br>4%||
||9%||||


|H20, Q|Col2|Qwen2.5-3 (64,16)|Col4|Col5|43% 32B|
|---|---|---|---|---|---|
||||||43%|
|||15%<br>3%|15%<br>3%|15%<br>3%||
||%|||||


22.7

0.0
32 64


6.8

0.0

|H200|Col2|0, Llama3 (64,16)|Col4|42% 3-8B|
|---|---|---|---|---|
|||||42%|
|3%|3%|8%<br>0%|8%<br>0%||
||||||

32 64


22.7

0.0

|H20,|, Llama3- (128,8)|Col3|67% -8B|
|---|---|---|---|
||16|16|%<br>67%|
|19%|50%|||

32 64


|H200|0, Qwen3- (64,16)|Col3|132% -14B|
|---|---|---|---|
||||132%|
||<br>~~5%~~|<br>~~5%~~||
|~~0% ~~|52%|||


|H200,|1 Qwen2.5- (64,16)|Col3|Col4|105% -32B|
|---|---|---|---|---|
|||||105%|
||<br>~~4%~~|<br>~~4%~~|<br>~~4%~~||
|~~0% ~~|39%||||


|H200|Col2|0, Qwen3- (64,16)|Col4|41% -14B|
|---|---|---|---|---|
|||||41%|
|||14%<br>2%|14%<br>2%||
||5%||||


|H200,|Col2|Qwen2.5- (64,16)|Col4|Col5|47% -32B|
|---|---|---|---|---|---|
||||||47%|
|||9%<br>1%|9%<br>1%|9%<br>1%||
||0%|||||


36.8

0.0
32 64


10.6

0.0
32 64


36.8

0.0

|H20,|, Qwen3-1 (128,8)|Col3|63% 14B|
|---|---|---|---|
||12|12|%<br>63%|
|11%|45%|||

32 64


67.9

0.0
32 64


22.1

0.0
32 64


67.9

0.0

|H20, Q|Qwen2.5-3 (128,8)|Col3|Col4|48% 32B|
|---|---|---|---|---|
||<br>~~10%~~|<br>~~10%~~|<br>~~10%~~|48%|
|9%|16%||||

32 64


139.0
104.2

0.0
32 64


44.2
33.1

0.0
32 64


139.0
104.2

0.0

|H20,|, Qwen3-1 (128,8)|Col3|117% 14B|
|---|---|---|---|
||~~11~~|~~11~~|117%|
|~~10%~~|63%|||

32 64


256.2
192.1

0.0
32 64


81.6
61.2

0.0
32 64


256.2

0.0

|H20, Q|Qwen2.5-3 (128,8)|Col3|Col4|92% 32B|
|---|---|---|---|---|
||~~8%~~|~~8%~~|~~8%~~|92%|
|7%|38%||||

32 64


6.8
5.1
3.4
1.7
0.0


28.3
21.2
14.2
7.1
0.0


|H200|0, Llama3 (128,8)|Col3|147% 3-8B|
|---|---|---|---|
||~~25~~|~~25~~|147%|
|~~20%~~|48%|||


|H200|0, Qwen3- (128,8)|Col3|140% -14B|
|---|---|---|---|
||~~16~~|~~16~~|140%|
|~~14%~~|58%|||


|H200,|1 Qwen2.5- (128,8)|Col3|Col4|112% -32B|
|---|---|---|---|---|
||~~13%~~<br>|~~13%~~<br>|~~13%~~<br>|112%|
|~~10%~~|42%||||


|H200|0, Llama3 (128,8)|Col3|69% 3-8B|
|---|---|---|---|
||~~27~~|~~27~~|~~%~~<br>69%|
|20%|15%|||


|H200|0, Qwen3- (128,8)|Col3|51% -14B|
|---|---|---|---|
||~~16~~|~~16~~|~~%~~<br>51%|
|18%|21%|||


|H200,|Qwen2.5- (128,8)|Col3|Col4|59% -32B|
|---|---|---|---|---|
||~~36%~~|~~36%~~|~~36%~~|59%|
|28%|13%||||


32 64
Seqlen (K)


32 64
Seqlen (K)


32 64
Seqlen (K)


32 64
Seqlen (K)


10.6
8.0
5.3
2.7
0.0


32 64
Seqlen (K)


22.1
16.5
11.0
5.5
0.0


32 64
Seqlen (K)


44.2
33.1
22.1
11.0
0.0


81.6
61.2
40.8
20.4
0.0


Figure 5: End-to-end training latency of FSA, NSA,
full attention.


Figure 6: Inference Prefill latency of FSA, NSA, full
attention.


posed of compressed tokens, selected tokens, and recent tokens from a sliding window (Yuan et al.,
2025).


4.4 PERFORMANCE BREAKDOWN & ABLATION STUDIES (Q3)


In this section, we evaluate FSA at both kernel and end-to-end (training or inference) levels. At the
kernel level, we analyze forward and backward performance separately, and examine each of the
three attention mechanisms within NSA: Compression, selection, and sliding window on key/value
tokens. We conduct ablation studies to assess the effectiveness of FSA kernel optimizations. We
validate the implementation correctness of FSA by comparing training loss across FSA, NSA, and
full attention in Appendix D.


**Forward** **and** **backward** **breakdown.** We conduct a detailed breakdown to analyze forward and
backward attention computation latencies of FSA, NSA, and full attention across various NSA configurations. As shown in Figure 7, FSA demonstrates superior performance in both forward and
backward attention computations across all evaluated scenarios. For forward computation, FSA
achieves up to 2.36 _×_ speedup and on average 1.62 _×_ lower latency compared to NSA, and up to
3.23 _×_ speedup and on average 1.83 _×_ lower latency compared to full attention. Backward computation analysis reveals even more pronounced advantages, since FSA avoids computation costs for
index tensors _Ii_, _Oi_ for _i_ -th KV block (see details in §3.2). FSA achieves up to 4.32 _×_ speedup and
on average 2.59 _×_ lower latency compared to NSA, and up to 7.45 _×_ speedup and on average 6.89 _×_
lower latency compared to full attention. Performance improvements remain consistent across different NSA configurations, demonstrating that FSA provides robust efficiency gains.


**Compression, selection, and sliding window breakdown.** We conduct detailed breakdown experiments for the three essential steps in NSA. As demonstrated in Figure 8, the token selection phase
dominates overall attention computation performance, accounting for up to 79% and on average
65% of total attention overhead across all evaluated configurations. And FSA achieves substantial
performance improvements in token selection, delivering up to 7.6 _×_ speedup and on average 3.4 _×_
lower latency compared to NSA in this critical phase. These results highlight that FSA’s primary
performance advantages stem from its efficient handling of token selection computation.


**Ablation** **study** **on** **sparse** **attention** **performance.** We present an ablation study of FSA kernel
performance in Figure 9, where we disable each of additional optimizations of FSA we mentioned
in §3. Results demonstrate that by disabling the inner loop (one thread block for one query batch),
performance of FSA kernel drops by up to 18.9% and on average 11.9%, and by disabling early return optimization, performance drops by up to 25.2% and on average 18.2%. These empirical results
demonstrate the importance of each component of our FSA optimization in enhancing performance.


9


Compressed (FSA)
Compressed (NSA)


Sliding (FSA)
Sliding (NSA)


FSA-fwd
FSA-bwd


NSA-fwd
NSA-bwd


Full Attention-fwd
Full Attention-bwd


Selected (FSA)
Selected (NSA)


0.5

0.4

0.3

0.1


0.6

0.5

0.3

0.2


0.2

0.2

0.1

0.1


3.4

2.6

1.7

0.9

0.0


3.4

2.6

1.7

0.9

0.0


1.3

1.0

0.7

0.3

0.0


1.3

1.0

0.7

0.3

0.0


1.3

1.0

0.7

0.3

0.0


1.3

1.0

0.7

0.3

0.0


H200, GQA=4

(128, 8)


1.2

0.9

0.6

0.3


H20, GQA=1

(64,16)


H20, GQA=4

(64,16)


H200, GQA=1

(64,16)


H200, GQA=4

(64,16)


3.4

2.6

1.7

0.9

0.0


3.4

2.6

1.7

0.9

0.0


H20, GQA=1

(64,16)


H20, GQA=1

(128, 8)


H20, GQA=4

(64,16)


H20, GQA=4

(128, 8)


H200, GQA=1

(64,16)


H200, GQA=1

(128, 8)


H200, GQA=4

(64,16)


0.0
fwd bwd


0.0
fwd bwd


0.0
fwd bwd


0.0
fwd bwd


0.6

0.5

0.3

0.2


0.7

0.5

0.4

0.2


0.3

0.2

0.1

0.1


1.5

1.1

0.8

0.4


H20, GQA=1

(128,8)


H20, GQA=4

(128,8)


H200, GQA=1

(128,8)


H200, GQA=4

(128,8)


0.0
fwd bwd


0.0
fwd bwd


0.0
fwd bwd


0.0
fwd bwd


Figure 7: Experimental breakdown of FSA, NSA,
and full attention latencies during forward and backward computation.


FSA w/o Inner Loop w/o Early Return


Figure 8: Experimental breakdown of token compression, selection, and sliding window attention overhead during forward/backward pass.


MLP (FSA)
Attention (FSA)


MLP (NSA)
Attention (NSA)


MLP (Full Attention)
Attention (Full Attention)


0.16

0.12

0.08

0.04

0.00


0.31

0.23

0.16

0.08

0.00


0.15

0.11

0.08

0.04

0.00


0.30

0.22

0.15

0.07

0.00


H200, GQA=1

Seqlen=64K


78.4
58.8
39.2
19.6
0.0
32 64


(128,8)


H20, Llama3-8B


H20, Qwen3-14B


(128,8)


H20, Qwen2.5-32B


(128,8)


0.25

0.19

0.13

0.06

0.00


0.50

0.38

0.25

0.13

0.00


H20, GQA=1
Seqlen=32K


H20, GQA=1
Seqlen=64K


H20, GQA=4
Seqlen=32K


H20, GQA=4
Seqlen=64K


H200, GQA=1

Seqlen=32K


H200, GQA=4

Seqlen=32K


127.4
95.5
63.7
31.8
0.0
32 64


234.8
176.1
117.4
58.7
0.0
32 64


0.10

0.08

0.05

0.03

0.00


0.18

0.14

0.09

0.05

0.00


H200, GQA=4

Seqlen=64K


25.9
19.5
13.0
6.5
0.0


H200, Llama3-8B

(128,8)


32 64
Seqlen (K)


40.5
30.4
20.2
10.1
0.0


H200, Qwen3-14B

(128,8)


32 64
Seqlen (K)


74.8
56.1
37.4
18.7
0.0


H200, Qwen2.5-32B

(128,8)


32 64
Seqlen (K)


Figure 9: Ablation study (with or without FSA optimizations) on FSA kernel.


Figure 10: Breakdown of computation time for attention and MLP during end-to-end training.


**End-to-end training breakdown.** To isolate the source of performance improvements, we conduct
a breakdown analysis of the end-to-end training latency. As shown in Figure 10, results demonstrate that FSA’s performance improvements originate from attention computation. Within this
component, FSA achieves up to 1.4 _×_ and on average 1.23 _×_ lower latency than NSA, and realizes a
speedup of up to 3.87 _×_ and on average 2.91 _×_ over full attention. This analysis confirms that overall
end-to-end speedup is driven by FSA’s fundamental optimizations in NSA token selection.


5 CONCLUSION


We presented Flash Sparse Attention (FSA), a kernel design that broadens the applicability of Native
Sparse Attention (NSA) to modern LLMs where each GQA group contains a small number of query
heads. By inverting kernel loop order and introducing tailored optimizations for non-contiguous
memory access, online softmax, and accumulation, FSA eliminates padding inefficiencies that limit
NSA on current GPUs. Evaluation demonstrates that FSA achieves substantial improvements in
both kernel-level and end-to-end performance, offering consistent speedups in training/inference
across state-of-the-art long-context LLMs. These results highlight that algorithm–system co-design
is critical for translating theoretical efficiency of sparse attention into practical acceleration. We
believe FSA provides a foundation for future exploration of hardware-efficient sparse attention.


ACKNOWLEDGMENT


This work is supported by the HKUST startup grant R9895 from CSE; RGC-ECS project 26218024;
RGC-NSFC project CRS ~~H~~ KUST601/24.


10


REFERENCES


Shantanu Acharya, Fei Jia, and Boris Ginsburg. Star attention: Efficient llm inference over long
sequences. _arXiv preprint arXiv:2411.17116_, 2024.


Saurabh Agarwal, Bilge Acun, Basil Hosmer, Mostafa Elhoushi, Yejin Lee, Shivaram Venkataraman, Dimitris Papailiopoulos, and Carole-Jean Wu. Chai: clustered head attention for efficient
llm inference. In _Proceedings_ _of_ _the_ _41st_ _International_ _Conference_ _on_ _Machine_ _Learning_, pp.
291–312, 2024.


Joshua Ainslie, James Lee-Thorp, Michiel De Jong, Yury Zemlyanskiy, Federico Lebr´on, and Sumit
Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. _arXiv preprint arXiv:2305.13245_, 2023.


Anthropic. The claude 3 model family: Opus, sonnet, haiku, 2024. [URL https://www-cdn.](https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf)
[anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_](https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf)
[Card_Claude_3.pdf.](https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf)


Yaofo Chen, Zeng You, Shuhai Zhang, Haokun Li, Yirui Li, Yaowei Wang, and Mingkui
Tan. Core context aware transformers for long context language modeling. _arXiv_ _preprint_
_arXiv:2412.12465_, 2024.


Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. _arXiv_
_preprint arXiv:2307.08691_, 2023.


Tri Dao, Daniel Haziza, Francisco Massa, and Grigory Sizov. Flash-decoding for long-context
inference, 2023. [URL https://pytorch.org/blog/flash-decoding/.](https://pytorch.org/blog/flash-decoding/)


Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha
Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models.
_arXiv e-prints_, pp. arXiv–2407, 2024.


Huiqiang Jiang, Yucheng Li, Chengruidong Zhang, Qianhui Wu, Xufang Luo, Surin Ahn, Zhenhua
Han, Amir H Abdi, Dongsheng Li, Chin-Yew Lin, et al. Minference 1.0: Accelerating pre-filling
for long-context llms via dynamic sparse attention. _Advances in Neural Information Processing_
_Systems_, 37:52481–52515, 2024a.


YOUHE JIANG, Ran Yan, and Binhang Yuan. Hexgen-2: Disaggregated generative inference of
llms in heterogeneous environment. In _The_ _Thirteenth_ _International_ _Conference_ _on_ _Learning_
_Representations_ .


Youhe Jiang, Ran Yan, Xiaozhe Yao, Yang Zhou, Beidi Chen, and Binhang Yuan. Hexgen: generative inference of large language model over heterogeneous environment. In _Proceedings of the_
_41st International Conference on Machine Learning_, pp. 21946–21961, 2024b.


Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph
Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model
serving with pagedattention. In _Proceedings of the 29th symposium on operating systems princi-_
_ples_, pp. 611–626, 2023.


Xunhao Lai, Jianqiao Lu, Yao Luo, Yiyuan Ma, and Xun Zhou. Flexprefill: A context-aware sparse
attention mechanism for efficient long-sequence inference. 2025.


Heejun Lee, Jina Kim, Jeffrey Willette, and Sung Ju Hwang. Sea: Sparse linear attention with
estimated attention mask. _arXiv preprint arXiv:2310.01777_, 2023.


Enzhe Lu, Zhejun Jiang, Jingyuan Liu, Yulun Du, Tao Jiang, Chao Hong, Shaowei Liu, Weiran He,
Enming Yuan, Yuzhi Wang, et al. Moba: Mixture of block attention for long-context llms. _arXiv_
_preprint arXiv:2502.13189_, 2025.


Maxim Milakov and Natalia Gimelshein. Online normalizer calculation for softmax. _arXiv preprint_
_arXiv:1805.02867_, 2018.


11


NVIDIA. Cuda c++ programming guide. [https://docs.nvidia.com/cuda/](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
[cuda-c-programming-guide/, 2024a.](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) Section on Atomic Functions.


NVIDIA. Nvidia h20 solution brief. [https://images.nvidia.com/content/pdf/](https://images.nvidia.com/content/pdf/dgx-apps/NVIDIA-H2O-Solution-Brief-June17.pdf)
[dgx-apps/NVIDIA-H2O-Solution-Brief-June17.pdf, 2024b.](https://images.nvidia.com/content/pdf/dgx-apps/NVIDIA-H2O-Solution-Brief-June17.pdf)


NVIDIA. H200 tensor core gpu. [https://www.nvidia.com/en-us/data-center/](https://www.nvidia.com/en-us/data-center/h200/)
[h200/, 2024c.](https://www.nvidia.com/en-us/data-center/h200/)


NVIDIA. Cuda c++ best practices guide. [https://docs.nvidia.com/cuda/](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
[cuda-c-best-practices-guide/, 2024d.](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)


NVIDIA. Parallel thread execution isa version 9.0 - warp-level matrix instructions.
[https://docs.nvidia.com/cuda/parallel-thread-execution/index.](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions)
[html#warp-level-matrix-instructions, 2025.](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions)


OpenAI. Openai gpt-4o, 2024. URL [https://platform.openai.com/docs/models/](https://platform.openai.com/docs/models/gpt-4o)
[gpt-4o.](https://platform.openai.com/docs/models/gpt-4o)


FLA Organization. Native sparse attention. [https://github.com/fla-org/](https://github.com/fla-org/native-sparse-attention)
[native-sparse-attention, 2024.](https://github.com/fla-org/native-sparse-attention)


Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan
Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism. _arXiv preprint arXiv:1909.08053_, 2019.


Connor Shorten. Ml-arxiv-papers. [https://huggingface.co/datasets/CShorten/](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers)
[ML-ArXiv-Papers, 2024.](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers)


Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, and Song Han. Quest: queryaware sparsity for efficient long-context llm inference. In _Proceedings of the 41st International_
_Conference on Machine Learning_, pp. 47901–47911, 2024.


Yi Tay, Dara Bahri, Liu Yang, Donald Metzler, and Da-Cheng Juan. Sparse sinkhorn attention.
_arXiv preprint arXiv:2002.11296_, 2020.


Qwen Team. Qwen2 technical report. _arXiv preprint arXiv:2407.10671_, 2024.


[Triton. Fused attention tutorial. https://triton-lang.org/main/getting-started/](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
[tutorials/06-fused-attention.html, 2024.](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)


Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. _Advances_ _in_ _neural_ _informa-_
_tion processing systems_, 30, 2017.


Xindi Wang, Mahsa Salmani, Parsa Omidi, Xiangyu Ren, Mehdi Rezagholizadeh, and Armaghan
Eshaghi. Beyond the limits: A survey of techniques to extend the context length in large language
models. _arXiv preprint arXiv:2402.02244_, 2024.


Chaojun Xiao, Pengle Zhang, Xu Han, Guangxuan Xiao, Yankai Lin, Zhengyan Zhang, Zhiyuan
Liu, and Maosong Sun. Inf m: Training-free long-context extrapolation for llms with an efficient context memory. _Advances in Neural Information Processing Systems_, 37:119638–119661,
2024a.


Guangxuan Xiao, Jiaming Tang, Jingwei Zuo, Shang Yang, Haotian Tang, Yao Fu, Song Han, et al.
Duoattention: Efficient long-context llm inference with retrieval and streaming heads. In _The_
_Thirteenth International Conference on Learning Representations_, 2024b.


Chejian Xu, Wei Ping, Peng Xu, Zihan Liu, Boxin Wang, Mohammad Shoeybi, Bo Li, and Bryan
Catanzaro. From 128k to 4m: Efficient training of ultra-long context large language models.
_arXiv preprint arXiv:2504.06214_, 2025a.


Ruyi Xu, Guangxuan Xiao, Haofeng Huang, Junxian Guo, and Song Han. Xattention: Block sparse
attention with antidiagonal scoring. In _Proceedings_ _of_ _the_ _42nd_ _International_ _Conference_ _on_
_Machine Learning (ICML)_, 2025b.


12


Ran Yan, Youhe Jiang, Xiaonan Nie, Fangcheng Fu, Bin Cui, and Binhang Yuan. Hexiscale: Accommodating large language model training over heterogeneous environment. _arXiv_ _preprint_
_arXiv:2409.01143_, 2024.


An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu,
Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. _arXiv_ _preprint_
_arXiv:2505.09388_, 2025.


Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Heng Li, Jiangcheng
Zhu, Jianqun Chen, Jing Chang, et al. Yi: Open foundation models by 01. ai. _arXiv_ _preprint_
_arXiv:2403.04652_, 2024.


Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie,
YX Wei, Lean Wang, Zhiping Xiao, et al. Native sparse attention: Hardware-aligned and natively
trainable sparse attention. _arXiv preprint arXiv:2502.11089_, 2025.


Organization Zai. Longbench: A benchmark for long-context language models., 2023. URL
[https://huggingface.co/datasets/zai-org/LongBench.](https://huggingface.co/datasets/zai-org/LongBench)


Jintao Zhang, Haofeng Huang, Pengle Zhang, Jia Wei, Jun Zhu, and Jianfei Chen. Sageattention2: Efficient attention with thorough outlier smoothing and per-thread int4 quantization. _arXiv_
_preprint arXiv:2411.10958_, 2024a.


Jintao Zhang, Jia Wei, Haofeng Huang, Pengle Zhang, Jun Zhu, and Jianfei Chen. Sageattention: Accurate 8-bit attention for plug-and-play inference acceleration. _arXiv_ _preprint_
_arXiv:2410.02367_, 2024b.


Jintao Zhang, Rundong Su, Chunyu Liu, Jia Wei, Ziteng Wang, Pengle Zhang, Haoxu Wang,
Huiqiang Jiang, Haofeng Huang, Chendong Xiang, Haocheng Xi, Shuo Yang, Xingyang Li,
Yuezhou Hu, Tianyu Fu, Tianchen Zhao, Yicheng Zhang, Youhe Jiang, Chang Chen, Kai Jiang,
Huayu Chen, Min Zhao, Xiaoming Xu, Jun Zhu, and Jianfei Chen. A survey of efficient attention
methods: Hardware-efficient, sparse, compact, and linear attention. 2025a.


Jintao Zhang, Jia Wei, Pengle Zhang, Xiaoming Xu, Haofeng Huang, Haoxu Wang, Kai Jiang,
Jun Zhu, and Jianfei Chen. Sageattention3: Microscaling fp4 attention for inference and an
exploration of 8-bit training. _arXiv preprint arXiv:2505.11594_, 2025b.


Li Zhang, Youhe Jiang, Guoliang He, Xin Chen, Han Lv, Qian Yao, Fangcheng Fu, and Kai
Chen. Efficient mixed-precision large language model inference with turbomind. _arXiv preprint_
_arXiv:2508.15601_, 2025c.


Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song,
Yuandong Tian, Christopher R´e, Clark Barrett, et al. H2o: Heavy-hitter oracle for efficient generative inference of large language models. _Advances in Neural Information Processing Systems_,
36:34661–34710, 2023.


Guangxiang Zhao, Junyang Lin, Zhiyuan Zhang, Xuancheng Ren, Qi Su, and Xu Sun. Explicit sparse transformer: Concentrated attention through explicit selection. In _arXiv_ _preprint_
_arXiv:1912.11637_, 2019.


Qianchao Zhu, Jiangfei Duan, Chang Chen, Siran Liu, Xiuhong Li, Guanyu Feng, Xin Lv, Huanqi
Cao, Xiao Chuanfu, Xingcheng Zhang, et al. Sampleattention: Near-lossless acceleration of long
context llm inference with adaptive structured sparse attention. _arXiv preprint arXiv:2406.15486_,
2024.


13


A THE USE OF LARGE LANGUAGE MODELS


In this paper, we leverage LLMs to enhance academic writing quality by ensuring grammatical
correctness and improving sentence structure.


B NOTATIONS


The notations used in this paper are summarized in Table 1.


Table 1: Notations and Explanations.

|Notation|Explanation|
|---|---|
|_N_|Sequence length.|
|_dK_|Head dimension for query and key tensor.|
|_dV_|Head dimension for value tensor.|
|_d_|Uniform head dimension, i.e.,_ d_ =_ dK_ =_ dV_ .|
|_h_|<br>Number of Q heads.|
|_hK_|Number of KV heads.|
|_g_|GQA group size, defned as_ g_ =<br>_h_<br>_hK_ .|
|_T_|<br>Number of selected KV blocks of each query token.<br>(Hyperparameter of the NSA sparse attention module.)|
|_BK_|Block size of each KV block; a NSA hyperparameter.|
|_b_|Number of KV blocks;_ b_ =<br>_N_<br>_BK_ .|
|_BQ_|<br>Query batch size in FSA; a FSA hyperparameter.|
|_Ii_|The set of query indices attending to the_ i_-th KV block.<br>(_Ii_ contain non-contiguous query indices, usually_ |Ii| ≤N_.)|
|_Oi_|<br>The output tensor mapping for the_ i_-th KV block; e.g.,_ Oi_[_j_]<br>gives the storage position of token_ j_ in the output buffer.|
|_N_valid|The number of valid query tokens in_ Ii_.|
|**T**|Sparse selected KV block indices in NSA.|
|**Q**,**KV**|Full query, key, and value tensor for attention computation.|
|**Q**batch|Non-contiguous Query batches introduced in FSA.<br>(One thread block processes multiple** Q**batch.)|
|**K**_i_,**V**_i_|The_ i_-th KV block with_ BK_ contiguous KV tokens.|
|**O**buf|<br>Intermediate buffer which holds query attention results<br>without scaling with online softmax in FSA.|


C FSA IMPLEMENTATION DETAILS


FSA is implemented using 10K lines of Python and Triton code. To optimize system performance:
(i) We apply fine-grained control over FSA selected attention kernel and reduction kernel to optimize warp-level parallelism. FSA usually assigns 4 warps per thread block for FSA selected
attention kernel, which contains matrix multiplication operations, to enable sufficient computational
resources of a given thread block. FSA usually assigns 1 to 2 warps per thread block for reduction
kernel, which mainly consists of elementwise operations. Warp assignment for reduction kernel
efficiently utilizes warp-level parallelism, reducing reduction kernel execution latency. (ii) We speculatively compute online softmax statistics once per KV heads. Due to invariant nature of online
softmax (Milakov & Gimelshein, 2018), correctness of FSA is maintained, while significant cost
for computing online softmax statistics is amortized.


D FSA CORRECTNESS


**FSA** **correctness.** To evaluate correctness of FSA kernels, we fine-tune Llama3-8B model using
ML-ArXiv-Papers dataset (Shorten, 2024). We replace attention module of Llama3-8B model with


14


2.0

1.5

1.0

0.5

0.0


|FSA|NSA FA|
|---|---|
|||
|||
|||
|||


1 200 400 600 800 1000 1200 1400 1600 1800 2000
Step


Figure 11: Loss comparison of FSA/NSA/full attention in end-to-end Llama3-8B training.


either FSA or NSA, while initializing all other components with pretrained model checkpoints provided by Meta. For fair comparison with full attention, we reinitialize the parameters of the attention
module. Loss comparison among FSA, NSA, and full attention is presented in Figure 11. Results
demonstrate that all three methods achieve stable and similar convergence, and FSA exhibits a similar loss curve to NSA, validating the correctness of the FSA kernel.


**Further** **analysis.** To strengthen our accuracy evaluation, we conduct additional experiments by
fine-tuning smaller models across diverse tasks. Specifically, we fine-tune Llama-3.2-1B, Llama3.2-3B, and Llama-3.1-8B Dubey et al. (2024) on three representative LongBench Zai (2023) tasks:
multi-document QA (MQA) on the HotpotQA dataset, single-document QA (SQA) on the Qasper
dataset, and synthetic-data QA (Synthetic) on the PassR-EN dataset. For Llama-3.2-1B and Llama3.2-3B, we report the average loss after convergence over 2K training steps, comparing FSA, NSA,
and Full Attention; these results appear in Table 2. To more comprehensively assess accuracy preservation, we further evaluate perplexity and QA F1 across all three models and tasks. The results,
summarized in Tables 3 and 4, consistently show that FSA matches the accuracy of NSA and Full
Attention.


Table 2: Converged Loss Across Datasets and Attention Modes.

|MQA (HQA) SQ<br>Model Size<br>FA FSA NSA FA|A (Qasper) Synthetic (PassR-EN)<br>FSA NSA FA FSA NSA|
|---|---|
|1B<br>0.200<br>0.182<br>0.187<br>0.216<br>3B<br>0.173<br>0.153<br>0.166<br>0.087|0.191<br>0.184<br>0.231<br>0.224<br>0.231<br>0.082<br>0.078<br>0.123<br>0.119<br>0.118|


Table 3: PPL Across Datasets, Models, and Attention Modes.

|MQA (HQA) SQ<br>Model Size<br>FA FSA NSA FA|A (Qasper) Synthetic (PassR-EN)<br>FSA NSA FA FSA NSA|
|---|---|
|1B<br>5.40<br>6.79<br>6.82<br>8.77<br>3B<br>2.42<br>1.50<br>1.48<br>1.20<br>8B<br>1.57<br>1.17<br>1.16<br>1.28|9.48<br>9.45<br>3.48<br>2.52<br>2.49<br>2.64<br>2.62<br>1.87<br>1.94<br>1.96<br>1.71<br>1.70<br>1.21<br>1.26<br>1.27|


Table 4: QA F1 Across Datasets, Models, and Attention Modes.

|MQA (HQA) SQ<br>Model Size<br>FA FSA NSA FA|A (Qasper) Synthetic (PassR-EN)<br>FSA NSA FA FSA NSA|
|---|---|
|1B<br>0.05<br>0.10<br>0.11<br>0.08<br>3B<br>0.28<br>0.35<br>0.33<br>0.15<br>8B<br>0.32<br>0.38<br>0.37<br>0.23|0.07<br>0.06<br>0.22<br>0.32<br>0.31<br>0.11<br>0.12<br>0.39<br>0.47<br>0.48<br>0.20<br>0.19<br>0.83<br>0.86<br>0.86|


E FSA AND NSA THEORETICAL MEMORY ACCESS AND FLOPS ANALYSIS


To demonstrate how FSA outperforms NSA selected attention, we analyze as follows. For simplicity, we assume query/key/value have the same head dimension, i.e. _d_ = _dK_ = _dV_ .


15


**FSA analytic advantages.** _Theoretically,_ FSA _introduces lower memory access volume and num-_
_ber_ _of_ _floating-point_ _operations_ _(FLOPs)_ _for_ _small_ _GQA_ _group_ _sizes_ . We analyze FSA/NSA as
follows:


FSA _memory access volume and FLOPs._ We analyze the three key components in FSA as follows:


- **FSA selected attention kernel** launches _hb_ thread blocks, where _h_ is the number of query attention heads, and _b_ is the total number of KV blocks. For a sequence of _N_ tokens, the number of
KV blocks _b_ = _BNK_ [, where] _[ B][K]_ [is the KV block size.] [In one thread block, FSA selected attention]
kernel runs a two-level loop. In the outer loop, it loads 2 _BKd_ KV tokens; in the inner loop, it iteratively loads _BQd_ query tokens, performs attention computation with a FLOPs of 4 _BQBKd_, and
stores _BQd_ query attention results. We estimate the number of our inner loop as follows. Assume
each query token attends to each KV block with equal probability. Therefore, each query token
attends to a given KV block with a probability of _[T]_ _b_ [,] [resulting] [in] [an] [average] [number] [of] [tokens]

attending to a given KV block of _[NT]_ _b_ [, and an average number of query batches for one KV block]

of _bBNTQ_ [.] [Assuming] [each] [data] [occupies] [2] [bytes,] [we] [can] [calculate] [memory] [accessed] [in] [bytes] [by]
FSA selected attention kernel as 4 _dhN_ (1 + _T_ ), and FLOPs as 4 _dhNBKT_ .


- **FSA** **online** **softmax** **kernel** operates similarly to the FSA selected attention kernel, with three
key differences: It is called per KV head, omits V tensor loading and computation, and intermediate attention scores storage, storing only a single scalar value per (query token, KV block) pair.
Following a similar estimation logic as FSA selected attention kernel, the online softmax kernel
introduces 2 _dhKN_ (1 + _T_ ) memory access volume in bytes, and 2 _dhKNBKT_ FLOPs.


- **FSA reduction kernel** introduces negligible FLOPs, but for each query token, it involves loading
attention results of _T_ KV blocks and storing the final attention results. Therefore, FSA reduction
kernel introduces 2 _dhN_ (1 + _T_ ) memory access in bytes.
In total, FSA incurs _dN_ (6 _h_ + 2 _hK_ )(1 + _T_ ) memory access in bytes, and _dNBKT_ (4 _h_ + 2 _hK_ )
FLOPs.
_NSA_ _Memory_ _access_ _volume_ _and_ _FLOPs._ NSA selected attention kernel launches _hKN_ thread
blocks, where _hK_ is the number of KV heads. In each thread block, NSA kernel runs a two-level
loop. In the outer loop, NSA kernel loads one query token and _g_ = _hhK_ [Q] [heads] [that] [share] [the]
same KV head. Due to the hardware requirements on matrix multiplication shapes, when GQA
_<_ 8, NSA kernels must load 8 query heads (8 _d_ elements), perform computation, and mask out
the undesired computation results. In the inner loop, NSA kernel iteratively (up to _T_ times) loads
one KV block (2 _BKd_ elements) and performs attention computation with a FLOPs of 32 _BKd_ . To
maintain the causal property, i.e., avoiding query tokens to attend to future KV tokens, the actual
number of KV blocks that need to be loaded and participate in computations within a thread block
is on average _[T]_ 2 [.] [Finally,] [NSA kernel stores the attention results in the output tensor,] [incurring] _[ gd]_

memory access. Therefore, we can estimate the memory access volume (2 bytes per data) for NSA
kernel as 2 _dhKN_ ( _BKT_ + _g_ + 8). The FLOPs for NSA kernel are 32 _dhKNBKT_ .


FSA _selected attention kernels exhibit lower memory access volume and FLOPs._ With ( _BK, T_ ) =
(64 _,_ 16) and sequence length of 64K, which is the same configuration as presented in the NSA paper,
we observe that compared to the NSA selected attention kernel, our method incurs lower memory access volume and FLOPs for GQA _≤_ 8, detailed comparisons are presented in Figure 2. In particular,
for GQA=4, a common configuration in LLMs, our method theoretically reduces memory access
volume to 21.3% and FLOPs to 56.2% of those in NSA. Benefits from the more efficient hardwarealigned kernel design, our method substantially outperforms NSA across various GQA group sizes.
Additionally, our method demonstrates superior performance as the NSA hyperparameter _BK_ increases. This advantage stems from NSA’s inherent inefficiency with larger KV blocks. Although
NSA can easily skip loading KV blocks that fully violate causal property, to maintain causality constraints for KV blocks that partially violate causal property, NSA must mask out many KV tokens
within the KV block, leading to wasteful memory accesses where loaded data is only partially valid
for computation. As the KV block size _BK_ grows larger, this inefficiency becomes increasingly
pronounced, as a greater proportion of the loaded KV block remains unused due to causal masking.
In contrast, our method processes all query tokens that attend to a given KV block within a single
thread block, naturally satisfying causal constraints without requiring extensive masking. This approach achieves superior memory efficiency by ensuring that all loaded KV data contributes to the
computation, resulting in significantly lower memory access overhead.


16


**FSA** **trade-offs.** FSA _trades_ _lowered_ _memory_ _access_ _volume_ _and_ _FLOPs_ _with_ _non-contiguous_
_loading_ _and_ _more_ _buffer overhead._ Theoretical advantages of FSA come at the price of involving
non-contiguous memory access and more buffers that occupy HBM memory. We analyze how
these factors compromise FSA performance and how FSA optimizes memory access and buffer
management as follows:


- **Optimize** **memory** **access.** The non-contiguous loading on query batches, which is inefficient
on modern GPUs, compromises FSA selected attention kernel performance. Modern GPUs usually operate more efficiently under coalesced and contiguous memory access, which can improve
the L2-cache hit rate and thereby kernel efficiency (NVIDIA, 2024d). Therefore, the theoretical advantages of our method cannot be fully reflected in actual hardware, due to inevitably
degraded performance of non-contiguous memory access. Nonetheless, to our best effort, FSA
optimizes memory access with fine-grained early return mechanisms that filter out unnecessary
query batches loading. For example, for _i_ -th KV block, FSA compactly stores query indices in
set _Ii_, which is computed via a full index table. For each query token, the full index table records
whether it should attend to _i_ -th KV block, and _Ii_ filters the tokens that do not attend to _i_ -th KV
block. Therefore, when all query tokens in _Ii_ are exhausted, FSA returns early.


- **Optimize buffer management.** The newly introduced buffers, **O** buf appeared in Figure 1 (right),
bring memory overhead. FSA minimizes buffer overhead from two aspects: (i) FSA Token selection kernel processes a subset of query heads at each time, reusing the buffers for subsequent
query heads computations. (ii) FSA introduces an output index mapping tensor to store results
compactly. For each query head, FSA only reserves buffers for maximum query tokens that attend
to a given KV block. On average, this value is _BKT_, combining that _b_ = _BNK_ [,] [FSA] [introduces]
an output buffer with _dNT_ elements. Assume each data in the output buffer occupies 2 bytes,
for a sequence with 64K tokens, _T_ at 16, and _d_ at 128, **O** buf occupies 1 GB HBM memory (This
also applies for the buffer for intermediate gradients with respect to **Q** ). Compared to the high
HBM memory capacity in modern GPUs,e.g., 96 GB HBM memory on H20 (NVIDIA, 2024b)
and 141 GB memory on H200 (NVIDIA, 2024c), the additional buffer overhead in FSA remains
manageable.

**Attention** **Sink** **Optimizations.** The attention sink phenomenon in NSA sparse token selection
presents a challenge for FSA’s buffer management strategy. The initial KV block receives attention
from all query tokens, while subsequent KV blocks exhibit more selective attention patterns. This
asymmetry creates a buffer allocation dilemma: In practice, FSA allocates uniform buffer sizes
based on the maximum number of valid tokens across all KV blocks. However, the attention sink
property forces this maximum as full sequence length, thereby negating the memory efficiency gains
that FSA’s sparse buffer management is designed to achieve. To address this inefficiency, we implement a dual-buffer allocation strategy. We maintain separate buffer allocations for the attention
sink (first KV block) and the remaining KV blocks. The attention sink buffer accommodates the full
query sequence, while buffers for subsequent KV blocks are sized according to their maximum valid
query tokens, which are usually much smaller than full sequence length. This approach preserves
the memory optimization benefits for the majority of KV blocks while handling the attention sink’s
dense connectivity requirements.


**FSA** **online profiling module.** _In real-world deployment,_ FSA _dynamically selects kernel config-_
_uration_ _via_ _online_ _profiling,_ _and_ _potentially_ _falls_ _back_ _to_ _original_ _NSA_ _implementation._ To ensure
optimal performance across diverse NSA configurations, FSA incorporates a one-time online profiling mechanism. Upon its first execution with a new set of hyperparameters (e.g., sequence length,
GQA group size), FSA benchmarks its kernel performance across several candidate query batch
sizes (e.g., 1, 64, 128). When GQA group size is sufficiently large, a query batch size of 1 is additionally searched and serves as a potential fallback to original NSA strategy of batching query heads.
Once profiling is complete, the fastest configuration is cached. All subsequent calls with the same
hyperparameters directly use this optimal configuration, bypassing profiling step until hyperparameters change.


**Actual** **memory** **footprint** **of** **FSA** **buffers.** We conduct additional micro-benchmarks to measure
the memory footprint of FSA buffers. Concretely, we set the head dimension at 128 and use the NSA
hyperparameters ( _BK_, _T_ ) = (64,16) or (128,8), and report the profiled buffer overheads for sequence
lengths ranging from 32K to 256K in Table 5. Under extreme cases, i.e., when the sequence length
is 128K or 256K, FSA introduces 5.01GB or 12.36 GB buffer memory overhead, which is still much


17


smaller than the memory capacity of modern GPUs (e.g., H200 has 141GB memory). These results
confirm that the FSA buffer memory overhead remains acceptable.


Table 5: Profiled Buffer Overhead.


( _BK_, _T_ ) Seqlen (K) Profled Buffer Overhead (GB)
(64, 16) 32 0.52
(64, 16) 64 1.88
(64, 16) 128 5.01
(64, 16) 256 12.36
(128, 8) 32 0.26
(128, 8) 64 0.91
(128, 8) 128 2.28
(128, 8) 256 6.15


F EVALUATIONS FOR ULTRA LONG SEQUENCE LENGTHS.


We extend our evaluations to 128K and 256K sequence lengths. Fixing the head dimension at 128
and the number of query heads at 64, while varying the number of key and value heads, we evaluate
configurations where a GQA group contains 1 to 8 query heads. Using the NSA hyperparameters with ( _BK_, _T_ ) = (64, 16) or (128, 8), we benchmark the performance of FSA, NSA, and Full
Attention (FA) on both H20 and H200 GPUs.


F.1 INFERENCE PREFILL AND TRAINING EVALUATIONS


**Results discussion:** The experimental results in Table 6 and 7 show that FSA also outperforms NSA
for ultra-long sequence lengths. For inference prefill execution latency, FSA achieves up to 1.47 _×_
speedup and an average of 1.20 _×_ lower kernel latency on H20 GPUs, and up to 1.86 _×_ speedup with
an average of 1.23 _×_ lower kernel latency on H200 GPUs, compared to NSA. For training execution
latency — measured over one forward and one backward pass — FSA achieves up to 1.91 _×_ speedup
and an average of 1.37 _×_ lower kernel latency on H20 GPUs, and up to 2.55 _×_ speedup with an
average of 1.49 _×_ lower kernel latency on H200 GPUs, relative to NSA.


Table 6: H20 GPU, Inference Prefill and Training Latency for Different ( _BK_, _T_ ).


Seq Len FSA NSA FA FSA NSA FA
( _BK_, _T_ ) GQA (K) Fwd (s) Fwd (s) Fwd (s) F + B (s) F + B (s) F + B (s)


(64,16)


(128,8)


1 128 1.42 2.08 2.64 2.36 4.51 12.08
1 256 6.40 7.18 10.5 8.70 13.29 48.23
2 128 0.87 1.17 2.62 1.79 2.71 12.04
2 256 3.74 4.07 10.53 6.03 8.43 48.27
4 128 0.52 0.61 2.65 1.43 1.75 12.07
4 256 2.41 2.44 10.52 4.66 5.98 48.24
8 128 0.45 0.45 2.61 1.38 1.39 12.05
8 256 1.63 1.64 10.51 3.99 4.75 48.26

1 128 1.24 1.68 2.64 2.15 3.50 12.08
1 256 5.34 7.50 10.5 7.56 13.65 48.23
2 128 0.90 1.29 2.62 1.66 2.22 12.04
2 256 3.18 4.24 10.53 5.40 8.33 48.27
4 128 0.45 0.49 2.65 1.35 1.52 12.07
4 256 2.10 2.53 10.52 4.29 5.55 48.24
8 128 0.42 0.43 2.61 1.31 1.41 12.05
8 256 1.56 1.68 10.51 3.74 4.17 48.28


18


Table 7: H200 GPU, Inference Prefill and Training Latency for Different ( _BK_, _T_ ).


Seq Len FSA NSA FA FSA NSA FA
( _BK_, _T_ ) GQA (K) Fwd (s) Fwd (s) Fwd (s) F + B (s) F + B (s) F + B (s)


(64,16)


(128,8)


1 128 0.78 1.01 1.01 1.12 2.17 4.70
1 256 3.92 3.97 3.96 4.81 6.99 18.75
2 128 0.46 0.57 0.98 0.80 1.24 4.68
2 256 2.14 2.17 3.98 3.04 4.12 18.77
4 128 0.27 0.33 0.99 0.60 0.82 4.71
4 256 1.20 1.29 3.99 2.15 2.70 18.76
8 128 0.18 0.18 0.97 0.50 0.52 4.67
8 256 0.73 0.73 3.97 1.72 2.01 18.78

1 128 0.65 1.20 1.00 0.97 2.47 4.70
1 256 3.16 4.10 3.96 3.99 6.79 18.75
2 128 0.38 0.66 0.98 0.70 1.40 4.68
2 256 1.76 2.21 3.99 2.58 3.90 18.77
4 128 0.21 0.28 0.99 0.53 0.73 4.71
4 256 1.06 1.24 3.97 1.87 2.44 18.76
8 128 0.17 0.20 0.97 0.48 0.58 4.67
8 256 0.71 0.75 3.95 1.52 1.70 18.78


F.2 INFERENCE END-TO-END EVALUATIONS


By further fixing the number of generated tokens at 512, we evaluate the end-to-end inference execution latency of FSA, NSA, and Full Attention on both H20 and H200 GPUs.


**Results and discussion:** The experimental results in Table 8 and 9 demonstrate that FSA’s performance scales well for extremely long sequences. For inference execution latency: (i) compared to
NSA, FSA achieves up to 1.40 _×_ speedup and on average 1.16 _×_ lower kernel latency on H20 GPUs,
and up to 1.59 _×_ speedup and on average 1.15 _×_ lower kernel latency on H200 GPUs. (ii) Compared
to Full Attention, FSA achieves up to 7.20 _×_ speedup and on average 4.61 _×_ lower kernel latency on
H20 GPUs, and up to 4.71 _×_ speedup and on average 2.96 _×_ lower kernel latency on H200 GPUs.


Table 8: H20 Inference Latency (s) for ( _BK_, _T_ ) at (64,16) and (128,8).

|Method|(B, T)<br>K|GQA = 1|GQA = 2|GQA = 4|GQA = 8|
|---|---|---|---|---|---|
|Method|(_BK_,_ T_)|128K<br>256K|128K<br>256K|128K<br>256K|128K<br>256K|
|FSA|(64,16)<br>(128,8)|1.67<br>6.71<br>1.49<br>5.65|1.12<br>4.05<br>1.15<br>3.49|0.77<br>2.72<br>0.70<br>2.41|0.70<br>1.94<br>0.67<br>1.87|
|NSA|(64,16)<br>(128,8)|2.33<br>7.49<br>1.93<br>7.81|1.42<br>4.38<br>1.54<br>4.55|0.86<br>2.75<br>0.74<br>2.84|0.70<br>1.95<br>0.68<br>1.99|
|FA|–|4.34<br>13.45|4.32<br>13.48|4.35<br>13.47|4.31<br>13.46|


Table 9: H200 Inference Latency (s) for ( _BK_, _T_ ) at (64,16) and (128,8).

|Method|(B, T)<br>K|GQA = 1|GQA = 2|GQA = 4|GQA = 8|
|---|---|---|---|---|---|
|Method|(_BK_,_ T_)|128K<br>256K|128K<br>256K|128K<br>256K|128K<br>256K|
|FSA|(64,16)<br>(128,8)|1.06<br>4.24<br>0.93<br>3.48|0.74<br>2.46<br>0.66<br>2.08|0.55<br>1.52<br>0.49<br>1.38|0.46<br>1.05<br>0.45<br>1.03|
|NSA|(64,16)<br>(128,8)|1.29<br>4.29<br>1.48<br>4.42|0.85<br>2.49<br>0.94<br>2.53|0.61<br>1.61<br>0.56<br>1.56|0.46<br>1.05<br>0.48<br>1.07|
|FA|–|1.88<br>4.86|1.85<br>4.88|1.86<br>4.89|1.84<br>4.87|


G COMPARISON WITH FLASHDECODING


To compare with the state-of-the-art FlashDecoding kernel Dao et al. (2023), we conducted additional experiments measuring decoding execution latency for FlashDecoding and FSA. Given a


19


prefill sequence length (ranging from 32K to 256K), we present the average decoding latency across
1K generated tokens. We fixed the number of attention heads at 64 and the head dimension at 128.
For FSA, the sparse-attention hyperparameters were set to a block size _BK_ of 64 and TopK Value _T_
of 16.


**Result discussions:** Experimental results in Table 10 demonstrate that FSA achieves superior performance to FlashDecoding. Compared to FlashDecoding, FSA demonstrates an average speedup of
5.46x on H20 GPU and 2.16x on H200 GPU. During the decoding phase, FlashDecoding partitions
the key and value tokens and distributes the resulting attention computation tasks across multiple thread blocks, thereby increasing kernel-level parallelism and improving decoding throughput.
However, due to the sparsity in FSA, the FSA decoding throughput is still superior to FlashDecoding.


Table 10: Decoding Latency on H20 and H200 GPUs.

|Seq Len (K)|H20 Latency (ms)<br>FlashDecoding NSA FSA|H200 Latency (ms)<br>FlashDecoding NSA FSA|
|---|---|---|
|32<br>64<br>128<br>256|0.88<br>0.46<br>0.45<br>1.71<br>0.46<br>0.48<br>3.32<br>0.50<br>0.48<br>5.76<br>0.62<br>0.61|0.51<br>0.48<br>0.47<br>0.88<br>0.53<br>0.54<br>1.70<br>0.55<br>0.54<br>1.75<br>0.62<br>0.63|


H COMPLIATION OVERHEAD


To determine the optimal Triton kernel hyperparameters, both FSA and NSA incur a compilation
overhead. For a given NSA hyperparameter combination, this overhead occurs only once. Setting
( _BK_, _T_ ) at (64, 16) or (128, 8), we evaluate the compilation overhead of FSA and NSA for sequence
length across diverse sequence lengths. The experimental results are summarized in Table 11.


Table 11: Compilation Overhead on H20 and H200 GPUs.


Seqlen (K) Framework H20 Overhead (s) H200 Overhead (s)
32 FSA 2.16 1.82
32 NSA 2.12 1.78
64 FSA 2.37 2.01
64 NSA 2.33 1.95
128 FSA 2.59 2.24
128 NSA 2.55 2.19
256 FSA 2.80 2.36
256 NSA 2.76 2.30


I EVALUATIONS ON DISTRIBUTED PERFORMANCE.


I.1 DISTRIBUTED INFERENCE EVALUATION OF THE ATTENTION MODULE


Table 12: Distributed inference latency of the attention module on H20 GPU.


( _BK_, _T_ ) Seq Len (K) Framework TP=1 (ms) TP=2 (ms) TP=4 (ms) TP=8 (ms)
(64, 16) 32 FSA 82.50 45.00 25.94 16.25
(64, 16) 32 NSA 99.53 53.44 28.75 16.56
(64, 16) 64 FSA 195.84 110.63 61.25 38.63
(64, 16) 64 NSA 221.49 122.81 65.31 39.94
(128, 8) 32 FSA 80.31 43.44 24.69 15.63
(128, 8) 32 NSA 105.10 54.38 28.68 16.75
(128, 8) 64 FSA 187.50 102.68 56.25 33.75
(128, 8) 64 NSA 243.88 130.31 70.69 40.00


20


Table 13: Distributed inference latency of the attention module on H200 GPU.


( _BK_, _T_ ) Seq Len (K) Framework TP=1 (ms) TP=2 (ms) TP=4 (ms) TP=8 (ms)
(64, 16) 32 FSA 43.44 25.00 15.63 11.88
(64, 16) 32 NSA 50.31 27.19 17.63 12.69
(64, 16) 64 FSA 110.00 63.13 39.38 26.13
(64, 16) 64 NSA 121.56 66.81 40.75 27.00
(128, 8) 32 FSA 40.63 23.13 14.38 10.00
(128, 8) 32 NSA 59.06 31.25 17.50 11.63
(128, 8) 64 FSA 96.25 53.75 32.50 22.81
(128, 8) 64 NSA 124.38 65.69 37.50 25.56


We conduct additional experiments to evaluate the distributed inference performance of the attention
module using FSA and NSA on H20 and H200 GPUs. We fix the number of query heads at 32, and
the number of key and value heads at 8. This setting indicates that one GQA group contains 4 query
heads. The results for both methods — measured across different NSA hyperparameters, sequence
lengths, and tensor-parallel degrees — are summarized in the Table 12 and 13. Compared to NSA,
FSA achieves an average speedup of 1.16x on H20 GPUs and 1.17x on H200 GPUs.


I.2 END-TO-END DISTRIBUTED INFERENCE EVALUATION


Following the same configuration as Figure 6, we evaluate the distributed inference performance of
the Llama-3-8B model on H20 and H200 GPUs. The results for NSA and FSA, measured under
varying NSA hyperparameters, sequence lengths, and tensor-parallel degrees, are presented in the
Table 14 and 15. Compared to NSA, FSA achieves an average speedup of 1.13x on H20 GPUs and
1.11x on H200 GPUs.


Table 14: End-to-end distributed inference latency on H20 GPU.


( _BK_, _T_ ) Seqlen (K) Framework TP=1 (s) TP=2 (s) TP=4 (s) TP=8 (s)
(64, 16) 32 FSA 5.28 2.88 1.66 1.04
(64, 16) 32 NSA 6.00 3.22 1.84 1.06
(64, 16) 64 FSA 11.14 7.08 3.92 2.60
(64, 16) 64 NSA 12.04 7.86 4.18 2.66
(128, 8) 32 FSA 5.14 2.78 1.58 1.00
(128, 8) 32 NSA 6.40 3.32 1.80 1.10
(128, 8) 64 FSA 12.00 6.38 3.60 2.16
(128, 8) 64 NSA 13.72 7.10 4.00 2.12


Table 15: End-to-end distributed inference latency on H200 GPU.


( _BK_, _T_ ) Seqlen (K) Framework TP=1 (s) TP=2 (s) TP=4 (s) TP=8 (s)
(64, 16) 32 FSA 1.95 1.12 0.70 0.46
(64, 16) 32 NSA 1.97 1.22 0.70 0.49
(64, 16) 64 FSA 4.51 2.83 1.48 1.09
(64, 16) 64 NSA 4.61 2.81 1.51 1.16
(128, 8) 32 FSA 1.82 1.04 0.64 0.45
(128, 8) 32 NSA 2.37 1.33 0.78 0.53
(128, 8) 64 FSA 3.61 2.13 1.34 0.98
(128, 8) 64 NSA 4.62 2.63 1.61 1.15


21