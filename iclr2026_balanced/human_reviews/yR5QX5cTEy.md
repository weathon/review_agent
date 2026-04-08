## Human Reviewer 1

### Summary
Paper proposes a straggler-aware allreduce algorithm for synchronizing model weights among GPUs in distributed training. This is an important problem for large scale training as communication can be a significant bottleneck at scale and one slow GPU can slow down the whole training step. Overall my main concern is that the evaluation of the algorithm is weak, it’s not in realistic large scale setups and the benefits of the algorithm would not be significant in realistic environments. Finally, this is a distributes systems paper and ICLR may not be the best fit.

### Strengths
- Paper is about an important problem in enabling large-scale distributed training
- In the limited scenarios the algorithm is evaluated, it lowers the upper bound of  allreduce computation

### Weaknesses
- It is a distributed systems focused paper so an ML conference might not be a good fit in terms of its topic and audience. MLSys, ASPLOS or HPC conferences would be a better fit for this topic.
- The algorithm, upper bound calculations assume there is only one straggler. But in real scenarios the delays exhibit a distribution, rather than a binary choice of straggler vs non-straggler categorization (as also illustrated in Figure 2 in the paper). Furthermore, artificial straggler delays are used in evaluations in the end-to-end experiments, the GPUs are profiled ahead of time and a slow GPU is selected manually to represent as a straggler rank. This is not a realistic evaluation scenario representing delays at large scale (1000s of nodes).
- The evaluation setup assumes a uniform GPU-GPU bandwidth, it does not take into account topology of the network.
- The benefits of the algorithm increases when the straggler delay is significant ( it needs to be at least 6-8ms to give speedup). And the possible max delay in Perlmutter supercomputer is 8ms according to Figure 8. So StragglAR would not provide any speedup in a more realistic training setup.

### Questions
- How’s the algorithm bounds change when there are more than multiple stragglers?
- How would you change the algorithm to make it aware of the topology of the network?

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
2

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper presents an ALLReduce algorithm when stragglers exist.
It's found that multiple stragglers are found in real scenarios, which slows down the entire training.
StragglAR is an algorithm to make the non-straggler GPUs to make some forward progress, such that the time can be saved even though there exists a straggler.
Experiments are conducted with 1B and 3B model with up to 8 GPUs to show practicality.

### Strengths
**[S1]** Stragglers are causing real problems in datacenters, and this work could save a lot of monetary cost, which the paper directly presents.

**[S2]** This paper presents a solid algorithm with adequate proofs. I don't see any problems in the algorithm itself.

**[S3]** Quantitative bandwidth analysis, telling the readers how much benefit can be gained in arbitrary environments.

### Weaknesses
**[W1]** Comparison with potential existing algorithms would be beneficial. For example, the authors mention that there exists a tree-based algorithm (AdapCC). I did not check AdapCC, but a layman knowing a tree-based AR algorithm could design a AR/RS a tree such that the straggler participates as late as possible. I believe it can be a simpler alternative. Would it be possible to devise a performance model of this to compare?

**[W2]** The experiments are weak in scale and settings. 
- The experiments are done up to 3B model, only over 8 GPUs. This is way too small compared to what people are using for training, which usually spans multiple nodes. This is both good and bad. Good because the communication time is usually longer and straggling is more likely to exist in larger settings, but at the same time, but bad because the straggler effect could be dwarfed by other communication overheads. This is related to the next subitem.

- The authors report a few tens of milliseconds of straggler delays. However, it's difficult to find what portion of it occupies within the entire AR, or the entire (single) training step. Such information will be important to motivate this work.

- The cause and characterization of the straggler could be further investigated. I guess it might be out of scope to find the cause of straggers, but as a reader, I am very curious about at least how the stragger delay would change when using various larger models, or more servers. If the delay is coming of the network or network interface, it could remain relatively independent of the model size, where the slowdown of the GPU itself could cause the straggler delay to scale larger. This is related to the author's method of simulating the delays.


**[W3]** Minor suggestion. I was confused multiple times by the use of the word 'buffer' to indicate the volume of the communication. 
It sometimes seemed like the buffers are the overhead caused by the stragglAR algorithm. The buffer size indeed decides how much data is handled in a single CC transaction, but I don't think it's a common term for researchers in the field.

### Questions
Please see weaknesses.

### Soundness
4

### Presentation
3

### Contribution
4

### Rating
8

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper proposes StragglAR, a new AllReduce (AR) algorithm that leverages straggler delay to preemptively schedule ReduceScatter (RS) among healthy ranks. For large data packets, it shows significant speedup (~25%) over Ring-AR without any precision loss. As a fundamental, lossless communication primitive, it has broad applicability for technologies like data and tensor parallelism.

### Strengths
1. Well-motivated: This paper works on a practical problem prevalent in distributed training with stragglers. It has strong motivation with empirical and literature support for straggler prevalence. It has an innovative low-level AR redesign.
2. Solid Work: Solid algorithm design with clear exposition. The experiments on a small-scale environment provide preliminary verification of the proposed method's effectiveness under varying straggler delays. The end-to-end speedup is remarkable.

### Weaknesses
1. Unfair Complexity Analysis: The complexity analysis is not general, as it only considers the ideal case where RS execution is hidden by straggler latency. This makes the comparison with other AR algorithms unfair.
2. Missing Baseline: The paper omits a comparison with a critical and more recent baseline, MSCCL++ (https://arxiv.org/pdf/2504.09014), making it difficult to assess its incremental contribution.
3. Figure 7 shows that the straggler delay CDF varies by environment, yet Figures 5(b) and 5(e) use the same average delay. Why not use environment-specific average delays, especially given the hardware differences (H100 vs. A100)?
4. The end-to-end experiments in Section 4.2 lack discussion on model choice, use models that are not large-scale, and don't sufficiently prove that the "Ideal Straggler Delay" is similarly flexible in other environments.

Minor:
5. Figure 12 appears to be missing from the paper.

### Questions
I would increase the rating if the authors addressed my concerns.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper introduces Str-glAR, a novel AllReduce algorithm designed to mitigate the performance bottleneck caused by stragglers in distributed machine learning. The core idea is to utilize the idle time incurred while n-1 nodes wait for the straggler by proactively executing a ReduceScatter operation as a pre-processing step. Once the straggler is ready, the system executes a specialized, lightweight second phase of communication to complete the final, mathematically exact global aggregation.

### Strengths
1. Stragglers are a long-standing and impactful bottleneck in large-scale distributed systems. This paper tackles the problem not by compromising on correctness (approximation) or synchronicity (asynchrony), but by fundamentally redesigning the communication algorithm itself while preserving both. 

2. The most compelling contribution is the insight that by challenging the implicit assumption of a simultaneous start, it is possible to achieve a lower communication cost than the ~2Sβ bound. 

3. The evaluation is robust and covers multiple layers of the system stack, from low-level communication performance (microbenchmarks) and real-world application impact (end-to-end LLM training) to scalability (simulations).

### Weaknesses
1. Failure to Clearly Position its Novelty with Insufficient Discussion of Related Work: The paper's most significant weakness is its failure to adequately differentiate its core idea from the decades-long body of research on "overlap" in systems. The high-level concept of "doing useful work while waiting" is a classic optimization principle. The authors must more rigorously distinguish their "intra-algorithm overlap" from traditional "compute-communication overlap" in the related work section.

2. Overly Idealized Discussion of System Trade-offs: The paper emphasizes its lossless nature but does not sufficiently address the trade-offs between its benefits and the complexity it introduces.

3. Compared to simply dropping a straggler (which is lossy but simple to implement), Str-glAR introduces a complex two-phase protocol. The paper should more honestly discuss the trade-off where a simpler, approximate method might be preferable in latency-critical applications that can tolerate some error.

4. The ReduceScatter phase itself has a non-trivial cost. If the straggler's delay is insufficient to fully hide this cost, the net benefit will diminish. The analysis of scenarios with short delays is not deep enough.

5. The algorithm's design and theoretical analysis are heavily predicated on a single, well-defined straggler. Real-world completion times may follow more complex distributions (e.g., heavy-tailed). The paper needs to discuss the algorithm's behavior and trigger mechanism when a distinct straggler does not exist.

### Questions
1. Could you explicitly contrast Str-glAR's "intra-algorithm overlap" with the classic "compute-communication overlap" in your introduction and related work? Please clarify why your contribution should be considered a fundamental algorithmic innovation rather than another scheduling heuristic.

2. The execution time of the ReduceScatter phase, T_rs, is a critical overhead. Can you provide a more quantitative analysis of how the relationship between the straggler delay T_delay and T_rs impacts the final performance? Is there a tipping point where, if T_delay < T_rs, Str-glAR could perform worse than a standard Ring-AllReduce due to protocol overhead?

3. Your end-to-end experiments rely on a pre-identified, persistent straggler. How would the system adapt if the straggler's identity changes frequently between iterations? What is the runtime overhead of dynamically detecting the first n-1 arrivals and dispatching the appropriate communication strategy?

4. How would Str-glAR perform in the presence of multiple stragglers (e.g., two nodes that are significantly slower than the other n-2)? Can the current protocol be naturally extended to handle this, or would it require a completely new algorithmic design?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
4