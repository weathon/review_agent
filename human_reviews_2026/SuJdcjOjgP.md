# NLI : Non-uniform Linear Interpolation Approximation of Nonlinear Operations for Efficient LLMs Inference

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 2, 6, 8

## Abstract
Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of tasks, but their deployment is often constrained by substantial memory footprints and computational costs. While prior work has achieved significant progress in compressing and accelerating linear layers, nonlinear layers—such as SiLU, RMSNorm, and Softmax—still heavily depend on high-precision floating-point operations. In this paper, we propose a calibration-free, dynamic-programming-optimal, and hardware-friendly framework called \underline{N}on-uniform \underline{L}inear \underline{I}nterpolation (NLI). NLI is capable of efficiently approximating a variety of nonlinear functions, enabling seamless integration into LLMs and other deep neural networks with almost no loss in accuracy. NLI ingeniously recasts cutpoint selection as a dynamic-programming problem, achieving the \emph{globally} minimal interpolation error in $\mathcal{O}(M \times N^2)$ time via Bellman’s optimality principle. Based on the NLI algorithm, we also design and implement a plug-and-play universal nonlinear computation unit. Hardware experiments demonstrate that the NLI Engine achieves more than 4× improvement in computational efficiency compared to the state-of-the-art designs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes NLI, a software–hardware co-design for efficient approximation and computation of nonlinear operations in LLM inference. Specifically, the NLI framework performs non-uniform interpolation of nonlinear functions, formulating the cutpoint selection as a DP problem. In addition, NLI introduces a dedicated hardware engine for it that can be seamlessly integrated into existing LLM inference hardware.

### Strengths
* The analysis of the limitations of existing approaches (e.g., NN-LUT) for LLM inference is insightful and interesting.
* The proposed solution, encompassing both algorithmic and hardware aspects, appears technically sound, and the evaluations are thorough and convincing.

### Weaknesses
The motivation of this work is not sufficiently persuasive, which makes the overall contribution difficult to fully appreciate. While the authors emphasize the computational complexity of nonlinear operations in LLMs, I believe their actual overhead constitutes only a small fraction of the end-to-end inference time, since linear operations remain dominant even after various optimizations or compression techniques. Moreover, it is questionable whether the hardware cost of special function units in existing GPUs or NPUs is significant enough to justify the need for this entire line of research.

### Questions
Do we really need to optimize nonlinear operations in LLMs? This is my primary and most critical question regarding this paper. I would like to see both quantitative and qualitative discussions on the actual performance and hardware overheads associated with nonlinear operations, to better justify the necessity of this research direction.

### Soundness
1

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the high computational overhead introduced by nonlinear activation functions when deploying large language models on edge devices. It proposes a calibration-free, hardware-friendly non-uniform linear interpolation method called NLI, which employs dynamic programming for optimal segmentation. The NLI framework consists of two components: algorithm design and hardware design. 
Algorithmically, the authors propose NLI-Algorithm, which leverages dynamic programming to select optimal segmentation points on the FP16 numerical grid. This constructs a lookup table (LUT) that is reusable across different layers and models without requiring data calibration. 
On the hardware side, the authors design the NLI-Engine module, which implements a two-level address translation scheme and pipelined architecture. This design significantly reduces the number of comparators and overall hardware overhead. 
Software experiments demonstrate that this architecture achieves nearly negligible accuracy loss across various LLMs and vision models, while showing substantial improvement over NN-LUT.

### Strengths
1. Novelty: The use of dynamic programming for non-uniform segmentation point sampling represents a significant innovation, surpassing heuristic or data-dependent calibration methods. Experimental results confirm that this approach incurs almost no accuracy loss.
2. Full-Stack Hardware-Software Co-Design: The NLI algorithm was designed with hardware efficiency in mind from the outset, making it highly practical for edge deployment. The two-level address translation scheme is a particularly clever hardware optimization that drastically reduces the number of comparators, directly addressing a major source of area and power consumption. This integrated approach forms a compelling demonstration of the method's effectiveness.

### Weaknesses
1.	Computational Complexity of DP Search: The O(MN²) complexity of the DP search implies that it may become a bottleneck when attempting to directly search for a larger number of finer-grained segments over the full FP16 grid. The scalability of the DP method for more granular optimizations remains a potential limitation.
2.	Handling Outliers via Clamping: Although the authors argue through coverage analysis that the impact of clamping is negligible, its effect on worst-case performance or robustness in certain unexamined models or tasks could be a point of concern. This may necessitate the design of more robust fault-tolerant mechanisms.
3.	Dependence on Function Characteristics: The effectiveness of NLI is inherently tied to the behavior of the function's second derivative. Consequently, its robustness against potential activation functions that may emerge in future models remains somewhat uncertain.

### Questions
1. In the paper, minimizing the average relative error was chosen as the objective function for dynamic programming. Did the authors experiment with other objective functions (e.g., maximum absolute error, root mean square error)? Was the choice of relative error primarily to address numerical stability issues near small activation values, or did it empirically lead to better downstream task accuracy?
2. Although extreme activation values account for a small proportion, they may be critical in certain specific tasks or prompts. Could the authors provide a more in-depth analysis, such as showing which layers or attention heads these truncated activation values are primarily distributed in? Has there been any consideration of adopting a conservative, high-precision fallback computation mode for these "tail" values?
3. The dynamic programming is performed on an FP16 numerical grid, which implicitly defines an input value range. If a new activation function emerges in the future, whose effective dynamic range significantly exceeds that of FP16 (e.g., requiring FP12 or BF16 precision for representation), would the current method still be applicable? Would it be necessary to rerun the DP?
4. How do the authors guide designers in selecting the optimal configuration for the number of "macro intervals" (M) and the number of "micro cells" (D_n) per interval? Does the architecture support dynamic reconfiguration to accommodate different functions?

### Soundness
3

### Presentation
3

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
The paper proposes a hardware-friendly scheme, NLI, for approximating nonlinear operations in FP16 domain for faster inference. NLI uses non-uniform spline linear interpolation in the FP16 domain, and selects interpolation cut-points with dynamic-programming to minimize interpolation error, yielding a calibration-free lookup table for each nonlinear operation. The authors also design a plug-and-play NLI unit and present hardware experiments showing speedup improvement in compute/area efficiency compared to prior designs.

### Strengths
This paper addresses a critical problem of the expensive nonlinearities in LLMs for faster inference, and provides a well-defined algorithmic contribution: the dynamic-programming formulation for optimal non-uniform linear interpolation is mathematically clear and hardware-aware. It’s defined in a two-stage manner for better efficiency and yields a calibration-free lookup table. 
It provides a clear (performance and efficiency) evaluations of the proposed method, including error analysis, influence to the downstream tasks and perplexity across multiple open models, and the promising results demonstrate great potential of NLI.
It also includes efficiency analysis, proposed hardware implementation and evaluation, bridging algorithmic/design work with real system implications. The method is presented as plug-and-play manner across a variety of nonlinear functions.

### Weaknesses
NLI selects cutpoints from the FP16 domain and performs interpolation using FP16 arithmetic (Stages 1–4). A direct comparison with naïve FP16 evaluation of the same nonlinear operations should be provided to contextualize NLI’s advantages. 

- This comparison should include for example:
i) error-wise: NLI reports an error of $1.2\sim1.5\times10^{-3}$; what is the corresponding error for plain FP16 evaluation?
ii) performance-wise: How do models perform across tasks when nonlinear operations are evaluated directly in FP16?
iii) efficiency-wise: How does the computational efficiency of NLI (lookup + FP16 linear arithmetic) compare with that of standard FP16 implementations? Additionally, NLI could also select cutpoints from the FP32 domain and use FP32 arithmetic for Stages 1–4, will there be any efficiency concern (compared to naive FP32 and current fp16 implementations)?

Such analyses would substantiate the benefits of NLI over standard FP16 inference.

The two-level cut-point design is promising, but the paper would benefit from ablation studies and discussion on its configuration, such as how many non-uniform cutpoints are chosen at the first level, how many uniform ones at the second level, and what the optimal total number of cut points is. 

Addressing these concerns could justify a higher score.

### Questions
- what’s a good error level to avoid model performances degradations (like on zero-shot tasks on perplexity)? Currently $1.2\sim1.5\times10^{-3}$ looks good

- Do you need to pre-fix the set of cutpoints before implementing the NLI hardware version? Or alternatively, after implementing the proposed NLI on hardware, does it support taking in any custom set of cutpoints?

- In table 3, the non-uniform version performs worse than NLI, do you have the results (and averaged) for other tasks, just like table 1? Could this suggest that DP is not good enough for minimizing the error when the number of cuts increases?

- In table 8, NLI (with fp16 arithmetic from stage 1-4) is even slightly better than FP32 for 7 out of 8 models, why is this so? And again what’s the performance of using FP16?

- Do you plan to make the code public for reproduciblity?

- Line 339, the reference to the algorithm broke, and in page 12, the spacing between references are so large, can you fix these?

### Soundness
3

### Presentation
3

### Contribution
3
