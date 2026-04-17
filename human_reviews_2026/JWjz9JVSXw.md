# CMPS: Constrained Mixed Precision Search

- Decision: Reject
- Scores: 2, 2, 4

## Abstract
The increasing complexity of deep neural networks (DNNs) requires effective model compression to reduce their computational and memory footprints for deployment on resource-constrained hardware. Mixed-precision search is a prominent bit allocation method based on neural architecture search (NAS) that has been shown to significantly reduce the DNN footprint while preserving the accuracy of the model by allocating bits to each layers based on their quantization sensitivity. However, mixed-precision search is often defined as a dual optimization problem handled with a single heuristic objective function, which does not provide strong guarantees of the resulting compression rate. We propose a post-training reformulation of mixed precision search as an explicit constrained optimization problem, solved using interior-point methods within a framework based on NAS. Our method requires minimal calibration data, as few as 128 samples, in a post-training setting. We corroborate this approach with experiments that span multiple transformer architectures with up to 4 billion parameters, using the MXFP family of data formats. We show that this constrained formulation provides users with higher resolution over compression rates, and we show that explicitly satisfying hardware budgets while optimizing for accuracy can outperform uniform allocation methods, improving performance by up to several standard deviations over the uniform baselines.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces a differenciable NAS algorithm for data format allocation in a network. It first formalizes the optimization problem, including architecture constrains (the maximum average number of bits for a model), then proposes a gradient descent based heuristic for solving the mixed precision data format allocation problem. While the method is fully post-training, it still requires a small calibration data set to perform the training of the data format precision parameters.
The paper shows that using this formalism, mixed precision constrained NAS can achieve better results than uniform quantization.

### Strengths
- The mathematical formulation of the constrained optimization problem for mixed precision data format allocation seems fairly general;
- The large number of results (which are combinations between models and tasks used for the calibration) seems to demonstrate the robustness of the approach.

### Weaknesses
- The paper completely lack any comparison with the state-of-the-art! No comparison with other mixed-precision post-training optimization methods is even attempted... yet plenty exists. That is clearly a major issue in this paper.
- While the fundations of differentiable NAS methods seems to be adequately described and cited, the novelty of the proposed method remains hard to grasp. I would suggest to add a short but clear statement on what it brings compared to the closest SoTA work.
- While the method seems very general, it is frustrating see it tested on a single NAS scenario, namely, 4.5 bit mixed precision with the MXFP data format. What about mixing different formats (integer, FP...)? Or testing other maximum average number of bits (like 3.5, or 5.5...)?
- The perplexity/accuracy gains of the method remain modest and the proposed NAS scenario is too limited.

### Questions
Please carefully answer the issues mentioned in the weaknesses section.

I may increase my rating provided that at least 1) quantitative comparison with other SoTA methods is provided. 2) Additional NAS scenario, beyond 4.5 bit mixed precision is evaluated.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
A DNAS-based post-training mixed-precision quantization method (CMPS) is proposed. CMPS provides fine-grained control over model compression, enabling stable and predictable performance. The proposed CMPS method is compared with uniform quantization baselines, demonstrating the advantages of learnable mixed-precision bit allocation.

### Strengths
1. This paper works on the post-training mixed-precision quantization with controllable compression ratios. The problem studied is important and the motivation is clear.
2. The detailed theoretical analysis is provided.

### Weaknesses
1. Quantization details are missing. It seems that CMPS is a weight-only quantization method. However, the quantization details are not provided.
2. Optimization cost is not provided. The advantage of PTQ is its efficiency in quantization optimization. The CMPS relies on end-to-end tuning with multiple branches. The speed and memory cost overheads should be reported.
3. Comparison with previous methods is also missing. The authors didn't provide any quantization details, including the uniform quantization baselines. In the llm quantization literature, many high performance PTQ methods are proposed. What's the performance advantages over these methods? How the proposed CMPS can be combined with these techniques? Moreover, the authors only compared with uniform quantization baselines, the comparison with previous mixed-precision methods are missing.
4. In several places, it says "hardware-constrained bit allocation", however, only "total model size in bits" is modeled during the optimization. Moreover, only two bit levels are explored in the bit allocation (MXFP4 and MXPF8).
5. In the experiments part, previous methods commonly use wiki2 for calibration in addition to C4. For zero-shot scenario, only one task of LAMBADA is evaluated, which is clearly not enough. The largest model used is 3B, experiments on larger models or architectures like MoEs are also needed.
6. In the limitations, regarding the statement "the memory required to hold activations or gradients for multiple low-bit options might still be comparable to, or less than, holding a single higher-precision (e.g., FP16 or BF16) baseline tensor", more careful and precise expression should be used. Many PTQ methods do not need to store all activations, and the gradients are not needed. However, in CMPS, full-precision activations of all layers and gradients are needed, which expands the memory usage. If these tensors (activations and gradients) can be stored in low-bit, then the authors should verify it use controlled experiments.

### Questions
Please refer to the Weaknesses for further questions.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work proposes a new constrained mixed precision search for post training quantization. To solve the constrained optimization problem the authors leverage barrier-based interior-point method. The method keeps model weights frozen, and needs only a small calibration set (128 samples). Experiment on various LLMs report consistent gains over uniform precision baselines at the same or lower effective bit budgets.

### Strengths
1. The work discusses the problem of reducing computational and memory footprints for deployment of DNNs which is practical and important.

2. The paper is well written and easy to follow.

3. 4.5-bit the proposed method often beats MXFP in terms of perplexity, on the examined benchmarks.

### Weaknesses
1. The authors claim that after rounding there always remains a strictly feasible solution with respect to the budget. I believe a proof for this claim is required.

2. The comparison is limited. The work only compares itself to the MX baselines but there are many other strong PTQ techniques. Only a single dataset was used in the experiments.

3. No thoughput\latency comparisons are provided.

4. The improvement over the baselines is marginal.

5. How does the method operate compared to integer PTQ techniques?

6. According to the experiments, the proposed algorithm does not always meet the constraint.

### Questions
How were the samples for calibration chosen?

What is the meaning of the upsidedown question mark in the caption of Figure 2?

There is a typo in  in line 75 (double "“Our contributions are as follows:")

### Soundness
3

### Presentation
3

### Contribution
2
