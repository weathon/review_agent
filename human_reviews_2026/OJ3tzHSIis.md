# QuIC: Quantum-Inspired Compound Adapters for Parameter Efficient Fine-Tuning

- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
Scaling full fine-tuning of large foundation models strains GPU memory and training time. Parameter Efficient Fine-Tuning (PEFT) methods address this issue via adapter modules which update only a small subset of model parameters. In this work, we introduce Quantum-Inspired Compound Adapters (QuIC Adapters), a PEFT approach inspired from Hamming-weight preserving quantum circuits that can effectively fine-tune a model using less than $0.02\%$ memory footprint of the base model. QuIC adapters preserve pretrained representations by enforcing orthogonality in weight parameters, and have native deployment mechanisms on quantum computers. We test QuIC adapters by fine-tuning large language models like LLaMA and vision transformers on language, math, reasoning and vision benchmarks. In its first-order configuration, QuIC recovers the performance of existing orthogonal methods, while higher-order configurations enable substantial parameter compression (over $40\times$ smaller than LoRA) for a modest performance trade-off, unlocking applications in highly resource-constrained environments.  Through ablation studies, we determine that combining multiple Hamming-weight orders with orthogonality and matrix compounding are essential for performant fine-tuning. Our findings suggest that QuIC adapters offers a promising direction for efficient fine-tuning of foundation models in resource-constrained environments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Quantum-Inspired Compound (QuIC) Adapters, a parameter-efficient fine-tuning method inspired by Hamming-weight preserving quantum circuits. The core idea is to construct adapter matrices using compound matrices (or determinants of minors) up to order K, which naturally preserve orthogonality and enable extreme parameter compression. The authors evaluate QuIC on language tasks and vision tasks, demonstrating competitive Pareto efficiency.

### Strengths
1. The math seems relative rigorous. The use of compound matrices naturally ensures orthogonality (Lemma 1) and provides clear parameter count formulae (Lemma 2). The connection to Hamming-weight preserving quantum circuits is well-established.
2. The evaluation is relatively comprehensive, including both language and vision tasks, and with various models sizes.
3. the appendix includes detailed ablation studies, providing good insights on what makes QuIC work.
4. the proposed method is extremely parameter efficient

### Weaknesses
1. Although the method is extremely parameter efficient and achieves "competitive Pareto efficiency", across most of the tasks, QuIC underperforms compared to the baseline methods. 
2. The paper says QuIC has "native quantum computer integration", but does not provide any experimental validation. All experiments are run on GPUs. It is unclear if it is a method that is meant for near-term quantum deployment with current experiments as proof of concept, or if this is a purely non-quantum method that draws inspiration from quantum computation.
3. The paper focuses on compound orders K<=3. What happens with higher K? Does the performance increase, or there is diminishing return?
4. The experiments seem to only use a narrow range of configurations. Given the main selling point of efficiency, it is valuable to show a more complete exploration of the trade off between parameter efficiency and performance, and if QuIC can dominate other methods.

### Questions
1. As mentioned in weaknesses, is QuIC intended to be eventually deployed on quantum computers, or is it a non-quantum method inspired by quantum concepts? If the former, is it possible to provide quantum circuit simulations or resource estimates for running on real quantum hardware?
2. Can QuIC achieve performance competitive with OFT/BOFT if you increase parameters (e.g., larger K, more adapters, different rank settings)? What does the full performance-parameter trade-off curve look like?
3. QuanTA (one of the baselines) also seems to be a quantum-inspired method. Can the authors provide a more detailed comparison with QuanTA beyond the quantum deployment critique? What are the differences and connections between QuIC and QuanTA? In what scenarios would a practitioner choose QuIC over QuanTA or vice versa?
4. The authors mention that OFT's motivation is to preserve pretrained knowledge. Does QuIC also demonstrate better retention of pretrained capabilities compared to LoRA?

I cannot recommend the paper to be accepted in the current form, but if the authors could help resolve my concerns, I'm happy to raise the score.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces QuIC (Quantum-Inspired Compound) Adapters, a new Parameter-Efficient Fine-Tuning (PEFT) method inspired by Hamming-weight preserving quantum circuits. The method constructs compound matrices of increasing order to generate orthogonal, multiplicative adapters that can be integrated into transformer-based models. The authors claim that QuIC achieves comparable or slightly reduced accuracy relative to existing methods (LoRA, OFT, BOFT, AdaLoRA) while offering extreme parameter compression (up to 40× smaller than LoRA) and native compatibility with quantum implementations.

### Strengths
1. The construction of compound matrices from Hamming-weight preserving quantum circuits provides an original mathematical and conceptual bridge between quantum computing and PEFT.
2. Achieves remarkable compression (<0.02% memory of base model) with minimal accuracy loss, demonstrating practicality for low-resource or edge deployment.
3. Evaluations span multiple modalities (language, vision, math, reasoning) and strong baselines (LoRA, OFT, BOFT, AdaLoRA, QuanTA).
4.  The paper provides formal lemmata for orthogonality preservation, parameter counting, and computational complexity.
5.  Includes thoughtful ablations on compound order, orthogonality, and alternative combinatorial operations (determinant vs. max/avg).

### Weaknesses
1. The intuition connecting Hamming-weight preservation to improved fine-tuning efficiency is conceptually appealing but not rigorously shown to yield empirical benefits beyond orthogonality constraints. The determinant-based compounding operation lacks a clear learning-theoretic explanation.
2.  On most GLUE and MATH10K tasks, QuIC underperforms baseline PEFT methods in raw accuracy. The paper leans heavily on Pareto efficiency to claim superiority; this metric, while interesting, does not fully compensate for weaker task performance.
4.  Experiments on LLaMA-7B are promising but minimal. The claim of “quantum-native readiness” remains speculative, with no practical implementation or simulation on quantum hardware.
5.  The introduction suggests QuIC “enables native deployment on quantum computers,” but the experiments are entirely classical. This risks overstating the quantum relevance.
6. While emphasizing parameter compression, it lacks comparison with efficient parameter fine-tuning methods such as VeRA[1] and FourierFT[2].

[1] Kopiczko, Dawid J., Tijmen Blankevoort, and Yuki M. Asano. "Vera: Vector-based random matrix adaptation." arXiv preprint arXiv:2310.11454 (2023).

[2] Gao, Ziqi, et al. "Parameter-Efficient Fine-Tuning with Discrete Fourier Transform." International Conference on Machine Learning. PMLR, 2024.

### Questions
1. Can you empirically demonstrate that the “compound determinant operation” contributes meaningfully to performance, rather than simply serving as an orthogonal regularizer?
2. How sensitive is QuIC to the choice of compound order K? Could the performance plateau indicate diminishing returns or gradient instability?
3. Could QuIC adapters be adapted for additive forms (e.g., LoRA-style) and, if so, how would that compare computationally?
4. The “native quantum implementation” claim is intriguing — can you outline a realistic near-term setup (e.g., number of qubits, gate depth) where QuIC could run on current quantum hardware?
5. How does QuIC behave under continual or multi-task fine-tuning? Does orthogonality indeed prevent catastrophic forgetting as well as OFT/BOFT?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This work introduces QuIC (Quantum-Inspired Compound) Adapters, a multiplicative, orthogonality-preserving PEFT method. The authors build QuIC from small “base” matrices whose compound matrices are assembled block-diagonally to form an (optionally) orthogonal adapter. Notably, when only the first-order compound is used, QuIC recovers OFT. Their adapter exposes simple knobs: choice of compound orders, block size, orthogonality enforcement, and parameter sharing, which trade off accuracy and trainable-parameter count, with all trainable weights confined to the small base matrix. Empirically, across language (GLUE with DeBERTaV3), vision (VTAB-1k with DINOv2), and reasoning/math tasks (LLaMA-2 7B on MATH10K and DROP), QuIC uses orders-of-magnitude fewer trainable parameters than strong baselines while attaining good Pareto scores and, in a few cases, competitive accuracy despite significant parameter reductions. Further, they provide experiments to demonstrate their adapter lies in the pareto frontier of performance v.s. memory. Finally, the design is “quantum-inspired” in that it mirrors how Hamming-weight-preserving quantum layers act block-wise on fixed-Hamming-weight subspaces via compound matrices.

### Strengths
* The paper tests multiple baselines across language and vision models and datasets. 
* They clearly mention their similarities to previous orthogonal adapters; for example, recovering OFT under their framework. 
* They demonstrate better Pareto scores for their adapter (particularly, it often requires fewer parameters than other methods).

### Weaknesses
* The frontier (Fig. 4.a) is constructed from single configuration points for each baseline rather than full \emph{parameter-budget sweeps} (e.g., LoRA ranks, (B)OFT block sizes). When tracing each method’s own curve, the frontier could shift significantly. 
* Results mostly compare against much larger adapters (in # of parameters). For a closer apples-to-apples comparison (in terms of benchmark accuracy as opposed to Parameter score), it would be nice to see how QuIC compares when parameter-matching it to other adapters. 
* Results emphasize adapter memory / trainable parameters. I think it would be nice to see what the FLOPs and latency tradeoff is in a table similar to the ones already included in the work. For multiplicative or block-diagonal adapters, the unfused per-token cost can exceed LoRA’s, unless the block count is very high or weights are fused. It would be interesting to observe what the tradeoff is in FLOPs space as well.

### Questions
Do you expect an additive form of QUIC to be more or less effective than a multiplicative form? I'd appreciate experiments on this.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes QuIC (Quantum-Inspired Compound Adapters), a PEFT method inspired by quantum compound matrices. The core idea is to construct orthogonality-preserving multiplicative adapters from a small base matrix A by composing its higher-order compounds ($A^{(k)}$), assembled into a block-diagonal structure that modulates pretrained weights ($W’ = \Delta W_Q W^\*$). This formulation generalizes Orthogonal Fine-Tuning (OFT) as a special case  and aims to achieve higher expressivity per parameter through determinantal “compounding.” Empirical evaluations on GLUE, VTAB-1k, and LLaMA-2-7B reasoning tasks show competitive accuracy with strong parameter compression, along with ablations demonstrating the effects of orthogonality, compound order, and adapter capacity.

### Strengths
+ The orthogonality-preserving property of compound matrices is rigorously proved via the Cauchy–Binet theorem, and the resulting adapter formulation is mathematically coherent. The method reduces cleanly to OFT when using only first-order compounds, providing a clear conceptual lineage.

+ The determinantal compound mechanism introduces a distinct parameterization not explored in prior PEFT work (LoRA, OFT, BOFT). The “quantum-inspired” construction offers a compact and theoretically elegant way to expand representational capacity while maintaining orthogonality.

+ The paper conducts thorough ablations on compound orders ($C_1–C_3$), orthogonality enforcement, and adapter scaling, demonstrating the necessity of including first-order components for stable training and showing how orthogonality substantially improves performance. The results are reproducible and resource-efficient.

+ The paper report parameter counts and memory usage, and they discuss inference mergeability and orthogonality drift — showing a good understanding of practical constraints.

### Weaknesses
- The main NLP evaluation is on GLUE, which is now a saturated and outdated benchmark for PEFT methods. Modern evaluations typically use SuperGLUE, MMLU, BIG-Bench Hard, or instruction-tuning datasets to assess adaptation quality in LLMs. As a result, the GLUE results provide limited evidence of QuIC’s effectiveness for large-scale or reasoning-centric fine-tuning.

- While the compound adapter design is original, the overall mechanism—orthogonality-preserving multiplicative fine-tuning—extends from OFT/BOFT. The “quantum inspiration” remains mostly metaphorical rather than a functional contribution.

- It seems that no multi-seed confidence intervals are reported, and there is no wall-time or training throughput analysis. Construction costs for compound matrices (potentially combinatorial in $n$ and $k$) are not profiled, which limits claims of efficiency. Evaluations on larger or more diverse LLM tasks would better establish generality.

- It remains unclear whether QuIC maintains efficiency or scalability for deeper LLMs (>7B) or high-rank adapters. The method’s deployment benefits over simpler OFT variants are therefore uncertain.

### Questions
Please consider to address questions in the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
