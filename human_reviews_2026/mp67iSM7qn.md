# Beyond Student: An Asymmetric Network for Neural Network Inheritance

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Knowledge Distillation (KD) has emerged as a powerful technique for model compression, enabling lightweight student networks to benefit from the performance of redundant teacher networks. However, the inherent capacity gap often limits the performance of student networks. Inspired by the expressiveness of pretrained teacher networks, a compelling research question arises: is there a type of network that can not only inherit the teacher’s structure but also maximize the inheritance of its knowledge? Furthermore, how does the performance of such an inheriting network compare to that of student networks, all benefiting from the same teacher network? To further explore this question, we propose InherNet, a neural network inheritance method that performs asymmetric low-rank decomposition on the teacher’s weights and reconstructs a lightweight yet expressive network without significant architectural disruption. By leveraging Singular Value Decomposition (SVD) for initialization to ensure the inheritance of principal knowledge, InherNet effectively balances depth, width, and compression efficiency. Experimental results across unimodal and multimodal tasks demonstrate that InherNet achieves higher performance compared to student networks of similar parameter sizes. Our findings reveal a promising direction for future research in efficient model compression beyond traditional distillation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces InherNet, a novel neural network compression method that extends beyond traditional knowledge distillation by directly inheriting both the structure and knowledge of teacher networks. InherNet leverages asymmetric low-rank decomposition and SVD-based initialization to construct lightweight but expressive networks while preserving the teacher’s representational capacity. The proposed method applies a unified framework that balances depth, width, and compression efficiency, enabling it to achieve superior performance compared to traditional student networks of similar size. Theoretical analyses provide rigorous guarantees on convergence, parameter efficiency, and representational power preservation, making InherNet a robust and impactful contribution to the field of efficient model compression.

### Strengths
1. The idea of asymmetric low-rank decomposition combined with an expert-style structure is innovative. The proposed approach is distinct from existing KD methods and parameter-efficient techniques like LoRA, offering a fresh perspective on model compression.
2. The paper addresses an important problem in neural network compression and proposes a novel method that goes beyond the limitations of traditional KD. By directly inheriting the teacher's structure and knowledge, InherNet opens new possibilities for designing lightweight networks with high performance.
3. The use of SVD-based initialization is a key technical contribution, ensuring stable optimization and effective knowledge transfer. The theoretical analysis of convergence guarantees, parameter efficiency, and representational power preservation is comprehensive and well-supported.

### Weaknesses
1. The paper briefly mentions LoRA but does not include a detailed comparison. Since LoRA also uses low-rank decomposition, it would be valuable to analyze InherNet’s advantages (or trade-offs) compared to LoRA in terms of performance, scalability, and parameter efficiency.
2. The asymmetric design and expert-style structure may add implementation complexity. The paper could benefit from pseudocode or a detailed algorithmic overview to make the method more accessible to practitioners.

### Questions
1. Explore InherNet’s performance on more complex multimodal tasks, such as Visual Question Answering (VQA) or video-text retrieval, and analyze how well it handles cross-modal alignment?
2. Could the authors provide a thorough empirical and theoretical comparison with LoRA, highlighting the differences in design, performance, and scalability?

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
3

### Summary
This paper proposes InherNet, a Neural Network Inheritance (NNI) framework that directly inherits the structure and knowledge of a teacher model by performing asymmetric low-rank decomposition and SVD-based initialization on the teacher’s weights, rather than relying on the indirect matching strategies commonly used in traditional knowledge distillation.

### Strengths
The paper systematically introduces the "Network Inheritance (NI)" paradigm for the first time, breaking away from the conventional knowledge distillation (KD) approach that transfers knowledge solely through soft labels. Comprehensive evaluations on multiple datasets, including CIFAR-100, GLUE, and CC3M, demonstrate the generality and robustness of the proposed method.

### Weaknesses
1. The related work “Matrix Compression via Randomized Low Rank and Low Precision Factorization” also employs LoRA for model compression. The authors need to clarify the distinction between their approach and that work, beyond merely stating that they come from different research domains.

2. The authors should provide an analysis of model performance under varying compression ratios, showing how performance changes as the compression rate increases.

3. The role of the MoE (Mixture of Experts) module is unclear. In traditional MoE architectures, different expert models are typically responsible for distinct tasks. However, in this work, each expert is fine-tuned on the same dataset without explicit specialization, making the proposed MoE appear more like a parameter expansion trick rather than a true mixture-of-experts mechanism.

4. It would strengthen the paper if the authors could evaluate the proposed method on larger-scale models to verify its scalability and generalization.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces InherNet, a novel neural network inheritance method designed to overcome the capacity gap that limits traditional student models in Knowledge Distillation (KD). Instead of training a student to mimic the teacher, InherNet directly inherits the teacher’s structure and knowledge through an asymmetric low-rank decomposition of the teacher’s weights. Using Singular Value Decomposition (SVD) for initialization, InherNet reconstructs a lightweight yet expressive network that retains the teacher’s principal components while maintaining architectural consistency. This approach effectively balances model depth, width, and compression efficiency. Experiments on both unimodal and multimodal tasks show that InherNet outperforms conventional student networks with similar parameter counts, demonstrating a new direction for efficient model compression beyond standard distillation frameworks.

### Strengths
1. The proposed SVD-Driven NNI algorithm directly compresses the teacher models instead of training a separate student model during the knowledge distillation process. Therefore, the proposed algorithm can build more complex but lightweight network architectures without harming the number of model parameters and inference time.

2. The proposed algorithm is supported by a comprehensive theory analysis. Thus, readers can fully understand the advantages of the proposed algorithm.

3. The paper demonstrates the effectiveness and efficiency of the proposed SVD-Driven NNI through comprehensive experiments across different models and applications.

### Weaknesses
1. The proposed algorithm performs layer-wise compression during the knowledge distillation process. It does not account for the correlation between layers when performing layer-wise compression. Due to this, the proposed algorithm might have suboptimal results after compression.

### Questions
1. Since the proposed algorithm performs layer-wise compression, does it result in suboptimal results for compressed models?

2. The proposed algorithm leverages multi-expert structure inheritance to perform compression. Does the number of expert heads result in quantitative results?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces a new method called InherNet, which is designed to improve how smaller neural networks (like student models) can "inherit" knowledge from larger ones (teacher models). Instead of just mimicking the outputs like traditional knowledge distillation, InherNet directly reuses and compresses the teacher’s internal structure using low-rank decomposition and SVD. It builds a kind of lightweight but expressive network that can perform well even with fewer parameters. The authors test InherNet across a range of tasks—vision, language, and multimodal—and show that it generally outperforms existing distillation methods, both in speed and accuracy. They also include a lot of theoretical analysis to explain why this approach works.

### Strengths
1. The authos also provide a rigorous mathematical analysis of the InherNet architecture, such as (1) convergence guarantees under standard assumptions, (2) proofs of parameter efficiency and knowledge preservation based on singular value spectrum, (3) formal definitions of Parameter Efficiency, Expressivity-to-Parameter Ratio, and Approximation Error.

2. The design of a fixed asymmetric expert-head structure combined with SVD-based decomposition is novel. The paper provides a clear architectural comparison with LoRA and other PEFT approaches. Their proposed method offers a new paradigm in model inheritance, distinct from traditional KD or PEFT methods.

### Weaknesses
1. I noticed is that a couple of the baselines, like MLKD and Logit Std., were trained for more epochs than other methods. That makes it tricky to know if InherNet is really better, or if the training schedule just favors it. For a fair comparison, I think all methods should be evaluated under the same settings. Otherwise, the results lose a bit of their strength.

2. In the analysis section, the authors mention that distillation can hurt performance for large InherNet models, which is super interesting, but they kind of gloss over it. That feels like a missed opportunity. If there’s a sweet spot where inheritance works better than distillation, we need more detail on that. Right now, it leaves a bit of a gap in understanding how to apply the method in different cases.

3. While the paper does include T5 for some NLP benchmarks, the whole method seems mostly built around CNNs. If this method is going to be truly general, we need to know how it applies to modern Transformer models, especially for large-scale language tasks. At the very least, I’d like to see a bit more discussion or even a small-scale experiment showing that InherNet can handle Transformer-style architectures.

### Questions
N.A. See weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
