# Enabling Sparse Autoencoders for Topic Alignment in Large Language Models

- Decision: Reject
- Scores: 1, 3, 8, 6, 6

## Abstract
Recent work shows that Sparse Autoencoders (SAE) applied to LLM layers have neurons corresponding to interpretable concepts. Consequently, these SAE neurons can be modified to align generated outputs, but only towards pre-identified topics and with some parameter tuning. Our approach leverages the interpretability properties of SAEs to enable alignment for any topic. This method 1) scores each SAE neuron by its semantic similarity to an alignment text and uses them to 2) modify SAE-layer-level outputs by emphasizing topic-aligned neurons. We assess the alignment capabilities of this approach on diverse public topics datasets, including Amazon reviews, Medicine, and Sycophancy, across open-source LLMs, GPT2, and Gemma with multiple SAEs configurations. Experiments aligning to medical prompts reveal several benefits over fine-tuning, including increased average language acceptability (0.25 vs 0.5), reduced training time across multiple alignment topics (333.6s vs. 62s), and acceptable inference time for many applications (+0.00092s/token). Our anonymized open-source code is available at https://anonymous.4open.science/r/sae-steering-8513/README.md.

## Human Reviews

## Human Reviewer 1

### Rating
1

### Rating Number
1

### Confidence
5

### Summary
The paper's content is interesting, but unfortunately, the code contains the name of a person presumed to be the author. Therefore, I think this paper should be rejected.

### Strengths
The paper's content is interesting, but unfortunately, the code contains the name of a person presumed to be the author. Therefore, I think this paper should be rejected.

### Weaknesses
The paper's content is interesting, but unfortunately, the code contains the name of a person presumed to be the author. Therefore, I think this paper should be rejected.

### Questions
The paper's content is interesting, but unfortunately, the code contains the name of a person presumed to be the author. Therefore, I think this paper should be rejected.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper introduces a method for topic alignment in LLMs using Sparse Autoencoders (SAEs) to emphasize sparse layer activation relevant to specific topics without fine-tuning the parameters. By scoring sparse token activation results, the method is able to perform topic alignment without fine-tuning the parameters, reducing training time and enhancing interpretability. Experiments across diverse datasets demonstrate the method’s adaptability.

### Strengths
1. Novel approach for using sparse token encoding to do the topic alignment.
2. This work is highly relevant for applications requiring frequent topic shifts in generated text, offering a scalable and interpretable alternative to fine-tuning.

### Weaknesses
1. Poor performance. The method's performance remains suboptimal. While the authors present a novel approach to topic-alignment fine-tuning, the results showcased in Table 2 still lack clarity, with the distance score favoring traditional fine-tuning as the more effective option. In addition, providing a ground-truth case will make it easier to know what kind of output is desired. From the current cases, I believe all responses are not helpful for the users. Meanwhile, could the authors clarify what specific benefits their method might offer in comparison, such as interpretability or flexibility, that justify its use over standard fine-tuning except for the efficiency? Efficiency considerations are addressed further in Weak Point 4.
2. Lack of proper comparison. The paper lacks a comprehensive comparison with other common methods for topic-based fine-tuning, such as prompt hints, LoRA, and token-based fine-tuning. Given that a simple prompt hint typically provides a strong baseline, it is unusual that this comparison is missing. It would be helpful if the authors could add this baseline, alongside relevant metrics like computational efficiency, interpretability, and overall performance, to provide a fuller assessment of their method’s value relative to established alternatives.
3. Lack of proper target models and experiment details. The experimental setup lacks sufficient detail regarding the target models used. Although GPT-2 and Gemma are mentioned briefly, there is no information on model size, version, or specific configurations. To support a more robust conclusion, I recommend that the authors apply their method across a variety of large language models (LLMs) and provide clear details on each model's configurations including framework, version, and size. This would mitigate any potential bias introduced by the training data of a single LLM and enhance the generalizability of the findings
4. The cost analysis in the paper is incomplete, particularly regarding the training time associated with adding a sparse encoding layer. Since a sparse encoding layer is not a standard module for most LLMs, the additional time and computational resources required for training should be included in the analysis. A more detailed breakdown of these costs would give readers a more accurate understanding of the method’s overall efficiency and practical feasibility

### Questions
1. What LLM did you use for the experiment? In detail, what is the version and model size? 
2. Have you tried using simple prompt hints? For example, please respond as a professional doctor or just use GPT-4 to rephrase the prompt to the desired domain.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents a novel approach for using Sparse Autoencoders (SAEs) to enable topic alignment in Large Language Models (LLMs) without requiring computationally intensive fine-tuning. The authors propose two main methodological contributions: (1) a scoring mechanism to identify SAE neurons relevant to alignment topics, and (2) a modification approach that uses these scores to alter SAE layer outputs in a context-sensitive way. The paper demonstrates the effectiveness of their approach across multiple datasets (Amazon reviews, Medicine, Sycophancy) and models (GPT2, Gemma), showing improvements in language acceptability and training efficiency compared to fine-tuning approaches.

### Strengths
1. **Novel Approach**: The paper presents an innovative method for topic alignment using SAEs, addressing a significant gap in the literature. The approach is well-motivated by the limitations of existing methods like fine-tuning.

2. **Technical Depth**: The methodology is thoroughly developed with clear mathematical formulations for both the scoring mechanism and modification approach. The authors provide detailed justification for their design choices.

3. **Comprehensive Evaluation**: The experimental evaluation is extensive, covering:
   - Multiple datasets and topics
   - Different model architectures
   - Various SAE configurations
   - Both quantitative and qualitative metrics

4. **Practical Utility**: The approach shows promising results with:
   - Reduced training time (333.6s vs 62s)
   - Acceptable inference time overhead (+0.00092s/token)
   - Improved language acceptability in some configurations

### Weaknesses
1. **Limited Scale**: While the approach is tested on GPT2 and Gemma, there's no evaluation on larger, more current models. This raises questions about scalability.

2. **Parameter Sensitivity**: Though the authors claim their approach doesn't require parameter tuning, the results show significant variation across different SAE configurations and layers. More analysis of these dependencies would be valuable.

3. **Baseline Comparisons**: While fine-tuning is used as a baseline, comparison with other lightweight adaptation methods (like prompt tuning or LoRA) would strengthen the evaluation.

4. **Theoretical Foundation**: The paper could benefit from stronger theoretical justification for why the proposed scoring mechanism effectively identifies relevant neurons.

### Questions
1. How does the approach scale to larger models? Have you tested or analyzed computational requirements for models like LLaMA or GPT-3?

2. The results show significant variation in performance across different layers. Could you provide more insight into how to select the optimal layer for applying the SAE modifications?

3. How robust is the approach to different types of alignment tasks? While medical domain alignment shows promising results, are there certain types of alignment that are particularly challenging?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes using Sparse Autoencoders (SAEs) for topic alignment in LLMs to achieve efficient and interpretable topic alignment by scoring SAE neurons based on their semantic similarity to specific alignment topics. Authors demonstrate the effectiveness of the proposed method across datasets (e.g., medical and Amazon reviews) and models like GPT-2 and Gemma to show the advantage of SAEs in reducing training times and improving interpretability. The authors also introduce contamination metrics to quantify topic alignment uncertainty.

### Strengths
1. The paper introduces an interesting approach by applying SAEs to align LLMs with specific topics, which is relatively unexplored. Compared to FT, this approach is more interpretable and efficient.

2. This paper evaluates the proposed method across several datasets and models, showing the effectiveness and generalizability of the results. Further analysis, including the alignment score, SAE neurons distribution, and model generation output, covers various aspects of alignment performance.

3. The contamination metric is simple to evaluate the uncertainty in alignment.

### Weaknesses
1. The experiments are conducted on GPT2 and Gemma. It is unclear if the method is easy to use on other model families or larger models e.g. LLAMA 405B. It seems that the configuration would be hard to obtain on larger models.
2.  It seems that the configuration of SAE is varied on different topics.  Further investigating the impact of different configurations may be valuable.

### Questions
1. Does the proposed score/method evaluate the impact on the polysemantic neurons?
2. Is this method easy to scale with much larger LLMs with billions of parameters?
3. The contamination metric is used to quantify alignment uncertainty. Is there any human evaluation other judgment to show the reliably of this metric?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces an approach for achieving topic alignment in LLMs using Sparse Autoencoders (SAEs), offering an alternative to computationally intensive fine-tuning methods. The key contribution is a technique that scores SAE neurons based on their semantic similarity to desired alignment topics and then modifies layer outputs by emphasizing high-scoring neurons.

The methodology consists of two main components: 1) A scoring mechanism that evaluates SAE neurons' relevance to target topics using semantic similarity and a reference prompt set, and 2) A "swap" approach that modifies SAE outputs by emphasizing aligned neurons in a context-sensitive way. This enables topic alignment without extensive parameter tuning while maintaining interpretability through the SAE structure.

### Strengths
- The paper presents a new application of SAEs for topic alignment that addresses key limitations of existing approaches like fine-tuning in the topic alignment research area. The scoring and modification methods are generally well-motivated.
- The experimental methodology covers multiple LLMs, topics, and SAE configurations. The evaluation metrics cover both performance (perplexity, linguistic acceptability) and efficiency aspects.

### Weaknesses
My main concern is the lack of qualitative performance comparisons with conventional topic alignment methods, as indicated in the first paragraph of your introduction -- the proposed SAE-based method has advantages in computational efficiency, interpretability, etc. You should compare them. Additionally, the current organization of the paper is not self-contained; for instance, ablation studies should be rearranged to the main text. Regarding the motivation, justifications for evaluation metrics and methods in lines 200-206 (e.g., why these three aspects (aligned, polysemantic, unaligned) of neurons) should be clearly presented.

### Questions
See weaknesses. I suggest including more baseline results in the rebuttal phase.

### Soundness
3

### Presentation
2

### Contribution
2
