# Beyond Neural Incompatibility: Easing Cross-Scale Knowledge Transfer in Large Language Models through Latent Semantic Alignment

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4

## Abstract
Large Language Models (LLMs) encode vast amounts of knowledge in their massive parameters, which is accessible to locate, trace, and analyze.
Despite advances in neural interpretability, it is still not clear how to transfer knowledge in a fine-grained manner, namely parametric knowledge transfer (PKT).
A key problem is enabling effective and efficient knowledge transfer across LLMs of different scales, which is essential for achieving greater flexibility and broader applicability in transferring knowledge between LLMs.
Due to neural incompatibility, referring to the architectural and parametric differences between LLMs of varying scales, existing methods that directly reuse layer parameters are severely limited.
In this paper, we identify the semantic alignment in latent space as the fundamental prerequisite for LLM cross-scale knowledge transfer.
Instead of directly using the layer parameters, our approach takes activations as the medium of layer-wise knowledge transfer.
Leveraging the semantics in latent space, our approach is simple and outperforms prior work, better aligning model behaviors across varying scales.
Evaluations on four benchmarks demonstrate the efficacy of our method. Further analysis reveals the key factors easing cross-scale knowledge transfer and provides insights into the nature of latent semantic alignment.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The study investigates knowledge transfer, specifically knowledge transfer between architecturally different LLMs. It proposes SemAlign, a layer-wise and semantics-aware activation distillation approach. It decomposes the teachers latent representations into semantic components. The authors claim that aggregation on a semantic aggregation level provides a better supervisory signal to the student. The paper conducts experiments using llama 2 7b as a student and different 13b models as teachers. The distillation performance is evaluated on mmlu, gsm8k, humaneval, and mbpp.

### Strengths
- The problem of architecture-agnostic knowledge distillation is relevant and interesting. 
- The presented performance results are promising and warrant further research in semantics-aware activation distillation.

### Weaknesses
- The paper lacks innovation and novelty. The main contribution of the paper centers on layer-wise semantics-aware distillation in settings where the student differs architecturally from the teacher. However, the semantics decomposition is introduced by Gu et al. 2024. Moreover, the architectural differences are simply resolved by mapping all activations from layers exceeding the depth in the student into the last layer. This is lacking experiments and theoretical grounding.
- Performance benefits, presented in Table 1, are unclear in contrast to other methods 
- Missing baselines: The method is not compared to regular layer-wise distillation.
- Layer-wise distillation has been introduced and applied in various other studies, including Liang et al. 2022.
- The presentation in the paper is heavily inspired by Gu et al. 2024

References:
- Liang, C., Zuo, S., Zhang, Q., He, P., Chen, W., & Zhao, T. (2022). Less is More: Task-aware Layer-wise Distillation for Language Model Compression. arXiv Preprint arXiv:2210. 01351.
- Gu, J., Aleti, A., Chen, C., & Zhang, H. (2024). A Semantic-Aware Layer-Freezing Approach to Computation-Efficient Fine-Tuning of Language Models. arXiv preprint arXiv:2406.11753.

### Questions
- How does you method differ from layer-wise distillation, used in i.e. Liang et al. 2022?
- How does you semantic-aware decomposition differ from Gu et al. 2024?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the challenge of transferring knowledge across LLMs of different scales. It argues that existing methods, which focus on transferring model parameters, are limited by "neural incompatibility" arising from architectural and parametric differences. The authors propose a new method, SemAlign, which instead uses layer outputs, or activations, as the medium for transfer. The method's core mechanism involves identifying and pairing critical layers, then using a "semantic basis" derived from the vocabulary to align the teacher's latent activations with the student's latent space. This aligned representation is then used as a supervisory signal to steer the student model's behavior.

Experimentally, the authors evaluate SemAlign by transferring knowledge from Llama 2 13B models to a 7B student on four benchmarks. The results show that SemAlign outperforms parameter-space baselines like Seeking and LaTen on tasks such as MMLU and HumanEval. The paper also demonstrates strong performance when transferring from specialized teacher models on coding tasks.

### Strengths
1. The paper proposes a new semantics-first perspective on parametric knowledge transfer (PKT), which targets the "neural incompatibility" bottleneck by using latent activations instead of raw parameters.

2. The proposed SemAlign method achieves empirical gains over existing PKT baselines on certain benchmarks, such as MMLU and HumanEval. The paper also reports advantages when transferring from specialized, code-focused teacher models.

3. The paper includes an analysis using CKA (Figure 4) to visualize the similarity of layer outputs across different model scales.

### Weaknesses
1. **Weak Empirical Validation for the Core "Semantic Basis" Mechanism.** The paper's central claim is on the superiority of transferring knowledge via "semantic alignment" rather than direct parameter manipulation. However, the empirical evidence provided specifically for this mechanism is thin. The concept of "Vocabulary-Defined Semantics" is adopted from prior work, and its validation within this paper is limited to a single experiment in Figure 2. This experiment, which validates the "resolution of semantics" hypothesis, is conducted on a Qwen3 model, while all main performance experiments are conducted on the Llama 2 family. There is no direct evidence or ablation to confirm that this specific decomposition/recomposition process is truly superior to other forms of representation steering, or why it works robustly on the LLMs.

2. **Narrow Experimental Scope.** All main experiments are confined to a single model family (Llama-2 7B/13B) plus two code-specialized variants as teachers, leaving generality to other families untested. While Table 1 shows "Seeking" notably overshoots the 13B teacher on GSM8K and MBPP, the finetuned-teacher transfer results in Table 2 report only the coding benchmarks (HumanEval/MBPP), excluding MMLU and GSM8K, and thereby sidestepping these patterns on knowledge/reasoning tasks. This limited coverage weakens the claim of broad transfer and makes the comparative story less convincing.

3. **Insufficient Analysis.** The proposed pipeline is complex, involving specific choices for layer attribution, layer pairing, semantic alignment, and a joint optimization objective. The paper provides almost no analysis to disentangle the contributions of these individual components. While ablations are briefly mentioned in the Introduction, the results are not presented, making it impossible to understand the underlying factors driving performance. Furthermore, the single analysis experiment in Section 5.3 (Figure 4) is disconnected from the main experiments (Qwen 3 vs. Llama 2) and fails to provide a comparative visualization for the baseline methods. Without comparison with Seeking or LaTen, we cannot visually confirm the authors' central hypothesis that the proposed framework better resolves neural incompatibility than prior work.

### Questions
None

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper presents an approach to parametric knowledge transfer between models of the same family that share the same vocabulary. This is achieved by introducing an additional loss term that encourages mapping paired layer outputs encoded into a shared basis between the student and the teacher models. 

The method is demonstrated on general and specialized models from the Llama 2 family with four tasks, showing better faithfulness to the teacher performance and outperforming a competing method in some of the tasks.

### Strengths
The approach is simpler than previous work focusing on layer outputs rather than weights or logits

### Weaknesses
- The paper has a narrow literature focus comparing against two prior works only

- The extracted shared basis between the teacher and student models is overcomplete so the "resultant semantic" equation is not correct

- The models have to share the same vocabulary, limiting the applicability of this method

### Questions
What do the authors mean with "input-side semantic basis"?

Figure 2 caption says the model is Llama 2 while the text says Qwen 3

- The layer pairing feels arbitrary. Any ablations?

- When would supervision require "a target at an exact student depth "?

- If the goal of PKT is to achieve performance gains in a student model, why faithfulness to the teacher matters if better performance is attainable, e.g. through SEEKING?

- In the presented experiments, the models used share the same hidden size. How does this method generalize when the teacher-student model differ in their latent expressivity?

### Soundness
2

### Presentation
2

### Contribution
2
