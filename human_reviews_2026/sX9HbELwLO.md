# Attention Smoothing Is All You Need For Unlearning

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4

## Abstract
Large Language Models are prone to memorizing sensitive, copyrighted, or hazardous content, posing significant privacy and legal concerns. Retraining from scratch is computationally infeasible, whereas current unlearning methods exhibit unstable trade-offs between forgetting and utility, frequently producing incoherent outputs on forget prompts and failing to generalize due to the persistence of lexical-level and semantic-level associations in attention. We propose Attention Smoothing Unlearning (ASU), a principled framework that casts unlearning as self-distillation from a forget-teacher derived from the model’s own attention. By increasing the softmax temperature, ASU flattens attention distributions and directly suppresses the lexical-level and semantic-level associations responsible for reconstructing memorized knowledge. This results in a bounded optimization objective that erases factual information yet maintains coherence in responses to forget prompts. Empirical evaluation on TOFU, MUSE, and WMDP, along with real-world and continual unlearning scenarios across question answering and text completion, demonstrates that ASU outperforms the baselines for most of the unlearning scenarios, delivering robust unlearning with minimal loss of model utility.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Existing large language model (LLM) forgetting methods suffer from an unstable trade-off between forgetting effectiveness and model utility. Some methods forget too much, resulting in incoherent output, while others forget insufficiently, still leaking factual knowledge. This paper proposes a new forgetting mechanism, Attention Smoothing Unlearning (ASU), which reformulates the LLM forgetting task as a self-distillation process. This method introduces temperature smoothing in the model's attention layer to make the model's attention more evenly distributed, thereby weakening the model's focused memory of specific tokens and semantic associations, achieving a natural and stable forgetting behavior.

### Strengths
Problem significance: The paper targets the selective unlearning of content in LLMs. It is an urgent privacy and safety problem for retraining is sometimes infeasible. Existing methods have problems such as excessive forgetting leading to meaningless output, or insufficient forgetting still leaking sensitive data.
Conceptual novelty: Using attention temperature smoothing to diffuse factual focus is novel in the unlearning domain.
Methodological clarity with comprehensive experiments: The formulation as a self-distillation process is well-motivated and technically sound. ASU constructs an in-model “forgetting teacher” via attention temperature and uses KL distillation to suppress factual recall while preserving syntactic fluency. It requires no external teacher or additional parameters, making it easy to reproduce and deploy on open-source models.
Broad and careful evaluation: The paper tests across multiple scenarios, models, and metrics. Objectives and losses are clearly specified; scenario coverage is broad; the appendix documents temperature selection and metric definitions.

### Weaknesses
Limited theoretical justification: The link between attention entropy and factual forgetting is primarily empirical. A stronger theoretical or representational analysis would be essential.
Evaluation depth: Results lack significance tests or confidence intervals, and there is no qualitative human evaluation of “naturalness” claims.
Potential annotation bias: The analysis of entropy change relies on GPT-based classification of “factual” versus “functional” tokens. The single-model annotation may introduce bias due to GPT’s own preferences or inconsistencies.
Wording consistency: To align with common usage, replace “Question and Answer (QA)” with “Question Answering (QA)” and keep this consistent.

### Questions
How sensitive is ASU to the temperature parameter τ?
Can you quantify the computational efficiency of AS?
After deployment, does forgetting persist under benign continued training or repeated exposure to the forgotten content? Have you evaluated ASU’s robustness against relearning or adversarial triggering?
How resistant is ASU to jailbreak-style prompts?

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
This paper proposes a computationally simple and robust unlearning framework, ASU. This framework first constructs a naturalistic forgetting teacher model by increasing the softmax temperature in the self-attention mechanism and then fine-tunes the student model to imitate the teacher on the forget set through knowledge distillation. Extensive experiments demonstrate that the proposed method outperforms existing baseline methods in both real-world and continual unlearning scenarios.

### Strengths
- The proposed ASU framework requires minimal architectural changes, making it highly practical for large-scale unlearning.
- The methodology is clearly described and the paper is easy to follow.

### Weaknesses
- As mentioned in this paper, the teacher model is constructed by applying attention smoothing, i.e., increasing the softmax temperature in the self-attention mechanism. Will this operation hurts the performance of the teacher model.
- Since this article requires a student model and a teacher model, the computational cost of forgetting should also be used as an indicator to evaluate the effect of each method.
- Can you provide a detailed proof that applying attention smoothing can make the teacher model naturally forget?
- Lack of experimental details and no mention of the model used in the paper, only mentioning Llama-2-Chat-7B.
- The effectiveness of the proposed method has not been verified on more and larger model structures.
- The selection of temperature parameters lacks theoretical guidance.

### Questions
How sensitive is ASU to the temperature parameter across different datasets and model scales?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Attention Smoothing Unlearning (ASU), a simple framework that removes memorized knowledge in large language models by flattening attention distributions through temperature scaling and self-distilling from the smoothed teacher model. ASU effectively erases factual associations while maintaining coherent and useful language behavior, outperforming existing unlearning methods across benchmarks such as TOFU, MUSE, and WMDP. This approach achieves a superior trade-off between forget efficacy and model utility, offering a practical and stable solution for real-world unlearning scenarios.

### Strengths
1. The classification of unlearning methods into two categories (e.g., divergence and convergence) provides an interesting conceptual framework that effectively supports the central idea of the paper.

2. The paper conducts extensive experiments on multiple benchmarks, including TOFU, MUSE, and WMDP, demonstrating the robustness of the proposed approach.

3. The paper is well organized and clearly written, making the overall argument easy to follow and the experimental results easy to interpret.

### Weaknesses
1. Attention Smoothing Unlearning (ASU) can be interpreted through the lens of both divergence and convergence. Specifically, emphasizing knowledge distillation aligns with convergence toward the teacher model, whereas smoothing attention may introduce divergence by disrupting previously salient attention patterns. Therefore, it remains ambiguous whether ASU should be categorized under either paradigm or considered as an independent mechanism beyond both.

2. The rationale behind how ASU mitigates the “gibberish outputs” problem is not clearly explained. Since the attention mechanism underlies linguistic understanding (e.g., grammar and syntax), smoothing attention scores may inadvertently weaken the model’s language comprehension, leading to nonsensical outputs. As shown in Figure 2, the negative log likelihood on functional knowledge does not increase as sharply as that of factual knowledge, yet it still rises beyond the default baseline.

3. The experiments are conducted on only one LLM per evaluation setting, which raises concerns about the generalizability of the proposed method across different model architectures and scales.

4. More in-depth analysis is required to substantiate the main claim. (e.g., attention score distribution after distillation, layer-wise attention smoothing)

5. It would be more practical to directly adjust the attention of the student model by treating functional and factual tokens separately, which could offer a more effective framework.

### Questions
See Weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Attention Smoothing Unlearning (ASU), a method for removing specific knowledge from large language models (LLMs) without degrading overall performance. Instead of pushing the model away from forget data through divergence-based or entropy-maximization objectives, ASU constructs a teacher model by smoothing the attention distribution. The student model then distills knowledge from this teacher model on the forget set, effectively erasing targeted information while preserving general utility. Experiments show that ASU outperforms baseline methods across both QA and text-completion tasks, achieving stable forgetting with minimal loss of model quality.

### Strengths
- The paper provides a conceptually simple yet effective formulation for unlearning based on attention smoothing.
- The study includes comprehensive evaluations across QA and free-form completion settings, as well as scenario-based real-world tests.
- The experiments demonstrate that ASU consistently outperforms existing unlearning baselines.

### Weaknesses
- The work is heavily oriented toward experimental performance, and it lacks deeper analytical insight. The paper would benefit from additional analysis explaining why attention smoothing leads to differential effects on factual versus functional tokens.

- The paper does not provide formal guidance for selecting the optimal attention temperature parameter, leaving it heuristic.

### Questions
- When applying attention smoothing, why does entropy increase similarly for both factual and functional tokens, yet negative log-likelihood diverges between them, and how should this discrepancy be interpreted?

- Could the authors expand on the mechanistic explanation behind why factual information becomes more suppressed than functional information under attention smoothing, and analyze the stability of the teacher signal produced by smoothed attention? (for example, evaluating the teacher model on TOFU)

- Upon examining Table 10, it appears that the ASU-unlearned model exhibits hallucination-like behavior. This raises a concern that increasing entropy selectively for factual tokens may induce hallucinated responses, rather than safe refusal behavior. Could this mechanism therefore pose a greater risk than approaches that explicitly encourage the model to decline answering?

### Soundness
2

### Presentation
2

### Contribution
2
