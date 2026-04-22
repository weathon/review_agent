# Modulating LLM Behavior via Context-Specific Activation Steering

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 8, 2

## Abstract
LLMs achieve strong capabilities, yet precisely steering their responses with ever-shifting safety requirements remains unresolved.
Current activation engineering methods embed a static premise — prompt categories elicit distinct activation patterns — and coerce each input into a hand-crafted semantic category.
This premise fails when adversarial prompt variations (e.g. jailbreaks) perturb the activations, yielding collateral suppression or undetected risks.
We contend that the activation steering should be determined on-the-fly by the input itself within the semantic space, rather than being predetermined by rigid, hand-crafted categories.
In this paper, we propose Context-Specific Steering (COS-Steering), which maps the full safety-steering activation subspace and lets inputs locate its own steering coordinately.
COS-Steering recover this subspace by compressing a pool of steering signals into a compact set of basis vectors via SAE.
A lightweight module then reads the input activation and outputs weights these basis vectors for context-specific steering.
To evaluate robustness against distribution shift, we test COS-Steering in a mixed-attack setting, which combines multiple attack methods, across datasets and models.
Comparing to baselines, COS-Steering preserves strong refusal on harmful prompts while introducing negligible side-effects on benign queries.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Context-Specific Steering (COS-Steering), which replaces static, hand-crafted activation categories with dynamic steering. The method compresses a pool of steering signals into basis vectors via sparse autoencoders and learns input-specific weights for context-aware control. Experiments show improved robustness to adversarial prompts while maintaining low impact on benign queries.

### Strengths
- The paper introduces a context-specific steering mechanism that adapts dynamically to each input, addressing limitations of static, hand-crafted activation categories.
- The proposed method demonstrates strong effectiveness in alignment tasks, showing improved robustness against adversarial prompts.

### Weaknesses
**Major concern:**

The claim (Line 12) that “current activation engineering methods embed a static premise” is not entirely accurate, as important related work is missing. In particular, PaCE (NeurIPS 2024) introduces activation steering on-the-fly via sparse coding, which directly challenges this claim. 
- It would be helpful to clarify what specific challenges remain unresolved by PaCE that your method addresses.
- In addition, please elaborate on why your proposed approach remains necessary in light of PaCE.
- Strengthening the motivation section by positioning your contribution more clearly relative to PaCE would further highlight the novelty and significance of your work.

*Ref: Luo, J., Ding, T., Chan, K. H. R., Thaker, D., Chattopadhyay, A., Callison-Burch, C., & Vidal, R. (2024). Pace: Parsimonious concept engineering for large language models. Advances in Neural Information Processing Systems, 37, 99347-99381.*

**Other concerns:**
- Line 196: Could you clarify the criterion used to divide the bad cases dataset?
- Line 251: The Orthogonal Representation Penalty $P_{OR}$ appears to be constant with respect to the optimization variable $S_{r_l}$​. Could you please elaborate on how this term influences the optimization process?
- In Table 1, evaluation on benign prompts is missing. In particular, it would be valuable to report metrics such as BERTScore and KL divergence when applying your method to benign inputs. This would help assess whether the proposed approach introduces unintended distortions or shifts in non-adversarial settings.
- As the experiments are limited to Llama-2 and Vicuna, I am wondering how the method would perform on more recent LLMs such as Llama-3-8B. Including results or discussion on newer models would strengthen the empirical evaluation and provide a clearer picture of the method’s generality.
- Most experiments do not report statistical uncertainty. Adding standard deviations or confidence intervals would make the results more reliable.

**Minor comment:**
- Line 425: The link to the Appendix appears to be broken.

### Questions
Please see **Weaknesses** for my questions.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper studies how to guide large language models (LLMs) to refuse harmful prompts without blocking benign ones. Earlier activation‑steering techniques assume that each input can be mapped into a fixed semantic category and apply a single pre‑computed direction to the hidden activations of the model. The authors observe that adversarial jailbreak prompts can shift activations away from these hand‑crafted categories, leading to missed detections or excessive refusals. They propose **Context‑Specific Steering (COS‑Steering)**. The approach collects many safety‑related steering vectors and compresses them into a sparse basis using a sparse autoencoder. At inference time, a small controller reads the current input’s activation and outputs coefficients over the basis, producing a steering vector tailored to that input. The authors test the method on a mixed‑attack scenario that combines multiple jailbreak attacks. COS‑Steering maintains high refusal rates on harmful prompts and introduces little degradation on harmless queries compared with baseline steering methods.

### Strengths
- **Originality**. COS‑Steering replaces fixed, category‑based steering with a dynamic mechanism that weights multiple safety directions according to the input. This addresses a known limitation of activation steering and provides a new way to integrate sparse features into steering.
    
- **Method design**. Using a sparse autoencoder to derive basis vectors captures diverse steering signals while keeping the controller small. The lightweight module that reads activations and outputs weights allows context‑dependent interventions with minimal overhead.

### Weaknesses
-   **Missing LLM usage and reproducibility statements**. These are strongly recommended for ICLR submissions. Missing these key statements makes it difficult to evaluate the reliability of the work. While there are novel contributions in this paper, the authors should at least clarify details for creating the work before I raise my score.
-   **KL divergence is not sufficient to validate consistent output quality**. In the experiments, ASR and Harmful Score are used to evaluate safety, while BERTScore and KL divergence are used to measure performance retention. However, BERTScore and KL divergence are not very robust metrics for evaluating output changes and are insensitive to generation quality or over-refusals. The paper could report results on perplexity changes and over-refusal rates to further confirm the selectivity for safety responses.
-   **Lack of clear evidence on Sr choosing decoder vectors selectively**. While the paper's core motivation is that Sr will activate Sd vectors based on context, we need direct evidence of this. The negligible side effects on benign queries could be explained by the fact that the paper only trains a small adapter on the model to refuse harmful queries, similar to LoRA.
-   **There are some syntax errors in the paper**. For example, a wrong quote mark direction in Line 199 and a missing reference (Appendix ??) in Line 425.

### Questions
-   In Line 413: how is the "top-10% activation range" defined?
-   For Figure 5, why are there multiple bars for one layer? Are they different heads in that layer?
-   What if we don't train Sr and Sd separately, but train the whole adapter in one training session? What is the fundamental difference between COS-Steering and a LoRA adapter?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes COS-Steering, a dynamic method that allows inputs to determine their own safety-steering direction within the semantic space. Experiments show it maintains strong refusal of harmful prompts while causing minimal side effects on benign queries.

### Strengths
Solid Theory and Motivations
- The paper clearly states the weakness of the previous method. 
- provides solid observations of `Activation shifting driven by jailbreak attack` in the vision of LLM internal activation state.

Comprehensive Evaluations and In-depth Analysis
- The paper not only proves the effectiveness of the proposed method, but also provides non-trivial insights in its working mechanism.

Good Demonstrations and writing

### Weaknesses
There are minor weaknesses. Figure 2 should have a clearer resolution. Line 425 has a reference link failure.

### Questions
As the experiments only include the Llama2 series and the Vicuna series, why the latest version (at least one latest series) are not included, e.g., Llama3 series and Qwen3?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes COS-Steering, a context-specific activation steering method that learns a safe-steering subspace and applies input-conditioned linear combinations of basis vectors to steer an LLM’s activations only when needed. Concretely: (1) it trains multiple steering vectors on small shards of harmful prompts to elicit a harmless prefix; (2) compresses these steering vectors with a sparse autoencoder (SAE) to obtain a compact set of basis vectors; and (3) learns a tiny Steering Representor that maps current hidden states to weights over those bases, with an Orthogonal Representation Penalty.

### Strengths
1. The method moves beyond static steering by conditioning steering directions on input context, addressing attack diversity.
2. The design explicitly creates distribution shift by mixing attacks across datasets.

### Weaknesses
1. Steering vectors are trained to produce a specific harmless prefix. While the paper discusses why this can flip harmfulness effectively, it’s still a narrow textual proxy for safety and may overfit to that phrasing/distribution; consequences for content-level safety beyond prefix control deserve deeper auditing.

2. The SAE is used for compression rather than monosemantic discovery; however, the semantics of the learned bases are only lightly analyzed. More direct probes would strengthen interpretability claims.

3. We see hyper-parameter settings and a note that 2 epochs are “prudent,” but key ablations are missing: (i) How sensitive are results to the SAE bottleneck size? (ii) What if we remove POR or vary its strength? (iii) How many steering vectors/shards are needed before diminishing returns? (iv) Layer placement sensitivity beyond 9–30?

4. Reliance on Llama-Guard2 and GPT-judge templates can create evaluation circularity. Cross-checking with alternative safety classifiers would reduce metric bias.

5. Experiments are conducted only on Vicuna-7B and LLaMA-2-7B, which are now relatively dated models. Given the rapid evolution of open-weight LLMs, results may not generalize to newer architectures with different safety and activation structures. 

6. The motivation of claiming that static, category-based steering fails under adversarial prompt variations is insufficiently substantiated. The paper does not clearly demonstrate why or how these category boundaries collapse under jailbreak transformations. More in-depth analysis or explanation of activation drift would make the premise more convincing.

7. Only BERTScore and KL metrics are reported. Broader benign performance (e.g., instruction-following, creativity) should be measured.

### Questions
1. How does COS-Steering perform against paraphrased or long-context adversarial prompts that delay harmful intent?

2. What is the minimal number of steering vectors or shards required before performance saturates?

### Soundness
2

### Presentation
2

### Contribution
3
