# Revisiting LLM Reasoning via Information Bottleneck

- Decision: Reject
- Scores: 6, 4, 4, 4, 4

## Abstract
Large language models (LLMs) have recently demonstrated remarkable progress in reasoning capabilities through reinforcement learning with verifiable rewards (RLVR). By leveraging simple rule-based rewards, RL effectively incentivizes LLMs to produce extended chain-of-thought (CoT) reasoning trajectories, progressively guiding them toward correct answers. However, existing approaches remain largely heuristic and intuition-driven, limiting the development of principled methodologies. In this paper, we present a theoretical characterization of LLM reasoning grounded in information bottleneck (IB) principle, introducing \textit{IB-aware reasoning optimization} (IBRO), a framework that encourages reasoning trajectories to be both \textit{informative} about the final correct answer and \textit{generalizable} across diverse prompts. We derive a practical token-level surrogate objective and propose an efficient approximation, resulting in the lightweight \textit{IB regularization} method. This technique integrates seamlessly into existing RL-based post-training frameworks without additional computational overhead, requiring only a one-line code modification. Empirically, we validate IB regularization across multiple mathematical reasoning benchmarks and RL algorithms, demonstrating consistent improvements in LLM reasoning performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a theoretical characterization of large language model (LLM) reasoning grounded in the Information Bottleneck principle, introducing IB-aware Reasoning Optimization (IBRO), a framework that encourages reasoning trajectories to be both informative about the correct answer and generalizable across diverse prompts. This technique integrates seamlessly into existing RL-based post-training frameworks with negligible computational overhead. The approach demonstrates consistent performance improvements across multiple benchmarks.

### Strengths
- This paper establishes a theoretical framework for LLM reasoning, grounded in the Information Bottleneck principle.

- The framework is designed to maximize the informativeness of reasoning trajectories regarding the final answer while ensuring they remain generalizable across diverse prompts.

### Weaknesses
- It is recommended that the notation in Theorem 2 be carefully reviewed. The current use of $I(r|q,a)$  in the definition of $\mathcal{L}_IB$ is inconsistent with the paper's own context, where conditional entropy $H(r|q,a)$ is intended. Correcting this is essential for the theoretical rigor of the presentation.
- There are also some works that apply information theory to LLM [A-D]. It is recommended to supplement the relevant discussions and explain the differences with them.

[A] Understanding chain-of-thought in llms through information theory. [B] Learning to think: Information-theoretic reinforcement fine-tuning for llms. [C] EVINCE: Optimizing Multi-LLM Dialogues Using Conditional Statistics and Information Theory. [D] Exploring Information Processing in Large Language Models: Insights from Information Bottleneck Theory.

- The approximation $\lambda_t \approx -A_t$ is a critical yet under-justified step. The authors should better explain why the negative advantage can replace the ground-truth-dependent $\lambda_t$. While intuitively plausible (if a token is important, its uncertainty when knowing the answer should be low), this claim lacks both theoretical justification and empirical support (e.g., showing negative correlation between $A_t$ and $H(o_t | ..., a)$). We recommend adding a paragraph to clarify this approximation's rationale or providing a small validation experiment.

### Questions
- The paper seems to implicitly assume that the reasoning process is a monotonic progression toward the correct answer. However, in practice, reasoning can be non-monotonic, involving exploration, temporary detours, or even incorrect sub-goals. Could the authors clarify the scope of their framework in handling such non-ideal, yet common, reasoning trajectories? A discussion on how the token-level advantage $A_t$ captures and manages these dynamics would strengthen the theoretical grounding of the work.

- The authors could discuss whether their method enhances interpretability. For example, examining the alignment between high-advantage tokens and key reasoning steps would be valuable.
- Please supplement the relevant discussions about the differences and connection with prior  information theory-based works.

- Please evaluate the performance of the proposed method against a broader range of existing entropy regularization strategies to better validate its advantages.

- The authors emphasize the "negligible" overhead of IB regularization but provide no specific profiling data. Given the substantial cost of large-scale RL training, could the authors present the actual wall-clock time or FLOPs overhead per training step (or throughout training) introduced by IB regularization, quantified as a percentage compared to the no-regulation baseline? This would provide a critical data point for practitioners considering the adoption of this method.

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
The paper introduces Information Bottleneck–aware Reasoning Optimization (IBRO), an information-theoretic framework for improving LLM reasoning. It derives a token-level surrogate objective and proposes IB regularization, which adjusts token entropy based on token advantages. The method integrates easily into PPO and DAPO with minimal code change and yields small but consistent improvements on AMC23, AIME24, and AIME25 benchmarks, along with more stable entropy dynamics.

### Strengths
- IB regularization is elegant, lightweight for Reasoning RL
- Offers a clear theoretical perspective on LLM reasoning through the information bottleneck
- Includes a generalization bound that connects theory and empirical outcomes

### Weaknesses
- The improvements are relatively small (around +2 points) and may not be statistically significant. Figure 1 also suggests that the choice of training step or checkpoint may play a more substantial role in performance variation than the proposed regularization
- Experiments limited to one model (Qwen2.5-7B) and math reasoning tasks

### Questions
- How well does IB regularization generalize to non-math reasoning tasks?
- Hot to empirically confirm that token advantage aligns with informational importance?
- How robust is the method to different β and α settings?
- What are the expected effects when scaling to larger LLMs? Scaling?

### Soundness
3

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
3

### Summary
The paper applies the Information Bottleneck principle to design a lightweight regularization for RL with verifiable rewards (RLVR). The method is computationally cheap and easy to integrate (one-line change in the loss function). Experiments indicate that the proposed regularization term improves LLM reasoning performance compared to training without regularization and to standard entropy regularization on mathematical benchmarks (AMC-23, AIME-24, AIME-25).

### Strengths
* The core idea is well motivated and intuitive.

* The modification adds no extra computation and is extremely simple to implement (one-line change), which is important for practical adoption since many new methods are hard to implement and deploy.

* The work provides a generalization bound with proofs for the proposed IBRO loss (Theorem 2).

### Weaknesses
* The experimental scope is quite limited. The regularization seems to be broadly applicable beyond mathematics, yet evaluation is restricted to three very similar math benchmarks. Broader testing on more diverse reasoning tasks (code, common-sense, logic, etc.) would better evaluate the improvement from the proposed regularization.

* The link between the IB-based theory and actual regularization term is unclear. It is not evident how the specific regularization term (lines 266–267, and Listing 1) follows from the surrogate IBRO objective (Lines 205-207). The rationale for replacing the modulation coefficient $\lambda_t$ with the advantage $A_t$ is hand-wavy, non-obvious (Lines 263-269). Additionaly, Assumption 1 (lines 197–199) is also questionable: it treats policy entropy as constant during RL training and is used in Theorem 1, yet Figure 2 shows substantial entropy changes under PPO and DAPO relative to initial values. 

* Mathematical presentation and notation need refinement.  For example, on line 127 the PPO objective uses $r_t$ as the importance sampling ratio, while later $r$ denotes a random variable for the reasoning trajectory. Theorem 1 (lines 203–207) is stated without a proof. Although it seems to follow directly from the formula on line 200 and Assumption 1 (line 197), a written proof in appendix would be preferable. Equations in the main text are not numbered, which complicates referencing.

### Questions
4. **Questions**

* If Assumption 1 does not hold and the entropy of reasoning trajectories changes during training, as suggested by Figure 2, how can the success of the proposed regularization in Listing 1 be explained under the IB principle?
* The motivation on lines 268–269 that the term $A_t H_t$ encourages higher entropy for critical tokens and lower entropy for tokens with small or negative advantage seems counterintuitive. Shouldn’t critical reasoning tokens be generated with lower randomness, while stylistic tokens can vary more? Moreover, this rationale appears only loosely connected to the IBRO surrogate loss $\ell^t_{\text{IB}}$ on line 258. A low $\ell^t_{\text{IB}}$ for critical tokens does not by itself imply higher $H(o_t \mid q, o_{<t})$, since $\ell^t_{\text{IB}}$ also depends on $H(o_t \mid q, a, o_{<t})$, which could be the main driver of the low surrogate loss for such tokens.

### Soundness
3

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
This paper introduces information bottleneck (IB)-aware reasoning optimization (IBRO), aiming to maximize informativeness of reasoning trajectories with respect to the correct answer while minimizing reliance on prompt-specific details. The method derives a token-level surrogate for the IB objective and provide an efficient advantage-weighted entropy regularization method that can be incorporated into standard RL-based LLM post-training pipelines with minimal modification. Empirical evaluation across several mathematical reasoning benchmarks (AMC23, AIME24, AIME25) and RL algorithms (PPO, DAPO) demonstrates improvements over baseline and naive entropy regularization.

### Strengths
The manuscript is clearly written, explores both theoretical underpinnings and practical implications.
The method is highly efficient and practical: It requires negligible computational overhead and can be implemented with just one line of code modification.

### Weaknesses
1.The experiments in this paper are insufficiently comprehensive, having been conducted only on QWEN 2.5-7B. The universality of the method needs to be validated across different series of models.

2. Lack analysis of the hyperparameter α. 

3.Assumption 1 (Page 4) that $\pi(\boldsymbol{r})$ remains invariant during post-training is asserted but not supported by empirical analysis or theoretical guarantee. In practice, LLM output distributions may shift as post-training progresses, potentially weakening the theoretical foundation. Can you give evidence (e.g., ablation or empirical distributional drift analysis) that this assumption holds?
I am happy to increase my score if these issues can be addressed.

### Questions
Generally, the response format is <think> </think> <answer> </answer>, with its length primarily determined by the reasoning process. If the length decreases due to an increase in the [EOS] probability, does this imply that the model did not output the final answer but instead directly output [EOS] during the reasoning process?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes IB-aware Reasoning Optimization (IBRO), grounding LLM reasoning in the information bottleneck (IB) principle, and derives a practical IB regularization term that multiplies token entropy by token advantages and adds it to standard RL objectives. The method is claimed to be a one-line change, with pseudocode shown, and is evaluated on AMC23/AIME24/AIME25 under PPO and DAPO, yielding small but consistent avg@32 gains and more stable entropy dynamics/length profiles.

### Strengths
[Originality]
A crisp, unifying viewpoint: formalizing “good CoT” as minimizing dependence on prompt-specific details while maximizing informativeness about the correct answer (IBRO) and turning this into a token-level surrogate and advantage-weighted entropy regularizer. 

[Quality]
Clear derivation from IBRO to a token-level objective and a practical approximation that requires only quantities already computed by PPO/GRPO-style RL (token entropy and advantages).
The “one-line change” is explicit (Listing 1). Results show consistent though modest improvements in avg@32 across three math benchmarks and two RL settings, plus interpretable plots for entropy and response length.

[Clarity]
Writing and structure are strong; key assumptions (e.g., treating H(r)H(r) as constant) and the IB surrogate objective (Theorem 1) are presented cleanly.

[Significance]
 Low-overhead regularization that slots into existing RLVR pipelines is practically useful for groups already running PPO/GRPO-style training.

### Weaknesses
1. Practical novelty is limited (IB-on-tokens).
While the IB framing is elegant, the implementable contribution (advantage-weighted entropy) is close to known entropy/density-shaping ideas; the main novelty is the IB interpretation and the advantage weighting rather than a fundamentally new mechanism. Please position against recent entropy-minimization/entropy-shaping/diversity-aware policy works with a small comparison table (objective, where entropy acts, when it increases/decreases, and reported effects).

2. IB feasibility/faithfulness.
The mutual information terms among $r$, $q$, $a$ are not measured; they’re upper-bounded/approximated via entropy surrogates plus assumptions (e.g., treating $𝐻(𝑟)$ constant; bounding $𝐻(𝑜_𝑡\mid 𝑞, 𝑎)$ before mapping to advantage-weighted entropy. Please (i) quantify approximation error or (ii) at least correlate the proposed $𝐿_{𝐼𝐵}$ proxy with held-out accuracy during training to empirically support the IB interpretation.

3. Theory is largely incremental.
Appendix A reuses a known generalization framework (their Theorem 3 is Theorem 2 of Kawaguchi et al., 2023) and brings it into the present notation; the new step is substituting the IBRO-style loss $𝐿_{𝐼𝐵}$​ and relating it to $\|\delta\theta\|$. This is a solid contextualization but not a new bound. Please state this clearly and emphasize the paper’s value as a principled application of IB theory to RLVR rather than a theoretical breakthrough.

4. Baselines are narrow.
Only PPO and DAPO are considered; all are RLVR. To support broader claims about “reasoning,” please add at least one non-RL baseline (e.g., SFT/CoT-SFT, DPO), or show that adding IB-reg to a non-RL finetuning objective also helps. If compute is tight, a smaller-scale ablation on GSM8K with LoRA would still be informative.

5. Benchmarks/metrics are narrow.
All three datasets are math-only (AMC23/AIME24/AIME25), and the primary metric is avg@32. Add 1–2 broader reasoning sets (e.g., GSM8K, BBH, MATH-500) and include at least pass@k/top-1 and a length-controlled metric to rule out length confounds.

6. Ablations are thin.
Current ablations only show entropy dynamics and mean lengths. Please include:
i) Hyperparameter sensitivity for \alpha and \beta;
ii) Remove advantage weighting (plain entropy reg) to isolate its contribution;
iii) Try sign-only advantages (positive vs negative) to test the mechanism posited for DAPO/GRPO;
iv) A short compute overhead table (wall-clock and FLOPs) to substantiate “negligible cost.”

7. Missing qualitative case studies.
Include side-by-side CoTs (success and failure) showing how IB-reg changes intermediate steps (e.g., earlier pruning of spurious paths or delayed [EOS] as hypothesized). This would ground the mechanism beyond aggregate metrics.

8. Reproducibility/finetuning details are underspecified.
It’s unclear whether training is full-model or LoRA/PEFT, and the exact compute profile is missing. Please clarify parameter-efficiency, optimizer state sharding, gradient checkpointing, reference policy handling (they say no KL to a reference), and batch/throughput numbers; this is critical for reproducing RLVR at 7B. The appendices list many hparams, but the finetuning mode (full vs PEFT) and hardware/runtime are not explicit.

### Questions
1. Faithfulness of IB surrogate:
Could you provide empirical evidence (e.g., correlation analysis) that the proposed surrogate 
$𝐿_{𝐼𝐵}$ meaningfully reflects mutual information among $𝑟$, $𝑞$, $𝑎$ or tracks accuracy improvements during training?

2. Novelty of the theoretical result:
Theorem 3 appears to restate an existing IB generalization bound (Kawaguchi et al., 2023) with minimal change. Could you clarify what new theoretical insight or justification your adaptation provides?

3. Baselines and generalization:
The experiments only include PPO and DAPO. Do you expect IB regularization to benefit non-RL fine-tuning methods (e.g., SFT, DPO, process supervision)?

4. Evaluation scope:
The benchmarks are all math reasoning datasets with avg@32 as the main metric. Could you test on an additional reasoning domain (e.g., GSM8K, BBH) or include other metrics such as pass@k?

5. Qualitative evidence:
Would you consider adding example reasoning traces to show whether IB regularization changes CoT structure, e.g., fewer hallucinations or better intermediate logic?

6. Training setup clarification:
Please clarify whether fine-tuning was full-model or parameter-efficient (e.g., LoRA). This detail is essential for reproducibility and understanding computational cost.

### Soundness
2

### Presentation
4

### Contribution
2
