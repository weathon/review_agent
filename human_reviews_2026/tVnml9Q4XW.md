# Turning Internal Gap into Self-Improvement: Promoting the Generation-Understanding Unification in MLLMs

- Decision: Accept (Poster)
- Scores: 2, 8, 6, 8

## Abstract
Although unified MLLMs aim to unify generation and understanding, they are considered to exhibit an internal gap, with understanding outperforming generation. Through large‑scale evaluation across multiple MLLMs and tasks, we confirm the widespread non‑unification of MLLMs, and demonstrate that it indeed stems from weak generation rather than misunderstanding. This finding motivates us to propose a simple yet effective internal gap-based self-improvement framework, which mitigates internal gaps by leveraging stronger understanding to guide weaker generation without relying on any external signals. We validate this strategy through comprehensive experiments: scoring generations with understanding to construct image data for post-training (e.g., SFT and DPO) significantly improves generation while promoting unification. Furthermore, we empirically discover a co-improvement effect of such self-improvement, a phenomenon well known in pre-training but underexplored in post-training. Specifically, as generation improves, understanding becomes more effective at detecting false positives that were previously misclassified as prompt‑aligned. To explain this effect, we extend learning dynamic theory to the MLLM setting, showing that the shared empirical neural tangent kernel between generation and understanding encourages aligned learning dynamics, thereby driving co-improvement. This interplay between generation and understanding further motivates a curriculum learning approach for stronger self‑improvement: progressively enhanced understanding and generation revisit samples underutilized by pre‑trained MLLMs, dynamically expanding post‑training data and leading to improved performance and unification.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper investigates the "internal gap" in unified Multimodal Large Language Models (MLLMs), where understanding consistently outperforms generation. The authors propose a self-improvement framework that leverages stronger understanding to guide weaker generation through standard post-training (SFT/DPO), achieving up to 20% generation improvement without external signals.

### Strengths
The paper provides systematic empirical validation of the generation-understanding gap across multiple MLLMs with clear methodology and reproducible experiments. The theoretical framework extending learning dynamics to multimodal settings offers mathematical formalization, though it essentially explains an expected outcome—that shared parameters naturally lead to correlated improvements. The work is technically sound with detailed ablations.

### Weaknesses
The work is technically sound with detailed ablations, but the core insight that two capabilities in a closed system can mutually improve through iterative training is conceptually straightforward, limiting originality. The self-improvement approach applies standard techniques (SFT/DPO) without novel algorithmic contributions, and still underperforms methods using external rewards, constraining its practical significance. Critically, the theory fails to address fundamental limitations: in a closed system without external supervision, two learners cannot indefinitely improve through mutual training—the paper lacks analysis of performance ceilings or convergence bounds.  Overall, the paper competently documents a predictable phenomenon rather than introducing fundamentally new concepts.
More specifically, the theoretical analysis (Propositions 1-2) provides mathematical formalization but offers limited conceptual depth beyond expected outcomes. Experimental scope is narrow: only two models tested, consistently underperforming external reward baselines, undermining practical value. The non-unification metric relies on potentially unreliable binary judgments, and using Qwen as ground truth introduces unexamined biases. Curriculum learning adds minimal gains without principled design.

### Questions
no more.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper investigates the generation–understanding internal gap in unified multimodal large language models (MLLMs), where the understanding branch consistently outperforms the generation branch.
To address this, the authors propose a simple internal gap–based self-improvement framework that leverages the model’s own understanding capability to guide and enhance generation, without any external rewards or supervision.Key contributions include:

1. Quantitative diagnosis of generation–understanding non-unification, introducing an internal Non-Unification Score to measure intra-model consistency and empirically verifying the widespread internal gap across six MLLMs (Sec. 3).

2. Internal gap–based self-improvement framework, leveraging the stronger understanding branch to score and filter generations for post-training (SFT/DPO), effectively improving generation quality and reducing non-unification without external signals (Sec. 4.1–4.2).

3. Theoretical analysis of the co-improvement effect, extending learning dynamics to multimodal models and revealing that shared empirical neural tangent kernels (eNTKs) drive aligned updates between generation and understanding (Sec. 5).

4. Curriculum-based self-improvement strategy, progressively reintroducing previously underutilized or difficult samples through curriculum replay to further enhance both branches and surpass external-reward baselines (Sec. 6).

### Strengths
1. The paper propose an internal gap–based self-improvement framework, which is conceptually simple but novel, requiring no external supervision or reward models.  The authors also introduce a new internal evaluation metric (Non-Unification Score) to quantify intra-model consistency.

2. The work provides strong empirical evidence through large-scale experiments on six unified MLLMs and three task difficulty levels (Figure 2).

3. Figures and algorithms are clearly presented — e.g., Algorithm 1 succinctly formalizes the self-improvement process, and Figure 5 (a) intuitively illustrates the co-improvement effect with side-by-side visual examples.

4. The discovery of a shared empirical NTK between generation and understanding offers a novel theoretical perspective on multimodal model coupling ( Eq. 2–3). The curriculum-based self-improvement strategy demonstrates a scalable path to strengthen MLLMs without external data or rewards, surpassing reward-model–based baselines such as T2I-R1 and HermesFlow.

### Weaknesses
1. Theory–practice gap: While the shared eNTK explanation is conceptually interesting, its empirical validation remains limited, and the theoretical section is notation-heavy, reducing accessibility for non-theoretical readers.

2. Limited model diversity: Main experiments focus on Janus-Pro and Show-o. Although six models were initially analyzed, most in-depth post-training results come from only two, limiting the generality of conclusions.

### Questions
1. How stable is the self-improvement process when the understanding branch provides noisy or incorrect judgments?

2. Can the proposed framework generalize to other modalities (e.g., video or audio) or to non-generative MLLM tasks?

3. How sensitive are the results to the number of generated candidates N and the selection threshold in the understanding branch?

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
This paper systematically investigates the prevalent "generation-understanding non-unification" phenomenon in Multimodal Large Language Models (MLLMs), where understanding capabilities typically outperform generation capabilities. The authors propose an internal gap-based self-improvement framework that leverages the model's own stronger understanding branch to guide its weaker generation branch. Without relying on external signals, this approach significantly enhances generation quality and reduces the internal gap through post-training methods like SFT or DPO. Experiments reveal that this method also induces a "co-improvement" effect, where improvements in generation simultaneously enhance understanding, particularly in identifying misaligned generated samples. Furthermore, drawing from learning dynamics theory, the paper identifies the shared empirical Neural Tangent Kernel (eNTK) between generation and understanding as the key mechanism behind co-improvement. Based on this, a curriculum learning strategy is proposed to dynamically expand the training data, further boosting model performance and unification.

### Strengths
- Rigorous Problem Verification: The paper first confirms the internal gap in MLLMs where "generation is weaker than understanding" through large-scale evaluation (across 6 models and tasks of varying difficulty). To achieve this, the authors innovatively propose a "non-unification score" that does not rely on external evaluators. 
- Simple and Effective Solution: An "internal gap-based self-improvement" framework is proposed, which leverages the model's own stronger understanding capability to filter generated data (for SFT or DPO). This method requires no external reward models, achieving closed-loop self-improvement.
- Discovery and Explanation of the "Co-improvement" Effect: this paper introduces that "generation-targeted training" also enhances "understanding capability" (particularly in correcting false positive samples). From a learning dynamics perspective, the paper attributes this to the shared eNTK (empirical Neural Tangent Kernel) between the generation and understanding branches.
- Clear Logical Structure Throughout the Paper: The paper exhibits a clear and coherent logical structure from beginning to end.

### Weaknesses
- Experimental Results Heavily Rely on a Single, Insufficiently Strong Judge Model:Two of the paper's key conclusions—that the "internal gap stems mainly from weak generation" and the "co-improvement effect"—rely heavily on using Qwen2.5-VL-72B-Instruct as the sole external judge.
    However, Qwen2.5-VL-72B is not a strong enough multimodal model to serve as a reliable evaluator, especially when dealing with complex or "hard tasks." To enhance the credibility of the experimental results, it is recommended to adopt more advanced SOTA models or incorporate multiple models as evaluators.
- Limited Innovation and Effectiveness of the Curriculum Learning Part:
    The curriculum learning (C-SFT/C-DPO) strategy proposed in Section 6, whose core idea is to reuse "difficult" samples discarded earlier due to the model's limited capability, is lacking innovation and Effectiveness.
    This method is a conventional application of curriculum learning and lacks significant methodological innovation.Based on the experimental results (Table 1 and Table 5), the improvement of C-SFT over standard SFT is very marginal. For example, on the "Overall" metric for T2I-CompBench++ (Table 5), the generation score for Janus-Pro only increased from 43.29 to 44.18, and for Show-o, only from 52.67 to 52.82. This marginal gain seems insufficient to justify the introduction of this strategy.
- Insufficient Experimental Evidence and Lack of Generalization for "Co-Improvement Effect":The "co-improvement effect" (Finding 2), a key contribution, claims that self-improvement targeted at generation also enhances understanding. As mentioned, the effect is primarily shown via the custom, Qwen-dependent "win rate" metric.The improvement on standard benchmarks is Negligible(Table 8). For example, after SFT, Janus-Pro-7B's score on POPE \textit{decreased} from 89.04 to 88.45, the improvement on MMB was only 0.74 (76.23 $\rightarrow$ 76.97), and on GQA, only 0.1 (56.02 $\rightarrow$ 56.12).This significant disconnect between the custom metric (Win Rate) and standard benchmarks (Table 8) suggests that the so-called "co-improvement" might just be overfitting to the internal understanding task or the Qwen judge's preferences, rather than a genuine, generalizable improvement in understanding ability for standard VQA or hallucination detection tasks.

### Questions
A critical ambiguity exists regarding potential data leakage in the T2I-CompBench++ experiments. The authors state in Section 4.2.1 that post-training data was constructed using ``about 6000 text prompts as post-training candidates'' from T2I-CompBench++. Subsequently, the model's performance is evaluated on the T2I-CompBench++ evaluation set (as shown in Table 1 and Table 5) . However, the paper fails to explicitly state whether the 6000 prompts used for post-training (SFT/DPO) were drawn from the benchmark's designatedtraining split. If any overlap exists between these post-training prompts and the prompts in the evaluation set, the reported performance gains on this benchmark would be invalid due to data contamination. This lack of clarity undermines the reliability of these key experimental results.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors let the understanding branch of MLLMs judge the correctness of images generated by the generation branch, as a reflection of the model's internal gap, across multiple models and tasks. They leverage this gap to improve the models' generation ability using SFT and DPO with data generated by the generation branch and verified by the understanding branch, resulting in improved generation, understanding, and internal unification. They explain the enhancement of the understanding ability with a learning dynamics theory and support it with empirical analysis. They further propose to incorporate data that could not be utilized initially in their data-collection pipeline into the post-training process, after the model's ability has improved, thereby further enhancing the training outcome.

### Strengths
1. The self-improvement approach, which uses the understanding ability to guide the generation branch, is intuitive and intriguing in concept and effective in practice.
2. The work is generally solid and logically rigorous, with a motivation validated across multiple models, comprehensive experiments conducted on two models evaluated by both the authors’ proposed metrics and existing benchmarks, theoretical interpretation, and empirical validation. One of the relatively unconvincing and inelegant aspects—using Qwen as an external judge—is compensated for by a human verification experiment.
3. The learning dynamics analysis provides theoretical insights into the mechanism of the co-improvement effect.
4. The paper is generally well written, and most details are explained clearly.

### Weaknesses
The empirical evidence for some of the paper's claims is less conclusive than stated, which lacks further clarification:

1. In Fig.2(c), the claimed "trend of increasing with task difficulty" for the non-unification score is not obvious or monotonic. The variation between models seems to dominate any difficulty-based trend. 
2. In Fig.7, the difference in similarity between improved samples and random samples is also not clear, especially for image pairs.

### Questions
1. What was the detailed setup of the human evaluation in Fig. 9? The appendix mentions the human check but omits details on the number of annotators, the interface used, the specific instructions given. 
2. Could the authors evaluate the self-improved Show-o model on the understanding benchmarks, similar to Janus-Pro-7B in Tab. 8?

### Soundness
3

### Presentation
4

### Contribution
3
