# LLM Unlearning via Calibrated and Tokenized Negative Preference Alignment

- Decision: Reject
- Scores: 6, 2, 4, 6

## Abstract
Pretrained knowledge memorized in LLMs raises critical concerns over safety and privacy, which has motivated LLM Unlearning as a technique for selectively removing the influences of undesirable knowledge. Existing approaches, rooted in Gradient Ascent (GA), often degrade general domain knowledge while relying on retention data or curated contrastive pairs, which can be either impractical or data and computationally prohibitive. Negative Preference Alignment has been explored for unlearning to tackle the limitations of GA, which, however, remains confined by its choice of reference model and shows undermined performance in realistic data settings. These limitations raise two key questions: i) Can we achieve effective unlearning that quantifies model confidence in undesirable knowledge and uses it to calibrate gradient updates more precisely, thus reducing catastrophic forgetting? ii) Can we make unlearning robust to data scarcity and length variation? We answer both questions affirmatively with CaTNiP (Calibrated and Tokenized Negative Preference Alignment), a principled method that rescales unlearning effects in proportion to the model's token-level confidence, thus ensuring fine-grained control over forgetting. Extensive evaluations on MUSE and WMDP benchmarks demonstrated that our work enables effective unlearning without requiring retention data or contrastive unlearning response pairs, with stronger knowledge forgetting and preservation tradeoffs than state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper focuses on LLM unlearning—selectively erasing undesirable knowledge such as hazardous procedures or copyrighted content while preserving general capabilities. Traditional GA methods often cause catastrophic forgetting and require retention data to maintain performance. NPO improves on this but remains limited by static reference models and length-biased gradients. The authors introduce CATNIP, a calibrated and tokenized negative preference alignment method that uses a reverse policy (1 − πθ) as an adaptive reference to scale unlearning strength with model confidence, and applies token-level optimization to eliminate sequence length bias. This enables precise, data-efficient unlearning without needing retention datasets or contrastive pairs. The approach is theoretically grounded in policy ranking via the Bradley-Terry model and empirically validated on hazardous knowledge (WMDP) and copyrighted text (MUSE-Harry Potter) tasks.

### Strengths
The paper stands out for its rigorous theoretical foundation, deriving the method from preference optimization principles and providing clear gradient formulations that explain why it outperforms GA or NPO, backed by ablation studies that isolate components like tokenization and calibration.

### Weaknesses
- Use of un-normalized reverse policy (1 − πθ) as reference risks instability when model confidence is low or distributions are multimodal; no discussion of failure modes or mitigation.
- Evaluation limited to two benchmarks and smaller models; lacks comparison to recent retention-free methods or large-scale 7B+ models.
- Data efficiency claims rely on 132 QA pairs but include no scaling analysis below this threshold or under noisy/real-world unlearning requests.

### Questions
- Why not explore learned or EMA-based reference policies instead of the fixed reverse approximation?
- Any plans to evaluate on larger models or more diverse unlearning targets?
- How does CATNIP differ from [1] (which mentions calibration) and [2] (which mentions tokenization)—do they share core ideas or are the approaches fundamentally distinct?

[1] Wang, Qizhou, et al. "Towards effective evaluations and comparisons for llm unlearning methods." arXiv preprint arXiv:2406.09179 (2024).

[2] Yang, Puning, et al. "Exploring Criteria of Loss Reweighting to Enhance LLM Unlearning." arXiv preprint arXiv:2505.11953 (2025).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper aims to address LLM unlearning challenges with two perspective: token-wise loss fuction design and real-time reference model optimization. While the topic is clear and easy to understand, the overall motivation and contribution of the work are also clearly articulated but it is not novel. The paper does not convincingly demonstrate why the proposed approach is necessary or how it significantly differs from existing methods.

From a methodological perspective, the proposed idea appears to be a relatively minor variation of prior work. Although this paper compare with several classicial methods in LLM unlearning, recent progresses in ICLR2025 and ICML2025 are ignored which are significantly related tothe contributions of this paper. Thus, the novelty is limited, and the technical contribution lacks depth. More details about this evaluation will be presented in weaknesses.

While the presented experiments provide some evidence of the proposed method’s effectiveness, the claimed superiority and general applicability remain questionable. Consequently, the analysis and conclusion — particularly the statement that the method achieves “stronger knowledge forgetting and preservation trade-offs than state-of-the-art methods” — are not fully convincing.

The paper is well written, with clear exposition and logical organization. However, despite the good writing quality, the major weaknesses in motivation, methodology, and experimental validation significantly limit the overall contribution. Clear writing alone cannot compensate for these substantive shortcomings, and therefore cannot serve as a sufficient reason for acceptance.

### Strengths
The motivation of the paper and the corresponding methodological design are centered around two main observations:

1. Sample-wise optimization limitation: Existing unlearning methods operate in a sample-wise manner, without distinguishing between short and long answers. As a result, longer samples receive disproportionate optimization attention, potentially biasing the learning process.

2. Static reference model issue: Current approaches rely on a static reference model (i.e., the pre-unlearned model). During training, some samples are densely distributed in the static model’s feature space, which limits the effectiveness of unlearning for those samples and prevents the model from fully leveraging the knowledge they contain.

These issues are indeed timely and important for LLM unlearning. The paper clearly identifies them and attempts to propose a “novel” method to address them. Overall, the manuscript is well organized and clearly written.

### Weaknesses
1. Motivation: As mentioned in the Strengths section, the paper’s motivation focuses on two subtle aspects of classical unlearning methods. These aspects are indeed relevant and could, in principle, inspire meaningful improvements in future work. However, it is unfortunate that the authors seem largely unaware of recent advances in this area.

    Specifically, both of the paper’s stated motivations have already been explored in the recent literature.

    (1). Token-wise optimization: Several works published at ICLR 2025 [1] and ICML 2025 [2] have already introduced token-wise loss designs for LLM unlearning tasks, addressing the same limitation discussed here. Thus, the first motivation lacks novelty.

    (2). Dynamic reference models: Similarly, existing token-wise approaches [1][2][3] already adopt dynamic reference models ($\phi_t$) instead of static ones ($\phi_0$). Consequently, the second motivation also fails to offer a genuinely new perspective.

    In summary, while the motivation is clearly articulated, it does not present any substantial innovation compared to recent state-of-the-art research.


2. Method: As shown in Equation (9), the proposed CaTNiP essentially functions as a token-wise reweighting method. Prior studies have already demonstrated empirically that token-wise approaches generally outperform sample-wise ones. 

    However, this paper does not include comparisons with other token-wise baselines (e.g., WGA[1], SatImp[2]), yet still claims to outperform the state of the art. Such a claim is not well supported and therefore lacks credibility. 

3. Experiment: The experimental section is insufficient to validate the proposed method. 
    
     (1). Benchmark: Only the WMDP and part of the MUSE benchmarks are utilized, leaving out other important datasets such as TOFU. 

     (2). Model: Given that Zephyr-7B is a standard backbone for WMDP, the choice is reasonable; however, the rest of the experiments are conducted solely with LLaMA-3B. This setup is too limited to demonstrate the generality of CaTNiP. 
    
    It remains unclear how the method performs on smaller (e.g., 1B) or larger (e.g., 8B) models. Similarly, results on MUSE-News are missing—only MUSE-Books is reported. Furthermore, TOFU is one of the most widely adopted benchmarks for LLM unlearning, yet no experiments are provided on it. The absence of TOFU results significantly weakens the empirical validation and makes it difficult to assess the robustness and general applicability of the proposed approach.

[1] Rethinking LLM Unlearning Objectives: A Gradient Perspective and Go Beyond. ICLR2025

[2] Exploring Criteria of Loss Reweighting to Enhance LLM Unlearning. ICML2025

[3] RULE: Reinforcement UnLEarning Achieves Forget–retain Pareto Optimality. Arxiv 2506.07171, NIPS2025

### Questions
Please refer to the weakness.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes CaTNiP, a new method for LLM unlearning, i.e., selectively removing specific undesirable knowledge (e.g., copyrighted or hazardous information) from pretrained large language models.

Extensive experiments on WMDP (hazardous knowledge) and MUSE (copyrighted text) benchmarks show that CaTNiP achieves superior unlearning–retention trade-offs compared to GA, NPO, SimNPO, FLAT, and RMU, even without retention or contrastive data.

### Strengths
The paper contributes a novel perspective on LLM unlearning, grounding it in policy-level preference alignment rather than response-level contrastive optimization.

The tokenized unlearning formulation is an insightful response to a real practical issue—sequence-length bias—that plagues many recent alignment and unlearning methods.

The paper provides clear mathematical derivations, including how token-level calibration induces rescaled gradients (Eq. 9) and the monotonic weighting behavior 

Experiments： multiple datasets (WMDP–Bio, WMDP–Cyber, MUSE–Books), strong baselines (GA, NPO, SimNPO, FLAT, RMU), and detailed ablations (CATNIPref, CATNIP without tokenization).

### Weaknesses
Limited theoretical formulation: 
While intuitive, defining the reference policy lacks theoretical justification from probabilistic or game-theoretic perspectives. The authors could better formalize why this choice leads to stable optimization and avoids degeneracy.

Absence of comparisons with more recent LLM editing approaches: 
The study focuses on unlearning baselines (GA, NPO, RMU, FLAT) but omits recent localized editing and LoRA-based selective forgetting methods.

Evaluation limited to mid-size models (≤7B).
All experiments are conducted on Llama-3B and Zephyr-7B. It remains unclear whether the calibration dynamics scale to larger instruction-tuned models (e.g., 13B, 70B).

### Questions
scalability:  How does CaTNiP perform when applied to larger models (e.g., Llama-13B or Mistral-7B)?

Does the adaptive reference introduce additional computational overhead or memory cost compared to standard NPO?

robustness and composability:  Can CaTNiP be composed sequentially for multiple unlearning tasks (e.g., forgetting multiple domains)?

interpretability:  Could the authors provide visualizations of token-level unlearning beyond the single case study (Fig. 2)?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work proposes a loss objective (CATNIP) for LLM unlearning. The proposed loss is motivated by negative preference optimization (NPO) but differs in two aspects that address the reference model bias and the token-level bias. Concretely, CATNIP uses $1-\pi_\theta(\cdot|x)$ as the reference model, which corresponds to the model’s confidence in its output. In addition, CATNIP performs a token-wise average instead of a response-wise average to allow for varying confidence levels within the same response. Notably, CATNIP achieves comparable (or better) performance to NPO and SimNPO on multiple unlearning benchmarks, including MUSE and WMDP, without using additional retention data.

### Strengths
I find the paper well-written and easy to understand. The motivation of CAINIP is explained. The authors also perform extensive evaluation and comparison of their method with other baselines. I find the experimental results convincing and demonstrate the effectiveness of CAINIP.

### Weaknesses
It is unclear why $1-\pi_\theta(\cdot|x)$ is a good choice for reference model (or confidence). In principle any positive function decreasing in  $\pi_\theta(\cdot|x)$ can be used as a reference model.

### Questions
1. As above, what are the motivations for using $1-\pi_\theta(\cdot|x)$ as the reference model.


2.  I wonder whether the authors have observed different unlearning speeds across tokens within the same unlearned samples. More specifically, do tokens containing sensitive information tend to be unlearned faster, while tokens providing non-sensitive information (e.g., grammatical structure) maintain higher probabilities?


3. Can the performance of CATNIP be further improved if adding a retention loss as in NPO+KL?

### Soundness
3

### Presentation
4

### Contribution
3
