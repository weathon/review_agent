# Learn to be Honest: Mitigate LLMs' Overconfidence for Improving Hallucination Detection with Self-Hesitation Activation

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 4, 4

## Abstract
While Large Language Models (LLMs) have demonstrated strong performance across a wide range of natural language processing tasks, plausible but unfaithful content is still inevitably generated, which is known as factual hallucination. Previous methods, such as classifier training and uncertainty estimation, have been proposed for hallucination detection. However, it is widely found that LLMs express overconfidence by attempting to rationalize the incorrect outputs, leading to a misalignment between perceived uncertainty and knowledge boundary perception. It significantly undermines the effectiveness of existing hallucination detection methods. We study the correlation between hallucination and overconfidence, arguing that they are systematically inseparable in traditional training strategies due to overtraining. To address this, a series of analyses are conducted and a method called Self-Hesitation Activation Fine-Tuning (SHAFT) is proposed to align the uncertainty with the factual correctness, making LLMs "More Honest". Experiments demonstrate that our approach significantly mitigates the overconfidence of LLMs and decouples overconfidence with hallucination, making the nonfactual instances more distinguishable. Furthermore, evaluations across three benchmarks reveal that SHAFT greatly improves the performance of various hallucination detection methods before generation, consistently indicating its generalizability and computational efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper addresses the problem that large language models (LLMs) often produce factually incorrect outputs while being overconfident, which undermines hallucination detection methods that rely on uncertainty signals. The authors identify a phenomenon they call self-hesitation, a transient state during fine-tuning where model confidence better aligns with correctness. They propose Self-Hesitation Activation Fine-Tuning (SHAFT), a fine-tuning strategy designed to stabilize this state. The method is evaluated across multiple datasets, two LLM families, and several baseline uncertainty estimation methods, showing improved alignment between confidence and correctness.

### Strengths
The paper presents an interesting observation on confidence propagation during supervised fine-tuning and proposes a novel method to mitigate model overconfidence. Experiments on multiple datasets demonstrate the effectiveness of the proposed approach.

### Weaknesses
(a) My major concern is the lack of comparison with standard SFT. While Table 1 compares the performance before and after the proposed fine-tuning method, the comparison is not fair because additional computation and extra data are utilized. To clearly demonstrate the benefit of the proposed approach, the baseline should include standard supervised fine-tuning (SFT) with cross-entropy loss.

(b) Averaging accuracy and AUROC in Table 1 is methodologically unsound — they measure fundamentally different aspects and are not directly comparable. This undermines the validity of some aggregated results and makes it unclear how much improvement truly comes from each dimension.

(c) The paper is generally understandable but would benefit from careful editing for grammar and phrasing. To name a couple: 

Line 13: .. enables LLMs to learning new knowledge... --> to learn

Line  306:  Three white-box and open-source models are introduced for the experiments, including LLaMA3-8B-Instruct (Dubey et al., 2024) and Qwen2.5-7B-Instruct (Yang et al., 2024). --> the paper only presents results on two models.

### Questions
see weakness

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
This paper proposes an uncertainty-based training method called SHAFT. It applies adaptive weighting based on model confidence: for confident-but-incorrect predictions, it assigns high training weights; for uncertain predictions, it uses moderate weights; for correct predictions where the target token already has the highest probability, it skips training entirely. Experiments demonstrate strong performance improvements across multiple hallucination detection methods on three QA benchmarks.

### Strengths
1. SHAFT’s selective and adaptive training strategy makes sense. It adjusts the output distribution to better reflect the model’s actual knowledge boundaries rather than blindly reinforcing the pretrained model’s priors. It naturally aligns confidence with correctness. The method avoids overtraining on samples the model already knows while focusing on challenging cases.
2. The performance improvements are substantial and consistent, with gains of 5-17% across multiple hallucination detection methods, two different LLMs, and three benchmarks, demonstrating both effectiveness and generalizability.

### Weaknesses
The paper lacks baseline comparisons with similar SFT-based calibration methods: (1) Uncertainty-Aware Causal Language Modeling (UA-CLM) (Liu et al., 2024) which also modifies the loss function during fine-tuning to emphasize uncertainty for wrong predictions while optimizing for certainty on correct ones; (2) Unlikelihood Training (ULT) (Welleck et al., 2020) which adjusts training objectives by penalizing certain tokens. It would be better to have some comparison with previous methods.

### Questions
TriviaQA failure analysis: Why does SHAFT catastrophically fail on TriviaQA (hallucination rate increases from 42.7% to 64.7%, Table 4), while improving on NQ and PopQA? The explanation about "knowledge domain inconsistency" is vague. What specific characteristics of TriviaQA cause this? Can you provide case studies showing what goes wrong and whether this indicates fundamental limitations for certain question types?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Self-Hesitation Activation Fine-Tuning (SHAFT), a training method that mitigates LLM overconfidence by stabilizing an intermediate "self-hesitation" state where models honestly express low confidence when generating incorrect outputs, thereby improving hallucination detection performance.

### Strengths
1. Effective calibration approach: This work calibrates LLM outputs by aligning confidence with correctness, thereby enhancing the reliability of model predictions.
2. Simple and accessible methodology: The proposed method is straightforward to understand and implement, requiring only minimal modifications to standard training procedures.

### Weaknesses
1. Similarity to prior calibration methods: The core idea overlaps significantly with previous works "Alignment for Honesty" (training models to honestly express uncertainty) and "R-tuning" (selective training on hard samples for calibration). The paper does not clarify how SHAFT differs or what specific advantages its loss formulation provides over these existing approaches.
2. Missing comparisons: No empirical comparisons or ablations against these related methods are provided, making it unclear whether improvements stem from SHAFT's novel aspects or simply from applying known principles.

### Questions
1. What specifically differentiates SHAFT from "Alignment for Honesty" and "R-tuning"? Please clarify the conceptual and methodological distinctions beyond the specific loss formulation.
2. Can you provide empirical comparisons? If the core ideas are similar, direct performance comparisons with these baseline methods would help establish SHAFT's advantages.


[1]. Yang Y, Chern E, Qiu X, et al. Alignment for honesty[J]. Advances in Neural Information Processing Systems, 2024, 37: 63565-63598.
[2]. Zhang H, Diao S, Lin Y, et al. R-tuning: Instructing large language models to say ‘i don’t know’[C]//Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). 2024: 7106-7132.

### Soundness
3

### Presentation
3

### Contribution
2
