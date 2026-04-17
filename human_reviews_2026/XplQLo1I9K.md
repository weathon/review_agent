# Reasoning to Regulate: Chain-of-Thought for Traffic Rule Understanding

- Decision: Reject
- Scores: 2, 4, 2, 6

## Abstract
Understanding and complying with traffic regulations is a safety-critical requirement for autonomous driving, yet remains challenging due to the diversity and context dependence of traffic signage. Importantly, regulation understanding is not a simple recognition task, but a reasoning problem: whether a rule applies depends on interpreting the sign in relation to the spatial layout of lanes and scene context.
To support such reasoning, MapDR provide fine-grained annotations that link each traffic sign’s regulatory rules to the specific lanes they govern. Existing methods, however, largely treat this as direct sequence prediction, ignoring the underlying reasoning that connects sign semantics and map structure.
To address this limitation, we explicitly incorporate reasoning into this task and propose a framework that equips vision-language models (VLMs) with chain-of-thought (CoT) capabilities. We first design a scalable CoT curation pipeline that bootstraps rationales from a strong LLM through a two-round strategy and employs a VLM-based verifier to filter out incorrect cases, yielding a high-quality set of (CoT, answer) pairs. Building on this foundation, we adopt a two-stage training scheme: supervised fine-tuning (SFT) to teach rationale-to-answer generation, followed by GRPO reinforcement learning with answer-grounded, fine-grained rewards to further improve final answer accuracy.
Extensive experiments on MapDR show that our approach significantly improves both interpretability and accuracy, establishing the first reasoning-based framework for regulation-aware autonomous driving.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes to introduce CoT into traffic sign understanding. A pipeline of CoT data collection, verification and application is designed, and deployment of such a method results in a better performance in traffic sign understanding.

### Strengths
1. The data collection (automatic pipeline) is carefully designed. While there could be problems (weakness 3), these can at least help to reduce human labor and improve the acceptance rate for auto-generated data.
2. The authors make use of LLM's comprehensive abilities in data collection and data judgement, to provide data (though quality hard to quantitatively evaluate) helpful to downstream tasks like traffic sign understanding, which could be beneficial in the engineering level.

### Weaknesses
1. The novelty might be limited. It seems that CoT is nearly a standard operation for any reasoning LLM, including those used as base models for driving LLMs. Furthermore, it seems that few labeled reasoning data are introduced, which to me further questions the contribution of this work in improving CoT reasoning.
2. The motivation is questionable. For most traffic signs, they are standard so understanding them via VLM might not be beneficial to the overall accuracy, considering VLM's performance compared to direct perceptions of signs in specific patterns. For rare non-standard traffic signs, they are normally presented with short and clear text, where OCR-then-LLM might be much more straightforward. Therefore, I recommend the authors to reconsider the necessity of motivation.
3. The quality of the CoT data should be human verified. These data serve as the textbook for the VLM to generalize, so their accuracy is very important. Without human verification or at least human evaluations of their accuracy, it is hard to evaluate their contributions in downstream tasks.

### Questions
Please see the weakness part.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces CoT reasoning to the task of traffic rule understanding and lane association using VLMs. It proposes a pipeline to generate and filter CoT data, followed by a two-stage training process (SFT + GRPO) with fine-grained rewards, achieving improved accuracy and interpretability on the MapDR benchmark.

### Strengths
The paper's key contribution is integrating explicit CoT reasoning into the challenging task of mapping traffic rules to specific lanes, moving beyond simpler end-to-end prediction. Its strengths include a novel, scalable CoT data curation pipeline featuring a two-round generation strategy and VLM-based filtering . The two-stage training (SFT + GRPO) with fine-grained rewards is well-structured and demonstrates effectiveness. Results show a significant boost in the overall F1 score, primarily driven by improved rule-lane correspondence reasoning, supporting the value of the reasoning-focused approach.

### Weaknesses
While the paper aims to unify rule understanding and lane association via CoT, the approach still addresses two distinct components (rule extraction and correspondence reasoning), as reflected in the metrics and reward structure . The CoT acts as a bridge, but its effectiveness in creating a truly synergistic solution versus a sequential execution of sub-tasks could be further explored. Additionally, the reliance on external VLM/LLM APIs for CoT generation and filtering raises concerns about dependency and potential bias propagation. The slight decrease in Rule Extraction performance versus the baseline also needs attention.

### Questions
1. The CoT generation and filtering depend on specific Qwen APIs. How sensitive is the data quality and subsequent model performance to the choice of these external models without finetuning themselves?
2. Table 1 shows improved Correspondence Reasoning but slightly lower Rule Extraction scores compared to RuleVLM. Does the CoT approach inherently cause a trade-off, potentially focusing reasoning resources on spatial association at the expense of fine-grained rule attribute extraction?
3. Need more elaboration on the VLM-based filter, how reliable it is? What kinds of incorrect reasoning does it successfully catch, and are there common failure modes where faulty CoT might still pass validation?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose a new reasoning data augmentation pipeline that claims to improve the rule-to-lane understanding of current models.

### Strengths
The paper curates a reasoning data augmentation pipeline that seemingly demonstrates its potential usefulness towards traffic scene understanding. However, the paper lacks of significant evidence and experimental details to support its arguments.

### Weaknesses
While the paper presents a new reasoning-based framework for traffic rule understanding, several aspects of its experimental reporting and analysis remain insufficiently detailed. First, the paper does not provide enough statistics or transparency regarding the training configuration of the vision-language models, such as dataset split ratios, number of training samples after filtering, hyperparameter choices, or computational resources used. This lack of detail makes it difficult to assess the reproducibility and scalability of the proposed approach. Second, while the authors emphasize the construction of a high-quality Chain-of-Thought (CoT) dataset through multi-stage filtering and verification, the paper does not clearly quantify how data quality improvements translate to model performance. For instance, it remains unclear how much the filtering process or rationale correctness directly contributes to the observed performance gains. Finally, the paper provides limited qualitative analysis to illustrate the strengths of the proposed method. Although some examples of reasoning outputs are shown, these are sparse and lack comparative visualization against baseline models, making it difficult to understand how the model’s reasoning improves interpretability or decision accuracy in specific traffic scenarios. Together, these gaps leave the reader uncertain about the robustness and practical advantages of the proposed reasoning framework.

### Questions
Refer to weaknesses.

### Soundness
2

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
This paper proposes a VLM with chain-of-thought plus two stage training to map traffic rules to governed lanes on MapDR. The problem is that existing models like RuleVLM directly generate rule lane outputs but cannot reason over sign attributes and lane topology together. Experiments like Table 1 show that the proposed method gets 72.3 F1 while RuleVLM gets 64.2 proving explicit reasoning improves lane level correspondence on MapDR.

### Strengths
- The reviewer finds the proposed idea to insert curated CoT and GRPO reward into MapDR style rule lane association to be interesting.
- Interestingly, rule extraction precision drops from 89.28 to 87.71 but overall F1 still rises which means better reasoning beats pure extraction.
- The experiments in Table 1, Table 2 and Figure 1 clearly show gains from CoT curation and from SFT plus GRPO on the same backbone.
- Writing is easy to follow.

### Weaknesses
- The CoT filtering stage in Section 3.1 keeps 4517 of 11060 samples but does not report recall or noise rate so reliability is unclear. Authors can clarify this. 
- The experiments do not add a simple geometry only lane nearest to sign baseline or a sign text only baseline although lines 324 to 335 explain GRPO reward on those parts so these should be easy to report.
- A minor typo - Line 209: contradict -> contradictory

### Questions
- In Table 1, "Ours" has lower PR.E than RuleVLM but higher overall F1. Can you decompose gains into rule understanding and rule lane association?
- Section 3.1 says a two round LLM generates and revises CoT and a VLM verifier filters but the exact prompt templates and pass thresholds are not given. Can you please add this to the paper?

### Soundness
3

### Presentation
3

### Contribution
3
