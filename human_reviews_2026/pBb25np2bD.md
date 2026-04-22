# How to Teach Large Multimodal Models New Skills

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
How can we teach large multimodal models (LMMs) new skills without erasing prior abilities? We study sequential fine‑tuning on five target skills while monitoring general ability on eight held‑out benchmarks across three model families. We observe that apparent “forgetting” on held‑out tasks after narrow fine‑tuning can partly recover at later stages. We trace this behavior to a measurable shift in the output token distribution, manifested through a simple counting‑bias probe that identifies the shift co‑varies with forgetting. Guided by this picture, we identify two simple, robust tuning recipes that learn strongly while limiting drift: (i) updating only the self‑attention projection layers, and (ii) updating only the MLP Gate\&Up while freezing the Down projection. Across models and tasks, these choices deliver strong target gains while largely preserving held‑out performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper focuses on the continual learning problem of LMMs. Through fine-tuning experiments and a counting-bias probe, it reveals that "forgetting" of held-out tasks post-fine-tuning is essentially recoverable output token distribution shift. It further proposes two fine-tuning schemes, updating only self-attention projection layers and updating only MLP’s Gate&Up layers while freezing Down layer, which ensure strong target-task learning with minimal forgetting across multiple models and tasks.

### Strengths
1. Through fine-tuning experiments and a counting-bias probe, the paper first clarifies that the "forgetting" of held-out tasks in LMMs after fine-tuning is essentially a shift in the output token distribution. Moreover, this shift is partially recoverable via subsequent task fine-tuning, offering a crucial theoretical perspective on LMMs continual learning core issues.
2. The paper proposes two concise fine-tuning methods that balance strong target-task learning and minimal original-capability forgetting, validated across 3 model families, 5 target task types and 8 held-out benchmarks for robust generalization and reliability.

### Weaknesses
1. Regarding the question of whether the two fine-tuning methods can be combined to improve performance, Appendix F.1 notes simultaneous fine-tuning of SA Proj. and MLP (Gate&Up) offers no gain or even degrades performance, but lacks in-depth explanation of the underlying mechanism; alternative combinations (e.g., two-stage fine-tuning) are also unexplored.
2. The paper only compares its strategy with traditional methods (LoRA, WiSE-FT, MoE) and excludes recent mainstream continual learning schemes, failing to clarify the strategy’s competitiveness.
3. The "forgotten knowledge recoverability" claim lacks rigorous explanation/verification: no causal validation via experiments like "adjusting distribution without training new tasks", no quantification of distribution correction-recovery correlation, and unclear recovery triggers, reducing practical value.
4. Focused on quantitative metrics, the paper does not compare fine-tuned models’ output differences for the same input or analyze intermediate feature changes before/after forgetting recovery, hindering intuitive understanding of the strategy’s effect.

### Questions
1. The paper fails to quantify the training efficiency (parameter count, computation time, memory usage) of SA Proj. and MLP (Gate&Up), precluding efficiency comparison with full-model and LoRA fine-tuning. Can experimental data be supplemented to clarify applicability in resource-constrained scenarios?
2. The paper innovatively uses a counting-bias probe to verify token distribution-forgetting correlation. Can similar probes (e.g., for medical VQA, clock reading) be tested to confirm if such probes can be generalized as a universal tool for measuring task-specific token distribution shifts?

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
In this paper, the authors compare various layer-level fine-tuning strategies and find that selectively tuning only the self-attention projection layers or the MLP’s up and gate components allows efficient learning of new tasks while retaining performance on previous ones. Using a counting-bias probe method, the paper further shows that forgetting mainly arises from output distribution drift rather than genuine knowledge loss.

### Strengths
- The use of held-out benchmarks is highly valuable. Existing benchmarks typically measure forgetting only with respect to previously trained tasks within the same benchmark, without considering the preservation of the model’s intrinsic capabilities.
- The discovery that fine-tuning the Self-Attention Projection (SA Proj.) or MLP Gate&Up layers can acquire new knowledge while greatly reducing forgetting of existing abilities is both effective and practically straightforward.

### Weaknesses
- The conclusion that tuning SA Proj. and Gate&Up does not lead to significant forgetting has not been validated on other benchmarks. Therefore, it is difficult to rule out the possibility that this finding stems from dataset bias in the current experimental domain.
- The paper lacks direct comparisons with recent SoTA methods (published in recent two years). Although this paper demonstrates that tuning SA Proj. and Gate&Up is effective, it remains unclear how effective this approach is relative to SoTA baselines.
- The analysis of output distribution drift relies mainly on the counting-bias probe method, so it remains unclear whether the same conclusion holds for other cases where the task outputs are not primarily numeric.

### Questions
See weaknesses.

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
This paper studies continual learning and forgetting in VLM training. The authors propose two simple but effective methods to mitigate the forgetting, that are, (1) only updating the attention and projection (Wq, Wk, Wv, Wo), and (2) updating only the MLP gate and up layers. run experiments on different tasks. The authors do test on 5 different tasks sequence, 8 held-out benchmarks, and 3 different VLM families to verify the effectiveness of the simple methods. There is also an interesting understanding part, trying to map the forgetting to the token-distribution shift.

### Strengths
(1) very clear writing: from my side, it is easy to understand
(2) easy but effective method
(3) experiments are relatively comprehensive on my side
(4) i really appreciate the understanding part, where the authors dive deeper into the reason of forgetting, and map it into the token-distribution shift.

### Weaknesses
The main contribution of the paper is to study which part of the parameters to update (to my understanding, correct me if I am wrong). While indeed the proposed methods already show signal, it is not clear if there is any logic/reasons behind selecting those parameters. There are many other confounding factors, which may make the conclusion change. for example
(1) If the model is larger, are there any other rules for selecting the update parameters?
(2) If the model is larger, will it be beneficial to use LoRA?
(3) Will it be better to select the parameters based on the layer index, e.g., if it is better to update on a later layer than an earlier layer?
I am afraid that in the end, it will just turn into an engineering problem, where you just run all the design choices and pick the best, it there are limited scientific guides.

However, I still agree that the token distribution is interesting.

### Questions
Please see the weakness part. my main questions are regarding these confounding factors.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies effective continual learning strategies in multimodal models. Major findings are: (a) Updating a selected set of parameters can reduce forgetting, and (b) forgetting can be traced in distribution shift in output tokens. The authors corroborate their study with extensive experiments on five target skills while monitoring general ability on eight held‑out benchmarks across three model families.

### Strengths
The main strength of the paper lies in its clear and simplistic presentation of the continual training experiments. By varying the parameters for updating in this sequence learning setting, they clearly show that selecting the right parameters to update can substantially reduce forgetting. Furthermore, through a mechanistic analysis, they connect forgetting behavior to mechanistic roles of attention vs. MLP layers. Through extensive experiments on wide variety of benchmarks and backbone models, the authors show that forgetting can be mitigated with a simple fine-tuning recipe.

### Weaknesses
As such, I don't find any key weaknesses with the paper. I have some questions about the experiment setup:

a) All tasks are vision-language. Do the findings generalize if a text-only task was included in the sequence? If not, is it primarily the issue of the way the backbone LLM has been converted to the multimodal LLM? 

b) How do the results change if you vary the order of the tasks in sequence? The authors evaluate multiple task orders but do not report variance or confidence intervals (e.g. in table 1).

### Questions
Please see above for my questions.

### Soundness
3

### Presentation
3

### Contribution
2
