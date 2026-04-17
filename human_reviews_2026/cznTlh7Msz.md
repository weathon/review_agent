# Rainbow Padding: Mitigating Early Termination in Instruction-Tuned Diffusion LLMs

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 4

## Abstract
Diffusion large language models (dLLMs) have emerged as a promising alternative to autoregressive models, offering flexible generation orders and strong performance on complex reasoning tasks. 
However, instruction-tuned dLLMs exhibit a critical vulnerability we term \<eos\> overflow: as allocated sequence length increases, responses paradoxically become shorter, collapsing into early termination or degenerating into streams of \<eos\> tokens. 
Although noticed in practice, this issue has not been systematically analyzed. We trace its root cause to the dual role of \<eos\> as both termination and padding, which concentrates probability mass on \<eos\> at later positions and propagates backward to trigger early termination. 
To address this, we introduce Rainbow Padding, a simple remedy that replaces repeated \<eos\> placeholders with a repeating cycle of distinct padding tokens, distributing probability mass and breaking \<eos\> dominance. 
Experiments show that Rainbow Padding substantially improves length robustness and output quality, with as few as seven padding tokens to prevent early termination. 
Moreover, the method integrates efficiently into existing instruction-tuned models: LoRA fine-tuning for a single epoch on minimal data yields significant improvements, making this solution highly practical.
The project is available at ~\url{https://ai-isl.github.io/rainbow-padding}

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work addresses the <eos> overflow phenomenon in instruction-tuned diffusion-based large language models (dLLMs). The authors identify the root cause is that the <eos> is both used as padding token and sequence terminator, which lead to overestimated probability of <eos> at later positions. The authors propose Rainbow Padding to replace <eos> padding tokens with a cyclical sequence of distinct padding tokens. Experiments are conducted to prove that the Rainbow Padding gains overall improvements on various benchmarks with minimal training cost.

### Strengths
1. The paper defines and analyzes the <eos> overflow issue, which has practical implications for deploying dLLMs in instruction-following scenarios.
2. Rainbow Padding is intuitive, easy to implement, and does not require architectural changes or complex decoding strategies.
3. Comprehensive experiments are conducted on multiple benchmarks with insightful analysis.

### Weaknesses
1. The paper lacks formal analysis or modeling of why Rainbow Padding works, especially in terms of training dynamics or probabilistic behavior.
2. The use of 7 distinct padding tokens is empirically justified, but no principled method or adaptive mechanism is provided for selecting this number.

### Questions
1. Have you considered adaptive methods for selecting the number of padding tokens?
2. Will Rainbow Padding introduce additional bias in length control, such as over-generation or difficulty in terminating properly?
3. Have you validate your methods on some larger LLM architectures?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper identifies a critical failure phenomenon in instruction-tuned diffusion LLMs (dLLMs), termed eos overflow. This phenomenon describes a performance degradation where a longer generation length leads to prematurely short or degenerated responses. The authors trace the root cause to the dual use of the eos token for both sequence termination and padding. To resolve this, the paper introduces Rainbow Padding, a simple yet effective method that reserves a single <eos> for termination and uses a cyclic sequence of <pad_k> tokens for padding. Experiments on models like LLaDA and Dream show that Rainbow Padding effectively eliminates this failure mode, substantially improving performance on reasoning and code generation tasks and restoring length robustness with minimal fine-tuning overhead.

### Strengths
- The paper defines and analyze a critical failure mode (<eos> overflow) in dLLMs. 
-  Rainbow Padding is a simple but effective solution. Its minimal computational overhead (demonstrated by the efficient LoRA adaptation) makes it a very practical method.
- The experiments are thorough and convincing. The performance improvements on length-sensitive tasks like MATH (e.g., from 0.9% to 34.3% on LLaDA) provide evidence of the method's efficacy. The validation across multiple models, tasks, and decoding strategies demonstrates its robustness.

### Weaknesses
- The evaluations on the length-sensitive MATH and GSM8K benchmarks were performed on randomly sampled subsets (>100 problems each) rather than the full test sets. I would like to see deterministic results on the full benchmarks, which may be helpful for future comparisons.
- Alternative Padding Schemes: The paper argues that a single <pad> token would reintroduce the problem of probability concentration. Did you experiment with any non-cyclic, deterministic schemes or a simple random sampling of padding tokens from a small, dedicated set? A empirical comparison might further strengthen the case for the cyclic approach.
- Generality Beyond Instruction-Tuning: Does the <eos> overflow issue also appear in pre-trained dLLMs as well? Is Rainbow Padding suitable for all stages of dLLM training, or specific for the instruction-tuning stage?
- Choice of K: The ablation on K (Table 4) is very insightful, showing a plateau around K=7. Is there a theoretical or heuristic basis for choosing an optimal K? For instance, could it be related to the vocabulary size, model capacity? Or is this a hyperparameter that one must tune empirically for each model?

### Questions
Please refer to weaknesses.

### Soundness
3

### Presentation
4

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
The paper describes a method for improving instruction-tuned dLLMs. It should be of interest to anyone training such models. 

- Identifies and formalizes “eos overflow”: increasing the maximum generation budget paradoxically shortens outputs for IT’d dLLMs.
- Traces the cause to dual use of eos as both stop symbol and padding during IT, which teaches strong tail-position priors for <eos>.
- Shows how any-order decoding heuristics then select high-confidence tail positions early, predicting <eos> and triggering a backward termination cascade.
- Proposes Rainbow Padding: keep a single eos for true termination and fill the padded tail with a deterministic cycle of K distinct pad tokens.
- Demonstrates content-first decoding after the fix, robust gains across tasks/models/decoding rules, and fast convergence of the pad-loss.
- Shows LoRA fine-tuning can retrofit existing IT models to eliminate length collapse.

### Strengths
- simple and effective: change in padding semantics + brief fine-tune yields large accuracy and length-robustness gains.
- decoder-agnostic robustness: works with confidence/margin/entropy decoding and without block scheduling.
- compelling mechanistic evidence: visualization of pad confidences, eos tail priors, and unmasking order link cause to effect.

### Weaknesses
- some results use random subsets (e.g., MATH/GSM8K) and only two dLLM families; full-set, multi-seed reporting would strengthen claims.
- limited head-to-head vs. stronger decoding-time fixes (e.g., calibrated eos priors, length-control objectives) or matched-budget AR models.

### Questions
- are pads new vocab items or reserved rare tokens? How is contamination avoided in prompts/instructions?
- does K=7 remain optimal at longer budgets (2k–4k tokens) and other corpora?
- how does Rainbow compare to (i) eos suppression with calibrated thresholds, (ii) separate <pad> with loss masking only, (iii) weak length-control priors?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper identifies and analyzes a major failure mode in dLLMs known as eos overflow, where allocating longer generation lengths paradoxically causes models to terminate earlier or output streams of eos tokens. The authors trace the issue to the dual use of the eos token as both a true sequence terminator and padding during instruction-tuning, which biases the model toward premature endings. To address this, they propose Rainbow Padding, a simple yet effective fix that replaces repeated eos paddings with a cyclic set of distinct padding tokens, thereby distributing probability mass and decoupling padding from termination. Experiments across reasoning and code generation benchmarks show that Rainbow Padding eliminates early termination, improves accuracy and length robustness, and can be applied efficiently via lightweight fine-tuning.

### Strengths
* **Originality**: The paper is original in systematically identifying, naming, and diagnosing the critical eos overflow failure mode in diffusion LLMs.
* **Quality**: The claims are supported by empirical evidence and thorough ablations, reliably demonstrating substantial performance gains across multiple models and benchmarks.
* **Clarity**: The paper is well-written, logically structured, and explains complex ideas using descriptive names and visual aids.
* **Significance**: This work addresses a reliability issue that previously hampered the practical utility of instruction-tuned dLLMs.

### Weaknesses
- Evaluation inconsistency: The paper reports that “MATH and GSM8K use randomly sampled subsets (>100 problems each).” Such random sampling can lead to unstable or non-reproducible results. It is recommended to evaluate on fixed and standardized test sets (e.g., the MATH-500 subset) to ensure reproducibility and fair comparison across methods.

- Insufficient experimental coverage: The experiments focus solely on fine-tuning from Base models. It remains unclear whether the proposed method generalizes to instruction-tuned models (e.g., LLaDA-Instruct). Additional experiments on such models would strengthen the claim of universality and demonstrate the method’s applicability to real instruction-following settings.

### Questions
The current analysis presents $K=7$ as a sufficient number of padding tokens, but the paper does not fully detail a principled or efficient method for determining this optimal $K$ valuefor dLLMs. We ask the authors to provide a clearer discussion or heuristic on how the smallest effective value of $K$ can be efficiently determined. Furthermore, please clarify whether the saturation point observed for LLaDA is expected to be a universal property that can generalize to other dLLMs.

### Soundness
2

### Presentation
2

### Contribution
2
