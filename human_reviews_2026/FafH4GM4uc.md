# Universal Reasoner: A Single, Composable Plug-and-Play Reasoner for Frozen LLMs

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 6, 6, 4

## Abstract
Large Language Models (LLMs) have demonstrated remarkable general capabilities, but enhancing skills such as reasoning often demands substantial computational resources and may compromise their generalization. While Parameter-Efficient Fine-Tuning (PEFT) methods offer a more resource-conscious alternative, they typically require retraining for each LLM backbone due to architectural dependencies. To address these challenges, here we propose Universal Reasoner (UniR) - a single, lightweight, composable, and plug-and-play reasoning module that can be used with larger frozen LLMs to endow it with specialized reasoning capabilities. Specifically, UniR decomposes the reward into a standalone reasoning module that is trained in a decoupled manner using predefined rewards, effectively translating trajectory-level signals into token-level guidance. Once trained, UniR can be combined with frozen LLMs at inference time by simply adding its output logits to those of the LLM backbones. This additive structure naturally enables modular composition: multiple UniR modules trained for different tasks can be jointly applied by summing their logits, enabling complex reasoning via composition. Experiments on mathematical reasoning and machine translation show that UniR surpasses existing baseline fine-tuning methods. Furthermore, UniR demonstrates weak-to-strong generalization: reasoning modules trained on smaller models effectively guide much larger LLMs. Beyond this, UniR generalizes across modalities such as in vision language models and domains such as medical reasoning. This makes UniR a cost-efficient, adaptable, and robust solution for enhancing reasoning in LLMs without compromising their core capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a method to decouple reasoning from the base model by introducing a standalone, pluggable reasoning module that can be integrated with any model size or architecture by adding its output logits to those of the base model. The paper shows that it is possible to compose even two reasoning modules to combine task-specific abilities, achieving improved performance over both the base model and the GRPO-trained model (in math and translation tasks).

### Strengths
1. A decoupled reasoning module that is model- and architecture-agnostic (assuming tokenizers can be aligned) with the base model, demonstrating that reasoning can be isolated and trained independently.

2. The reasoning module can be seamlessly integrated by adding its output logits to those of the base model, and it appears to enhance performance even in much larger base models.

3. A linear composition of reasoning modules is possible and effective, generalizing well even to unseen test sets (e.g., mortality prediction).

4. The paper is well presented and demonstrates strong understanding and integration of relevant theoretical concepts.

### Weaknesses
1. How are the coefficients determined when combining reasoning modules?

2. Is it possible to compose more than two modules, and if so, how does performance scale with additional compositions?

### Questions
1. Does the reasoning module require any SFT warm start? Prior studies have shown that directly starting with GRPO, especially on small models, often leads to suboptimal performance.

### Soundness
3

### Presentation
4

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
This paper studies LLM reasoning tasks and proposes UniR, a standalone reasoning module that reduces training costs while achieving generalization across models. UniR is composable and trained via GRPO. It provides token-level logit guidance to the backbone LLM. UniR demonstrates better performance than GRPO LoRA and GRPO Full with lower training costs. Additionally, UniR trained on a smaller backbone can improve task performance when combined with a larger backbone.

### Strengths
This paper has a clear motivation: reduce computational costs in post-training and improve the generalization of learned reasoning abilities. 

The content is high quality and the writing is clear and easy to understand.

The experiments are extensive, covering various benchmarks, tasks, model sizes, and series. The proposed UniR shows better performance compared with baseline methods and improves performance on larger backbones that were not used to train UniR.

When combined, different UniR modules show adjustable performance improvements on math and translation tasks.

### Weaknesses
One weakness is the lack of detailed discussion about time and resource costs. The paper should highlight how UniR differs from baseline methods to demonstrate its computational advantages. 

Another point: the paper needs more analysis of the design choices for the standalone reasoning module. The authors use a smaller LLM as the reasoning module, which is a good initial step. However, have they experimented with using parts of a model instead of a full LLM? Could the reasoning module be further reduced in size?

Additionally, more analysis comparing outputs from baseline GRPO versus UniR would better illustrate UniR's influence. For example, what helpful adjustments does token-level logit modification provide compared to optimizing the model using only the final reward signal?

### Questions
Regarding the first contribution (Line 75–76): circumventing the need for expensive preference dataset creation is not a major contribution of this work. The tasks studied here do not require such preference dataset construction.

In Figure 3, what is the baseline performance—that is, only the base model without the reasoning module? Showing the baseline performance would help readers better understand this figure.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a single "universal" model that converts practically any LLM into a reasoning-enabled LLM by adding logits at inference time. This enables reasoning behavior without the cost of fine-tuning and without the reduction in quality that reasoning-fine-tuning often produces.

### Strengths
This model seems very useful both as a research artifact (it seems to tell us that there isn't all that much to reasoning behavior; not many weights are actually needed to represent it "from scratch") as well as to users of AI tech. Enabling a "plug-and-play" ecosystem of foundation models and reasoning models could have a high impact on the community.

### Weaknesses
Some of the baseline numbers are suspicious. For example, this paper reports a "backbone only" performance on MATH of 28.7 for llama3.2 3B, but huggingface here (https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) reports a 48.0 accuracy and that's consistent with what ArtificialAnalysis reports (https://artificialanalysis.ai/evaluations/math-500).  I'm also having trouble finding the other baseline numbers in prior work or online (GSM8K seems to mostly be 8-shot in evaluations of llama3 as opposed to the 0-shot evaluated in the paper).

### Questions
Are there sources that can corroborate that the baseline numbers in Table 1 and Table 2 are correct (i.e. what we should expect for these models)?

Can you go into more detail about the architecture of the universal reasoner? The paper is vague on this point.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces UniR, a lightweight reasoning module trained with verifiable rewards that can be plugged into frozen LLMs at inference by simply adding its logits to those of the backbone. The module is (i) architecture-agnostic (only tokenizer alignment required), (ii) composable (multiple UniRs can be summed for multi-task reasoning), and (iii) shown to transfer from small to much larger backbones without further tuning. Experiments on mathematical reasoning, machine translation, vision-language math, and medical risk prediction report gains over full-model GRPO fine-tuning while using far fewer trainable parameters and less memory.

### Strengths
Originality
- Proposes a clean, modular training objective mapping trajectory reward to token-level log-probabilities and uses additive logit fusion at inference which is a conceptually neat design.
- Inference-time composition via logit addition is derived from a KL-regularized multi-objective RL objective (Eq. 6), giving the method theoretical grounding absent in prior adapter ensembles.

Empirical Quality
- Consistent gains across 5 math benchmarks (up to +7.4 avg. over GRPO-full) and 2 translation directions (+1.8 BLEU) with ≤ 1.5 B trainable parameters.
- Strong weak-to-strong transfer: a 1.5 B module trained with a 3 B backbone improves a frozen 14 B Qwen-2.5 by +2.5 avg. points (Fig. 2).
- Composability demo: combining math + translation modules on German math→English task beats either specialist alone (Fig. 3).

Clarity & Reproducibility
- Complete hyper-parameter tables (App. A.1), reward definitions, and prompts (App. D) provided.
- Seeds and std-dev reported (10 runs, Tables 1–2).

Significance
- Enables cheap, plug-and-play specialization of proprietary or edge-deployed models without gradients through the backbone.

### Weaknesses
- Tokenizer-alignment prerequisite limits practicality. The paper claims “architecture-agnostic” but requires shared tokenizer (sec. 1, line 6). Is there a way to extend the method with tokenizer-mismatch adaptation strategy (e.g., embedding-level mapping layer trained with UniR) or can we quantify performance drop under vocabulary misalignment.

- The central identity (Eq. (4)), representing a trajectory reward as the sum of token log-probabilities of some policy, is assumed and only informally justified; Theorem 1 is informal and depends on strong assumptions (convergence of $\pi_\theta$ to $\pi_{\theta^*}$, uniqueness of optimum, and that π_r can exactly represent the reward) (Sec. 4.1 & 4.3). Can you provide a formal version of the theorem in the appendix with the details of the assumptions and proofs. The proof relies on the assumption that the guided policy has converged to the optimum. It would be beneficial to discuss the conditions under which this assumption holds. 

- The paper compares to GRPO variants and LoRA but does not compare to other recent inference-time guidance / token-level reward approaches (beyond brief GenARM mention).

- Missing Related Work: The paper fails to cite or compare against a relevant line of work on plug-and-play modules for frozen transformers. Specifically, TART ("TART: A Plug-and-Play Transformer for Zero-Shot Task Adaptation", Bhatia et al., ICLR 2022) introduces a nearly identical architectural paradigm: a small, separately trained module whose outputs are combined with a frozen backbone model to achieve task adaptation. While TART's training objective and application (zero-shot classification) differ, the core concept of a "plug-and-play reasoner for frozen LLMs" is not entirely new.

### Questions
Please see the weakness section above for relevant questions. In addition, 

- Can you show failure cases where composed modules hurt performance? What are the symptoms of negative interference?
- Code release plan -- will the code for this paper be released?
- Regarding composability, how do you envision setting the combination weights $\alpha_i$ in a real-world scenario without an expensive sweep for each new task combination? Have you considered methods for dynamically learning or setting these weights?

### Soundness
3

### Presentation
2

### Contribution
2
