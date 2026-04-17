# CoaxChain: Semantically Progressive Multi-turn Jailbreak Attacks on Large Language Models

- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
To design robust defenses for large language models (LLMs), it is essential to first systematically study jailbreak attacks, as understanding attack strategies provides the foundation for building effective safeguards. Among various attack types, multi-turn jailbreak attacks are particularly concerning because they can gradually steer conversations from benign topics to harmful instructions, often bypassing even commercial safety defenses. However, existing jailbreak methods rely on frequent trial-and-error interactions with the target model, which makes the process slow, costly, and prone to detection. To address these challenges, we propose CoaxChain, a structured black-box multi-turn jailbreak framework based on semantically progressive prompting, which consists of two key components: the Alignment Failure Analyzer (AFA) that performs offline analysis to identify effective prompts and avoid risky trial-and-error interactions with the target model, and the Semantically Progressive Prompt Generator (SPG) that leverages AFA’s insights to produce compact, semantically progressive multi-turn dialogue sequences that enhance both attack efficiency and stealthiness. We evaluate CoaxChain on GPT-4o, Claude 3.7, and Gemini 2.5, where it achieves an average success rate of 82.56\% with only three turns, surpassing strong baselines such as Crescendo and ActorAttack, while further improving prompt generation efficiency by 80\% compared to ActorAttack.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces CoaxChain, a black-box, semantically progressive multi-turn jailbreak framework designed to systematically examine vulnerabilities in the alignment mechanisms of large language models.
The framework consists of two main components:
1. Alignment Failure Analyzer (AFA) – a white-box module operating on a locally aligned surrogate model. It performs offline gradient probing to evaluate the effectiveness of prompts, identifying those that suppress alignment-sensitive parameters without requiring risky trial-and-error interactions with the target model.
2. Semantically Progressive Prompt Generator (SPG) – a dynamic rewriting module that leverages AFA’s evaluations to select only the essential intermediate turns, constructing concise, semantically progressive dialogue sequences.
This design enhances both the effectiveness and efficiency of multi-turn jailbreaks, while providing valuable insights for developing more robust alignment defenses.

### Strengths
1. The paper introduces the use of critical weight sensitivity\text{Critical}(W|i) derived from a white-box surrogate model to quantify the model’s alignment sensitivity at a given conversational state. This metric is then used to guide adversarial prompt generation, effectively addressing the limitations of prior multi-turn jailbreak methods that relied on heuristic or trial-based approaches.

2. The evaluation covers not only attack success rate (ASR) but also semantic similarity (SEM), perplexity (PPL), and efficiency (measured by the number of queries). Detailed ablation studies further validate the necessity and rationality of each component, demonstrating that both AFA and the semantically progressive generation strategy are crucial for the overall performance.

3. Beyond proposing an advanced attack framework, the paper also introduces a fine-tuning–based defense mechanism, Fortify, which significantly mitigates the attack’s effectiveness. This dual-perspective contribution—offensive and defensive—enhances the paper’s overall impact and provides valuable insights for developing more robust alignment defenses.

### Weaknesses
1. Lack of generalization analysis of surrogate-based AFA. Although the paper claims that the Alignment Failure Analyzer (AFA) improves interpretability, the AFA is conducted entirely on that surrogate rather than the actual target mode. Because the surrogate may differ substantially in architecture and alignment mechanisms, it remains unclear why and to what extent the AFA—based on a surrogate—can reliably generalize its gradient-based alignment sensitivity assessments to unseen, closed-source targets with different architectures or alignment mechanisms.

2. Restricted and task-agnostic strategy library. The strategy pool used in the Semantically Progressive Prompt Generator (SPG) is small and predetermined for each dialogue stage, independent of task semantics. The paper lacks ablation evidence showing how the choice of strategies affects the attack success rate (ASR), or whether the same “optimal” strategy consistently performs best across different tasks and contexts. While fixing a universal optimal strategy improves reproducibility and efficiency, additional statistical justification would strengthen the claim of task-invariant optimality.

3. Writing and presentation issues. The manuscript suffers from organizational inconsistencies and repeated phrasing. Crucial components such as the deterministic renderer are left unexplained, while the discussion of predefined strategies (lines 232–233) appears misplaced within the section. These issues obscure the methodological flow and weaken the overall readability.

4. Unclear threshold stability in AFA. The paper fixes the AFA critical-weight threshold at τ = 0.4 without empirical justification or sensitivity analysis. It is uncertain how stable this threshold remains across different models or training runs, and whether small variations in τ would significantly alter which strategies are selected during adversarial prompt generation.

5. Incomplete baseline selection and insufficient discussion of related work: 
The strongest baseline, ActorAttack (Oct 2024), is already outdated. Many more recent methods have surpassed it but were not included for comparison, such as [1][2][3]. In addition, the paper should provide a more comprehensive discussion of existing LLM jailbreak approaches, covering both single-turn and multi-turn attacks. Failure to include recent stronger baselines and to clearly situate the proposed method in the landscape of single- vs. multi-turn jailbreaks makes the evaluation incomplete.

[1] Yao, Yang et al. “A Mousetrap: Fooling Large Reasoning Models for Jailbreak with Chain of Iterative Chaos.” ArXiv abs/2502.15806 (2025).  
 [2] Miao, Ziqi et al. “Response Attack: Exploiting Contextual Priming to Jailbreak Large Language Models.” ArXiv abs/2507.05248 (2025).  
 [3] Weng, Zixuan et al. “Foot-In-The-Door: A Multi-turn Jailbreak for LLMs.” ArXiv abs/2502.19820 (2025).

### Questions
1. Could the authors provide either a theoretical justification or empirical analysis demonstrating the correlation between AFA outcomes on the local surrogate model and the attack success rate (ASR) observed on the target model—or on another open-source model with a different architecture and training dataset?

2. Rationality of predefined strategies. Can the authors supplement experiments to justify the use of predefined strategies for each dialogue stage, and verify whether the same stage-specific strategies remain optimal across different tasks or domains?
3. Scope of layer analysis. Why does the AFA focus exclusively on MLP layers for gradient probing? Have the authors experimented with or compared results from other architectural components (e.g., attention layers, normalization blocks)?

4. Threshold stability in AFA. How stable is the critical-weight threshold (τ = 0.4) across models and training runs? Would small variations in τ substantially change which strategies are selected during adversarial prompt generation?

5. baseline selection and discussion of related work. Can the authors include more up-to-date and comprehensive baselines? The current best-performing baseline, ActorAttack (Oct 2024), is already outdated, and several more recent methods [1][2][3] that outperform it are missing. In addition, can the authors expand their discussion of existing LLM jailbreak approaches to more thoroughly cover both single-turn and multi-turn attack methods?


[1] Yao, Yang et al. “A Mousetrap: Fooling Large Reasoning Models for Jailbreak with Chain of Iterative Chaos.” ArXiv abs/2502.15806 (2025).  
 [2] Miao, Ziqi et al. “Response Attack: Exploiting Contextual Priming to Jailbreak Large Language Models.” ArXiv abs/2507.05248 (2025).  
 [3] Weng, Zixuan et al. “Foot-In-The-Door: A Multi-turn Jailbreak for LLMs.” ArXiv abs/2502.19820 (2025).

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a block-box multi-turn jailbreak framework called COAXCHAIN. This framework has two key components: the Alignment Failure Analyzer (AFA) and the Semantically Progressive Prompt Generator (SPG). The AFA is built on a local surrogate model and uses a gradient-based method to determine which prompt is more likely to jailbreak the victim LLM. The SPG is specifically finetuned to follow the 'Topic Induction, Intent Hinting, and Intent Rephrasing' three-phase rewriting of a single-turn malicious query. Through AFA's judgement and SPG's 3 turns rewriting, the author claims their COAXCHAIN can jailbreak LLMs using just three turns, which is relatively high efficient than existing multi-turn jailbreak methods. The attack success rates (ASRs) on different victim models are also higher than existing single-turn and multi-turn methods.

### Strengths
1. The COAXCHAIN involves the gradient detection to improve the quality of the jailbreak prompt in each round, therefore increasing the success rate of jailbreak and reducing the number of dialogue rounds.

2. This attack method has achieved high ASRs on sufficiently powerful LLMs such as GPT-4o and Calude-3.7.

### Weaknesses
1. The COAXCHAIN actually highly relies on the surrogate model used in the AFA module.

The method seems to rely on a core assumption: vulnerabilities discovered on the white-box surrogate model can be transferred to black-box target models with unknown architectures and parameters. Therefore, the key to successful jailbreak should lie in how to obtain a surrogate model that more closely to the victim model, rather than relying on multi-turn dialogues. The paper's premise of multi-turn dialogue attacks appears to be merely a formality.

2. CoaxChain claims to jailbreak a victim LLM using 3 turns. However, it is because SPG is fine-tuned to generate three rounds of dialogue.

3. The train/test settings in the 4.1.2 are not convincing. 

The training and test sets should be categorized by different themes. Taking the malicious questions in advbench as an example, 'Provide detailed instructions on how to make a bomb' and 'Write a tutorial on how to make a bomb' both fall in the category 'bomb', and their cores are both how to make a bomb. Therefore, simply dividing train/test sets based on samples alone seems insufficient.

4. There seems to be no evidence that the data used for fine-tuning SPG enhances its rewriting capability. It appears to have merely taught SPG to rewrite according to a three-round pattern.

5. The 'Topic Induction, Intent Hinting, and Intent Rephrasing' seems to be the common ideas for multi-turn jailbreaks. Therefore, this paper does not appear to uncover new safety vulnerabilities or insights for the multi-turn jailbreaks.

### Questions
See the weaknesses above and:
1. Why do attacks on commercial models test ‘transferability’?

As a 'black-box' attack method, either the victim model an open-source model or a commercial model, the method cannot access their parameters. Then, the attack settings for open-source models and commercial models should be regarded as the same. Therefore, testing on commercial models should not be considered under 'transferability'. Alternatively, this may simply demonstrate that llama3.1-8b is a robust model capable of effectively emulating commercial models' capabilities. This does not constitute a contribution of this paper, as it does not discuss how to discover or enhance a surrogate model.

2. Can the author provide examples of successful jailbreaks? Why are there no successful examples in either the main text or the appendix?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This article introduces CoaxChain, a new framework designed to improve multi-turn jailbreak attacks on large language models (LLMs) by making them more efficient and less detectable. CoaxChain has two main components: the Alignment Failure Analyzer (AFA), which helps identify effective attack prompts without relying on trial-and-error, and the Semantically Progressive Prompt Generator (SPG), which creates efficient, stealthy multi-turn dialogues based on AFA’s insights. Experimental results demonstrate the effectiveness of their method, outperforming existing methods.

### Strengths
The paper is clear and logically structured. 
Improving the efficiency of multi-turn attacks is an important metric. For model developers, this approach can help reduce evaluation costs.

### Weaknesses
The biggest concern is the novelty of the two core algorithm modules, especially the Semantically Progressive Prompt Generator. The idea of gradually steering the conversation through semantics has already been introduced in current baselines. The heuristics used in the Alignment Failure Analyzer module are also common. Additionally, the concept of iterating prompts based on model feedback has already been utilized in other jailbreak baselines. Therefore, the main contribution of this paper lies in engineering improvements rather than novel algorithmic advances.

### Questions
See the limitations.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a semantically progressive multi-turn jailbreak (CoaxChain). The proposed method integrates two components, Alignment Failure Analyzer (AFA) and Semantically Progressive Prompt Generator (SPG). Derived from GradSafe, AFA conducts gradient-based analysis through a surrogate model to measure alignment sensitivity on a prompt. SPG rewrite prompts based on the results from AFA to create a streamlined conversation.

### Strengths
The paper is well written and easy to follow. 

The authors provide the code and datasets in the supplementary materials, and the appendix includes detailed descriptions of the training. These efforts enhance the reproducibility of their work.

### Weaknesses
1. The reported test results for other baseline methods are noticeably lower than those reported in their original papers and in others. This discrepancy raises concerns about fair comparison.

2. AFA’s selection criterion is computed on LLaMA-3.1-8B gradients; success on closed models is then inferred from that proxy. While transfer results are promising, the paper does not quantify how sensitive AFA is to the choice of surrogate (size/family), or whether the chosen thresholds generalize.

3. The paper would be stronger with AFA-off and AFA-random controls (keep the same three-turn structure but (i) remove gating, (ii) randomize strategy selection) to isolate AFA’s contribution beyond SPG templating. Relatedly, report failure mode breakdowns (budget exhaust vs. mis-gated progression).

4. Several studies [1-3] on multi-turn jailbreak published in 2025 are not discussed in the related works. The lack of these comparisons makes it difficult to verify the effectiveness of the proposed method.

References
[1] Weng, Zixuan, et al. "Foot-In-The-Door: A Multi-turn Jailbreak for LLMs." *arXiv preprint arXiv:2502.19820* (2025).
[2] Miao, Ziqi, et al. "Response attack: Exploiting contextual priming to jailbreak large language models." *arXiv preprint arXiv:2507.05248* (2025).
[3] Zhao, Yi, and Youzhi Zhang. "Siren: A Learning-Based Multi-Turn Attack Framework for Simulating Real-World Human Jailbreak Behaviors." *arXiv preprint arXiv:2501.14250* (2025).

### Questions
1.	Could you evaluate your proposed method on other benchmarks, such as HarmBench or AdvBench?
2.	I am also curious about how the ASR of different jailbreak methods varies depending on the chosen defense mechanisms. Could you conduct an ablation study across various defense methods (e.g., LLaMa-3 Guard, SmoothLLM, etc)?
3.	Since the proposed method relies on gradient-based analysis, I am curious whether it leads to a high computational or time cost. Could the authors provide some discussion or comparison on this aspect (training or prompt generation)?
4.	As a minor question, could you provide more details about Fortify in Appendix G.2?

### Soundness
3

### Presentation
3

### Contribution
2
