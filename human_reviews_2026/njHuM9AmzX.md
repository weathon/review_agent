# Thought Purity: A Defense Framework For Chain-of-Thought Attack

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2

## Abstract
While reinforcement learning-trained Large Reasoning Models (LRMs, e.g., Deepseek-R1) demonstrate advanced reasoning capabilities in the evolving Large Language Models (LLMs) domain, their susceptibility to security threats remains a critical vulnerability. This weakness is particularly evident in Chain-of-Thought (CoT) generation processes, where adversarial methods like backdoor prompt attacks can systematically subvert the model’s core reasoning mechanisms. The emerging Chain-of-Thought Attack (CoTA) reveals this vulnerability through exploiting prompt controllability, simultaneously degrading both CoT safety and task performance with low-cost interventions. To address this compounded security-performance vulnerability, we propose Thought Purity (TP): a defense paradigm that systematically strengthens resistance to malicious content while preserving operational efficacy. Our solution achieves this through three synergistic components: (1) a safety-optimized data processing pipeline (2) reinforcement learning-enhanced rule constraints (3) adaptive monitoring metrics. Our approach establishes the first comprehensive defense mechanism against CoTA vulnerabilities in reinforcement learning-aligned reasoning systems, significantly advancing the security-functionality equilibrium for next generation AI architectures.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Thought Purity, a defense framework for large reasoning models (LRMs) against Chain-of-Thought attacks. Such attacks manipulate the intermediate reasoning process of models that produce multi-step explanations. The proposed framework integrates some elements: a safety-oriented data pipeline that labels suspicious reasoning segments (`<suspect>` and `<harm>`), a reinforcement learning stage based on Group Relative Policy Optimization (GRPO) with dual reward models (Outcome Reward Model and Process Reward Model). Experiments across four reasoning datasets and three model architectures show consistent improvements against BadChain-style attacks.

### Strengths
1. The paper addresses reasoning-stage vulnerabilities rather than only final outputs. This focus is increasingly important as chain-of-thought reasoning becomes standard in LLMs and LRMs.  
2. The experiments cover different architectures and task types.

### Weaknesses
1. The experiments focus entirely on the BadChain family of prompt-injection attacks. Although the authors vary injection locations and ratios, they do not test TP under different attack families.  

2. The same trigger token from BadChain (`@_@`) appears in both training and evaluation. The framework may therefore partially memorize the pattern instead of learning generalized reasoning hygiene.  


3. Only a few simple fine-tuning or RL baselines are compared. Existing safety defenses such as prompt sanitization or adversarial detectors are not included, which makes the relative advantage less clear.

### Questions
1. Have you conducted hold-out trigger or unseen target mapping experiments to evaluate generalization?  
2. Could the inserted `<suspect>` or `<harm>` tags cause false positives on clean reasoning data?

### Soundness
3

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
4

### Summary
This paper addresses the security vulnerabilities of Large Reasoning Models (LRMs), specifically focusing on their susceptibility to Chain-of-Thought Attacks (CoTA) that can subvert the model's reasoning process. The authors propose a new defense framework called "Thought Purity" (TP). This framework is built upon three main components: (1) a safety-optimized data processing pipeline that uses special tags (e.g., `<suspect>`, `<harm>`) to identify and segment malicious reasoning; (2) a reinforcement learning (RL) approach, specifically Group Relative Policy Optimization (GRPO), which is guided by both process-level and outcome-level rewards; and (3) adaptive monitoring metrics, including two new metrics proposed by the authors (Cure Rate and Reject Rate). The authors evaluate their framework on several reasoning datasets and LRM families, comparing it to a baseline "ORM-only" RL method.

### Strengths
1.  **Important and Timely Problem:** The paper addresses a critical and relevant issue. As models increasingly rely on complex, multi-step reasoning (like CoT), understanding and mitigating attacks against this process is a valuable area of research.
2.  **Sufficient Experimentation:** The authors have been thorough in testing their method across multiple datasets and model types, which provides a good breadth of evidence for their claims.

### Weaknesses
1.  **Marginal and Inconsistent Performance Gains:** This is the most significant concern. While the paper claims improvements, the empirical results shown in Table 1 are not consistently strong and the gains appear marginal. For instance, in several experiments, the defended model's ACC is not substantially improved (or is even slightly worse) than the original, and the reduction in ASR/ASRc is not always compelling. This calls into question the practical utility and robustness of the proposed framework.
2.  **High-Complexity Solution:** The TP framework introduces significant complexity. It requires a multi-stage data synthesis pipeline (generating CoT, simulating attacks, implanting special tags) followed by a full-scale RL training process. This high overhead is a major drawback, especially when the performance improvements are not overwhelmingly clear. A much simpler defense method might be preferred in practice.
3.  **Limited and Weak Baseline:** The primary baseline for comparison is an "ORM-only" model, which (as the authors show) performs very poorly, sometimes worse than the original undefended model. While this validates the authors' choice to include a PRM, it makes the TP framework's victory less impressive. The paper would be far stronger if it compared TP against other established (even if not CoTA-specific) defense methods for backdoor attacks or prompt injection, such as data filtering, simple SFT-based defenses, or other prompt-engineering-based defenses.
4.  **Overfitting to a Specific Attack:** The defense mechanism, particularly the `<harm>` tag, seems narrowly designed to counter the specific "redundant reasoning" attack pattern exemplified by the attack used in the paper. It is highly questionable whether this framework would generalize to more subtle or diverse forms of CoTA, such as attacks that slightly alter the direction of reasoning without adding an obvious, skippable "harmful" block.
5.  **Clarity and Presentation:** The paper's clarity could be improved. For example, the "Degree of Defense" (Equation 1) is formally introduced but does not seem to be explicitly measured or referenced in the experimental analysis, making its inclusion confusing. The analysis of the results could also be deeper, moving beyond a surface-level description of the metrics.

### Questions
Please see the weakness.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In this paper, the authors introduce Thought Purity, a defense that leverages GPT-4o-generated CoT data annotated with <suspect> and <harm> tags and trains LRMs via GRPO with dual outcome/process rewards to detect, skip, and recover from prompt-injected CoT (CoTA) attacks.

### Strengths
The paper focuses on reasoning models (LRMs) with CoT as a vulnerability surface rather than only LLMs and general prompt injection. That narrower scope is less studied (though there is work on CoT backdoors, as above). I do believe this scope should also be the focus of the researchers.

### Weaknesses
1. The underlying vulnerability (CoT prompting + backdoors/triggers) is already well documented (see BadChain, SABER, etc.). Doesn’t that make the novelty incremental rather than foundational?

2. I think the defense itself is not robust to adaptive attacks. If you are evaluating against Badchain with a fixed trigger "@_@", isn’t it a fatal flaw? An adaptive attacker can change the trigger (e.g., "cf" or natural language like "as per protocol") or vary injection timing or syntax.

3. Table 3 (Anti-TP) shows that reversing rewards already degrades performance. A real attacker would do far worse than flipping signs. Isn’t it possible for attackers to craft triggers to bypass your tags? I think the flaw here is the lack of ablation on trigger transferability.


I think the flaw here is the lack of ablation on trigger transferability. 

4. Looking into the cure rate and reject rate; I see the probability of Denominator instability: If ACC-clean−ACC-attack ~= 0 (near-perfect attack), CR blows up or becomes undefined. There are no confidence intervals or statistical tests.


5. If GRPO is being used to train the model to learn the tags, then why not use PPO, DPO, or another method? Is there a valid reason why only GRPO has been chosen? Can we achieve the same results with PPO or DPO? 

6. From my perspective, I only see the training with GRPO being done in addition to the insertion of <suspect>, <harm>…</harm> tags during training. But these tags are never seen at inference, correct? And the model must hallucinate them to activate defense? I believe that there is no proof that this happens reliably outside of your curated data.

7. The paper could have been stronger if the results have been further compared with other defenses; Self-Debate / Chain-of-Scrutiny (Li et al., 2024a), Paraphrasing input (Jain et al., 2024), Representation engineering (RAE) (Zou et al., 2024), Fine-tuning on refusal data (e.g., SafeRLHF)

I do believe that the paper significantly lacks novelty, which is why I provided the given score.


I do have few more questions:

Line 135: What is defense depth hierarchy 

Line 152: “CoTA simulate malicious users by implanting hidden system prompts, which impose fatal constraints on the model’s output normalization”.-- Can you please elaborate more on this?


Line 311: Can you please elaborate on this: “”The OutputRL-Llama model serves as the baseline defense. The design of this model stems from people’s simple wish: if ACC decreases, then enhance its ability until it rises.”””

Line 186 to 192: The language seems a tad off. Have the claims that have been made been tested?


Minor typo in line 165: “This” should be ‘this’

### Questions
Please see Weaknesses

### Soundness
1

### Presentation
2

### Contribution
1
