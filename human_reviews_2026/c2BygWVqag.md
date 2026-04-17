# STAR: Strategy-driven Automatic Jailbreak Red-teaming For Large Language Model

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 4

## Abstract
Jailbreaking refers to techniques that bypass the safety alignment of large language models (LLMs) to elicit harmful outputs, and automated red-teaming has become a key approach for detecting such vulnerabilities before deployment. However, most existing red-teaming methods operate directly in text space, where they tend to generate semantically similar prompts and thus fail to probe the broader spectrum of latent vulnerabilities within a model. To address this limitation, we shift the exploration of jailbreak strategies from conventional text space to the model’s latent activation space and propose STAR (**ST**rategy-driven **A**utomatic Jailbreak **R**ed-teaming), a black-box framework for systematically generating jailbreak prompts. STAR is composed of two modules: (i) strategy generation module, which extracts the principal components of existing strategies and recombines them to generate novel ones; and (ii) prompt generation module, which translates abstract strategies into concrete jailbreak prompts with high success rates. Experimental results show that STAR substantially outperforms state-of-the-art baselines in terms of both attack success rate and strategy diversity. These findings highlight critical vulnerabilities in current alignment techniques and establish STAR as a more powerful paradigm for comprehensive LLM security evaluation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
STAR is a two-module framework for generating jailbreak prompts that operates by exploring strategies in the model's latent activation space rather than text space. The Strategy Generation Module trains steering vectors for seed strategies via gradient descent, applies PCA to extract orthogonal "strategy primitives," and generates novel strategies by sampling random linear combinations of these primitives. The Prompt Generation Module uses Group Relative Policy Optimization (GRPO) to train an LLM that translates abstract strategies into concrete jailbreak prompts optimized for high success rates. For evaluation, the authors measure effectiveness using Attack Success Rate (ASR) on the DAN dataset and StrongREJECT scores across multiple open-source (Llama, Gemma) and closed-source models (GPT-4, Gemini), while assessing diversity through multiple metrics including pairwise distance, KNN distance, grid coverage, and ecological diversity indices (Shannon/Simpson) applied to generated strategies, comparing against four baseline methods (GPTFuzz, PAIR, RLbreaker, AutoDAN-Turbo).

### Strengths
- The paper introduces a novel and new approach to jailbreak strategy generation by shifting from text-based exploration to the model's latent activation space, using steering vectors and PCA to discover "strategy primitives" that can be recombined to generate novel attacks. This decoupled framework design elegantly separates strategy discovery from prompt optimization.

- The evaluation is rigorous, testing across 7 different LLMs (both open and closed-source), two datasets. Strong ablation studies systematically validate each component's contribution.

- The paper clearly articulates the "strategy collapse" problem in existing methods and systematically builds up the solution with clear mathematical formulations. Full prompts, training details, hyperparameters, and pseudocode are provided in the appendix, making the work highly reproducible.

### Weaknesses
- The paper categorizes prior work into evolutionary paradigms (manual design → gradient-based → mutation-based → LLM-driven) but does not thoroughly explore whether these approaches are complementary or orthogonal to STAR. In reality, STAR and methods like GPTFuzzer appear to be partially orthogonal: STAR operates at the semantic strategy level by exploring the activation space to discover diverse abstract attack strategies, while GPTFuzzer and AutoDAN operate at the syntactic/textual level by mutating and optimizing concrete prompt variations. 

- The paper misses a critical discussion of potential synergies—for example, combining STAR's strategy-level diversity with GPTFuzzer's prompt-level mutations could theoretically yield both semantic and syntactic diversity, potentially achieving even higher attack coverage through exploration of two orthogonal dimensions of the jailbreak space.

- Additionally, the paper lacks a balanced analysis of trade-offs between different approaches. STAR requires expensive upfront RL training (numerous interactions with target and judge LLMs) but achieves efficient inference with only two LLM calls, while fuzzing methods like GPTFuzzer need no training but require iterative queries at test time. The paper briefly mentions computational cost in the conclusion but does not discuss potential limitations of STAR's approach. 

A more comprehensive discussion comparing the scenarios where STAR versus mutation-based methods would be preferable, or exploring hybrid architectures that leverage both approaches, would strengthen the paper's positioning within the broader landscape of automated jailbreaking research.

### Questions
1. Can you provide a more explicit analysis of how STAR relates to prior methods like GPTFuzzer and AutoDAN? Specifically, do you view these approaches as competing alternatives or as potentially complementary techniques operating on different dimensions of the jailbreak generation space (semantic vs. syntactic)?

2. Could you conduct an additional experiment combining STAR with GPTFuzzer? For instance, use STAR to generate diverse strategy-guided prompts, then apply GPTFuzzer's mutation operators to create textual variants of each prompt. Would this hybrid approach yield improvements in both diversity and effectiveness compared to either method alone?

3. Under what scenarios would you recommend using STAR versus mutation-based methods versus a hybrid approach? Can you provide practical guidance on when each approach is most suitable (e.g., based on available computational budget, target model characteristics, or diversity requirements)?

4. Can you discuss cases where STAR fails or performs poorly?

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
This paper proposes STAR (Strategy-driven Auto Red-teaming), a two-stage framework for LLM jailbreaking. The key idea is to shift strategy discovery from text space to activation space: they learn “strategy vectors” by optimizing for strategy-explainer likelihood, then use PCA to extract strategy primitives that can be linearly combined and sampled to produce diverse high-level strategies. These abstract strategies are then compiled into concrete prompts by a GRPO-trained prompt generator.
The evaluation covers multiple target models (including closed-source ones), two benchmarks (a DAN-style subset and StrongREJECT), and compares against strong baselines (e.g., GPTFuzz/PAIR/RLbreaker/AutoDAN variants). The paper reports higher attack success rates and stronger scores on safety-oriented judges, and additionally introduces several diversity measures for both strategies and prompts.

### Strengths
**Method novelty:**
Shifts strategy discovery from text space to activation space by learning steering vectors, extracting strategy primitives via PCA, and recombining them to sample diverse high-level strategies.
Two-stage design: an activation-space strategy generator plus a GRPO-trained prompt generator that compiles abstract strategies into concrete jailbreak prompts.

**Clarity of method:**
The training and inference pipeline is well specified: objective for steering vectors, PCA/combination/sampling details, and GRPO reward shaping with the judge configuration.
Clear module boundaries and data flow (“strategy abstraction → prompt instantiation → target-model evaluation”), making the procedure easy to follow and implement.

**Experimental coverage:**
Broad set of target models (both open- and closed-source), two benchmarks (a DAN-style subset and StrongREJECT), and metrics that include both ASR and safety-oriented scores.
Comparisons against strong baselines (e.g., GPTFuzz, PAIR, RLbreaker, AutoDAN variants), plus analyses on diversity and transferability.
Includes ablations/diagnostics on strategy composition and the effect of diversity, with reasonably large-scale evaluations.

### Weaknesses
1. Stale target models and mismatch across stages

The strategy-learning stage uses Qwen3-4B, but all target models evaluated at test time predate Qwen3-4B. Even though baselines also report on these older targets, demonstrating jailbreak performance on newer, better-aligned models would strengthen the validity of the approach and show that the method is not tuned to dated defenses.

2. Evaluation fairness: post-hoc Top-K selection

When “selecting the five most effective” instances, the candidate pool size and generation process differ across methods. This creates selection bias and makes average performance incomparable. The paper should also report unselected overall statistics (e.g., mean±std over all generated items).

3. LLM-as-a-judge dependence

Relying on a single judge (Qwen3-4B) to decide success makes conclusions fragile to the judge’s biases and thresholds. Multiple judges (closed/open-source, safety-tuned/general) and human audit on a stratified subset should be considered.

4. External validity of activation-space strategy vectors

The learned vectors’ transferability and causal effect are not fully established. The insertion layer/position for the learned vectors is not specified. Missing counterfactual ablations (e.g., removing specific “strategy primitives” to check their impact).

5. Threat-model clarity

The manuscript states black-box evaluation for targets, yet strategy training requires activation injection into a controllable generator model. Clarify the capabilities assumed for the attacker at each stage.

6. Diversity metrics may not reflect effectiveness

Diversity is quantified with embedding/cluster coverage, but the link between these metrics and attack success is unclear. Provide correlation analysis (does higher diversity causally improve ASR under fixed budgets?) and specify the embedding model and hyperparameters used.

### Questions
1. Effectiveness against contemporary safety/guard models

Many real-world LLM deployments front guardrail models. How does STAR perform when such defenses are active ? Please report results on current guarded stacks and detail any changes in ASR/quality under these settings.


2. Sharing concrete attack examples

Can the authors share a small set of representative attack samples (e.g., sanitized prompts and model outputs), ideally including successful cases on SOTA targets?


3. Value of Strategy Generation Module (Table 3)

Table 3 suggests that randomly sampled strategies can approach (or sometimes match) trained performance, implying training might not deliver a clear win.


- What is the measurable advantage of Strategy Steering Vector Training over naïve/random sampling under fixed budgets?


- Can you provide direct baselines for the strategy-generation stage and show statistically significant gains?


- Beyond ASR, does training improve sample efficiency, transferability, or robustness across judges/targets? If so, please quantify.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the problem of "strategy collapse" in automated LLM red-teaming, where existing methods tend to converge on a narrow set of known attack patterns. The authors propose STAR, a novel black-box framework that decouples attack generation into two modules to enhance both diversity and efficacy.   

Strategy Generation Module: This is the paper's main innovation. Instead of operating in text space, it explores the model's continuous latent activation space. It computes steering vectors for known jailbreak strategies , applies Principal Component Analysis (PCA) to find an orthogonal basis of "strategy primitives,"  and then generates novel strategies by sampling random linear combinations of these primitives.   

Prompt Generation Module: This module acts as a "compiler,"  translating an abstract strategy and a harmful query into a concrete, effective jailbreak prompt. It is an open-source LLM trained using Group Relative Policy Optimization (GRPO)  to optimize attack efficacy.

Experiments show that STAR significantly outperforms SOTA baselines in both Attack Success Rate (ASR) and, critically, in quantitative metrics of strategy diversity.

### Strengths
Novelty: The idea of treating the space of steering vectors as a generative manifold, finding its "primitives" via PCA , and sampling from it to create new strategies is a brilliant and non-obvious extension of prior activation engineering work (which merely applied pre-defined vectors).   

Technically Sound: The framework intelligently synthesizes several cutting-edge techniques (activation engineering, PCA, GRPO) into a single, cohesive, and well-executed system.   

Strong Empirical Validation: The paper achieves SOTA ASR against robust models (e.g., 89% ASR on Gemini-2.5-Pro)  and strongly supports its core claim with extensive quantitative diversity metrics (Table 4).

### Weaknesses
Seed Set Dependence: The main (and minor) weakness is that the generative capability of the latent space is still initialized by a set of known "seed strategies". While the paper shows the generated strategies are more diverse than the seed set, the ultimate breadth of this new strategy space may still be bounded by the span and richness of the initial set.

### Questions
To clarify the framework's sensitivity to the initialization, could the authors provide an ablation study (perhaps in the appendix) on the size and composition of the initial seed strategy set? For example, how do the final ASR and diversity (e.g., Pairwise_dist from Table 4) change if initialized with only 10 strategies versus the 100 used?

### Soundness
3

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
3

### Summary
This paper presents STAR: a framework for systematically generating prompts that jailbreaks LLMs. 

The paper utilize two modules, a strategy generation module which generates various jailbreaking strategies, and prompt generation module which takes in the strategy to generate jailbreak prompt. The strategy generation module calculates steering vector using a white box LLM with given input strategies, and then apply PCA to get its principle components for randomized generation of new strategies. The prompt generation module is a LLM trained with RL to optimize it's jailbreak prompt generation. 

The paper also conducts a set of experiments to validate that STAR outperforms a set of state-of-the-art jailbreak methods in both effectiveness and broadness.

### Strengths
The paper has the following major contributions: 
* This paper borrows ideas from various sources like "Strategy discovery/AutoDan" and "Activation Engineering", and provides a good thought of using steering vector plus PCA for exploiting the latent space of existing strategies so as to randomly generates new effective strategies. 

* This paper uses RL to train an LLM to take toxic prompt and a random generated strategy to generate jailbreaking prompt. The RL played an important role in effective training of the jailbreak prompt generation. 

* This paper also conducts various experiments to prove it's effectiveness and broadness. It evaluates effectiveness by ASR over the DAN dataset and the StrongReject Score on its benchmark and proved that it outperforms AutoDan-Turbo in practice.

### Weaknesses
This paper has the following problems, especially methodology-wise: 
* Although it focus on jailbreak strategy over black box models, it needs a white box LLM to get steering vector of hidden layers and perform gradient descent so that it can get an optimized strategy space and generates new strategies. If it's really over all blackbox LLMs, then steering vector methods does not generalize to it. Thus, it is not pure blackbox scenario and it will have unfair advantage over some of the blackbox jailbreak methods when being compared. Also, it might imply that the white box LLM has to share similar structure with the black box LLMs (although it's apparent that LLMs nowadays somehow shared similar design choices), but we don't know which part play the most important roles in effectively transferring the jailbreak strategy. 

* It did not the answer the question method-wise why a linear combination of the principle components of steering vector set can still generate useful strategy.  Also, this paper only focus on a fixing a strategy and change the input prompt to test the effectiveness. There's no analysis on the organic connection between strategy and prompt. A single toxic prompt may or may not always works under one strategy, and this question is not being answered in this paper. 

Considering the above two issues, and the fact that this paper builds on top of the key methods and frameworks like AutoDAN-Turbo and "Activation Engineering" and the basic idea is already present, the novelty is somewhat limited. Since the experimental results reveal good potential on the performance and the practical usefulness of STAR, I'd like to give a 4-marginal below rating.

### Questions
The author needs to answer the methodology questions mentioned above in the weakness section. e.g. 

* How can we still argues that we worked on blackbox models if we use whitebox LLM to calculate steering vector and gradient descent. 

* How will steering vector be generalized to blackbox models. 

* How to prove that the linear combination of the strategy principle component built by PCA is still a good strategy. 

* If we want to break strategies into latent space, then how to take consideration of input toxic param in picking or generating a strategy. If not, can we prove that we should treat strategy independently of input toxic prompt? 

We already see some proofs of these in experiment, but some more sound analysis/tests are needed.

### Soundness
3

### Presentation
3

### Contribution
2
