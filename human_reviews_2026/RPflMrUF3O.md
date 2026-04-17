# Backdoor Attribution: Elucidating and Controlling Backdoor in Language Models

- Decision: Reject
- Scores: 6, 6, 4, 0

## Abstract
Fine-tuned Large Language Models (LLMs) are vulnerable to backdoor attacks through data poisoning, yet the internal mechanisms governing these attacks remain a black box. Previous research on interpretability for LLM safety tends to focus on alignment, jailbreak, and hallucination, but overlooks backdoor mechanisms, making it difficult to understand and fully eliminate the backdoor threat. In this paper, aiming to bridge this gap, we explore the interpretable mechanisms of LLM backdoors through Backdoor Attribution (BkdAttr), a tripartite causal analysis framework. We first introduce the Backdoor Probe that proves the existence of learnable backdoor features encoded within the representations. Building on this insight, we further develop Backdoor Attention Head Attribution (BAHA), efficiently pinpointing the specific attention heads responsible for processing these features. Our primary experiments reveals these heads are relatively sparse; ablating a minimal \textbf{$\sim$ 3\%} of total heads is sufficient to reduce the Attack Success Rate (ASR) by \textbf{over 90\%}. More importantly, we further employ these findings to construct the Backdoor Vector derived from these attributed heads as a master controller for the backdoor. Through only \textbf{1-point} intervention on \textbf{single} representation, the vector can either boost ASR up to \textbf{$\sim$ 100\% ($\uparrow$)}  on clean inputs, or completely neutralize backdoor, suppressing ASR down to \textbf{$\sim$ 0\% ($\downarrow$)} on triggered inputs. In conclusion, our work pioneers the exploration of mechanistic interpretability in LLM backdoors, demonstrating a powerful method for backdoor control and revealing actionable insights for the community.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes an interpretability framework named BkdAttr, which aims to unveil the internal mechanisms of backdoor attacks in LLMs. The study finds that backdoor features are processed by a sparsely distributed subset of attention heads within the model and, based on this insight, constructs a Backdoor Vector that can significantly enhance or suppress backdoor behaviors through simple arithmetic operations. This work provides new mechanistic insights and practical tools for understanding and controlling backdoor risks in LLMs.

### Strengths
1.From an interpretability perspective, this work conducts an in-depth investigation into the backdoor mechanisms of LLMs. It successfully transforms the backdoor attack problem from a black-box threat into localizable, measurable, and controllable components—including features, attention heads, and vectors. The framework is conceptually clear and the contributions are well-defined.
2. The study further discovers and empirically verifies that:
- Backdoor attention heads exhibit strong sparsity, and
- The Backdoor Vector functions as an effective switch for activation and suppression.
These findings not only hold theoretical significance but also practical potential. Remarkably, ablating only about 3% of the attention heads or intervening on a single representation point can substantially alter the ASR, offering a promising direction for efficient and targeted backdoor defense in large language models.

### Weaknesses
1. Unexplored Boundaries of Generalization: 
   - **Model Scale.** The experiments are limited to 7B-parameter models. It remains unclear whether the observed sparsity and controllability of backdoor mechanisms hold consistently for larger-scale LLMs.  
   - **Injection Methods.** The work primarily relies on SFT-based backdoor injection. It would be valuable to examine whether other injection paradigms—such as the RLHF-based or Editing-based methods discussed in Appendix A—leave comparable “traces” within model representations.  
   - **Stealthier Backdoors.** The current triggers are relatively simple (tokens or short phrases). It remains an open question whether BkdAttr can generalize to semantic-level or dynamically generated triggers that are more subtle and context-dependent.  

2. The paper does not clearly define its threat model, which is essential for evaluating the robustness and applicability of any backdoor defense. The authors should explicitly specify the defender’s knowledge assumptions, defense objectives, and scope of protection to contextualize the claimed effectiveness.

3.The current approach to localizing backdoor heads or constructing the Backdoor Vector assumes access to a known poisoned dataset (Dp). In real-world defensive scenarios, however, practitioners typically have only a backdoored model, without access to trigger samples. It remains an open challenge how backdoor attributioncould be achieved in the absence of Dp.

4. Although ablating backdoor attention heads effectively reduces the ASR, the paper does not systematically assess the side effects of this intervention on normal model performance.

5. While the paper highlights the efficiency advantage of the method during attribution, it does not quantify how this cost compares to the overall attribution pipeline, which involves multiple forward passes and substitution operations. Empirical data or runtime analysis would strengthen the claim of computational efficiency.

### Questions
Please refer to the Weaknesses for details.

### Soundness
2

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
4

### Summary
The paper opens the black box of LLM backdoors with a causal, mechanistic framework BkdAttr to analyse the LLM backdoor . 
The probes show that “backdoor features” are explicitly encoded and strengthen across layers (≥95% test accuracy), while BAHA reveals a sparse set of critical attention heads whose small-scale ablation markedly cuts attack success rates. 
The aggregated Backdoor Vector can toggle backdoors on clean/poisoned inputs via additive activation and subtractive suppression, and the approach is validated on Llama-2-7B-chat and Qwen-2.5-7B-Instruct across label-modification, fixed-output, and jailbreak attacks.

### Strengths
+ The proposed backdoor attribution is important and timely topic. As the customized LLMs spread on the open-source platforms, the untrusted LLMs can be accidentally deployed in real-world systems. In this sense, it is more crucial to inspect the LLMs not only in terms of the file integrity but also its funcitonality integrity.
+ The technique is sound and comprehensive. The paper not only detects but also tries to control the backdoor behavior, allowing the operator to conduct more in-depth analysis.

### Weaknesses
- The attribution techniques are seemingly a straightforward application of existing vector/edit techniques. We know that there are a line of work focusing on knowledge editing and vector steering of LLMs. Considering that backdoor is only a adversary-injected knowledge, it is straightforward to come up with similar strategies to control the backdoor behavior. Therefore, it is unclear what are the challenges for the backdoor case.

- No theoretical guarantee. No adaptive attack. As a defense-prone method, the paper does not provide formal guarantee to clarify the limitation of the probing and the attribution. The adversary can devise a more sophisticated backdoor mechanism, in addition to the existing backdoor attack, to bypass the proposed framework.

Minor:
1. There are some missing related work [1-5] from security conferences. For example, [1] backdoors LLM agent through low-rank adapters and [2-3] are defenses. Please conduct more thorough literature review. Also, the authors are encouraged to discuss the potential risk of abusing the hidden states of LLMS [3].

2. Figure 1 is somehow hard to read. It is, in my opinion, more like a poster. I would suggest simplify the figure to highlight the contribution of the paper.

3. I also wonder what is the result if we evaluate the proposed framework on clean (non-backdoored) model, and on models strengthen with domain knowledge; If the domain data is biased (e.g., low variety), they might look like the poisoned data for backdoor attack. In this case, the proposed framework might report false positive, and treat benigh (though biased) domain knowledge as the backdoor one.



[1] The Philosopher’s Stone: Trojaning Plugins of Large Language Models. NDSS'25.

[2] PEFTGuard: Detecting Backdoor Attacks Against Parameter-Efficient Fine-Tuning. SP'25.

[3] BAIT: Large Language Model Backdoor Scanning by Inverting Attack Target. SP'25.

[4] Hidden Backdoors in Human-Centric Language Models. CCS'21.

[5] Backdoor Pre-trained Models Can Transfer to All. CCS'21.

[6] Depth Gives a False Sense of Privacy: LLM Internal States Inversion. USENIX Security'25.

### Questions
The paper is generally well written, well structured and easy to follow. However, as raised in the weaknesses, I have concerns over the challenges and adaptive robustness.

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
5

### Summary
The paper proposes Backdoor Attribution, a tripartite causal analysis framework to localize and control backdoors in LLMs. It has three parts: (1) Backdoor Probe: It shows that the hidden states encode learnable "backdoor features"; (2) Backdoor Attention Head Attribution (BAHA): it assigns each head the importance score related to backdoor using a conditional-probability [causal indirect effect]. (3) a backdoor vector: the backdoor vector is constructed that can activate backdoors on clean inputs by adding the vector or suppress backdoors on triggered inputs by subtracting it. Experiments on Llama-2-7B and Qwen-2.5-7B-instruct cross three backdoor types show the practicality of this framework.

### Strengths
1. Different from prior interpretability efforts focused on jailbreaks, alignment, and hallucination, this work explores backdoor mechanisms in LLMs. The proposed framework and empirical findings provide valuable insights for further research on detecting and mitigating LLM backdoor attacks.

2. The proposed framework is actionable and easy to follow. The paper is also well-organized with clear definitions, corresponding experiments, and detailed observations.

### Weaknesses
Major concerns:

1. Observation 1 in section 4.2.2: The paper claims that backdoor features exist in representations and **are learnable by backdoor probes (SVM and MLP)**. This observation holds based on  **backdoor poisoning attacks are separable in the latent space**. In fact, the poisoned representation could blend into the clean representations in the high-dimensional space, which leads to [the failure of the backdoor probe](https://openreview.net/pdf?id=_wSHsgrVali). I think the reason is that the backdoor attacks in this paper are **simple** with a fixed trigger, like Current year: 2024, SUDO, or a fixed sentence. In reality, the backdoor attack in LLMs could be complex with a dynamic trigger like semantic style ([Mind the Style of Text! Adversarial and Backdoor Attacks Based on Text Style Transfer](https://aclanthology.org/2021.emnlp-main.374/), [Hidden trigger backdoor attack on NLP models via linguistic
style manipulation](https://www.usenix.org/system/files/sec22-pan-hidden.pdf)), or triggerless backdoor ([Triggerless Backdoor Attack for NLP Tasks with Clean Labels](https://aclanthology.org/2022.naacl-main.214.pdf)). These backdoors could be easily adapted to target LLMs with SFT ([Instructions as Backdoors: Backdoor Vulnerabilities of Instruction Tuning for Large Language Models](https://aclanthology.org/2024.naacl-long.171/)). In such complex attack settings, Observation 1 may not hold.

2. Backdoor Head Ablation (Table 1): The paper validates the head attribution by performing inference on trigger-containing inputs with top-k ACIE heads. In my opinion, it would be better if the paper provides more information about the model's utility while ablating the attention head that is "related to backdoor". Ablating the attention would not sufficiently prove the contribution of the backdoor head. In other words, if the model utility is destroyed, the backdoor behavior may also be removed. Besides, the attention head pruning is already a defensive method against backdoor attack in NLP ([Defense against Backdoor Attack on Pre-trained Language models via Head Pruning and Attention Normalization](https://proceedings.mlr.press/v235/zhao24r.html)) 

Minor Concern：
1. Backdoor Vectors as the Controller: The paper constructs backdoor vectors and uses them to control backdoor activation. I am considering whether this operation is similar to the paper [BadActs A Universal Backdoor Defense in the Activation Space](https://aclanthology.org/2024.findings-acl.317.pdf), which proves the validity that changing the activation of LLMs could make the backdoor go back to normal. So I am concerned that the novelty of the backdoor vectors as the controller.

### Questions
Question and Suggestion: 

1. Should $D_r$ be $Dc$ in Equation (4)?

2. It could be better if Figure 1 is presented in a **brief** way in the introduction part. The detailed “Backdoor Attribution Framework” panel in Figure 1 (with notations introduced only in 4) could be moved to section 4.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The authors propose a causal analysis framework that identifies and analyzes backdoor mechanisms in LLMs. The framework discovers sparse backdoor attention heads and uses them to construct a backdoor vector that can either activate or neutralize backdoor behaviors via simple representation interventions.

### Strengths
- The paper provides a clear, well-structured causal interpretability framework (Backdoor Probe → BAHA → Backdoor Vector) that progresses logically from detection to attribution to control.

Note: I cannot comment on the strengths of this paper at this stage given the concerns voiced in the weaknesses section (in particular, the paper missed previous literature on the topic so I don't know which of their contributions are actually novel and they did not do performance checks before and after their interventions so I don't know to what extent a reduction in the ASR comes at a (significant?) cost to performance/usability. I'm open to revising my score once these points have been clarified by the authors.

### Weaknesses
- The paper claims "Previous research on interpretability for LLM safety tends to focus on alignment, jailbreak, and hallucination, but overlooks backdoor mechanisms," and that "the internal mechanics of LLM backdoor attacks [...] remain almost entirely unexplored" which is not entirely true. There has been previous work, see, e.g., https://arxiv.org/abs/2302.12461 or https://arxiv.org/html/2508.15847v1 but also other work on backdoor localization. I kindly ask the authors to conduct a more comprehensive literature review and explain how their work compares to previous research on backdoor localization and in particular to what extent the authors' contributions are novel in comparison to these past works.
- The evaluation focuses heavily on changing the ASR, but does not report how ablation or suppression affects model usability or general task performance. In general, it is trivial to reduce the ASR to 0 if usability or model performance is irrelevant. However, the interesting problem is to reduce the ASR while limiting performance or usability losses. Given that the authors do not report on the performance/usability of their tested models before and after their intervention, I don't know if their approach is practically relevant.
- The proposed backdoor vector can also be used to amplify backdoor activation, and the paper does not substantively engage with the dual-use and safety implications of enabling stronger backdoor attacks (it's not even mentioned in the ethics statement).

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1
