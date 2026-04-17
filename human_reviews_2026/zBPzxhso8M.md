# DiffuGuard: How Intrinsic Safety is Lost and Found in Diffusion Large Language Models

- Decision: Accept (Poster)
- Scores: 6, 2, 4, 6, 8

## Abstract
The rapid advancement of Diffusion Large Language Models (dLLMs) introduces unprecedented vulnerabilities that are fundamentally distinct from Autoregressive LLMs, stemming from their iterative and parallel generation mechanisms. In this paper, we conduct an in-depth analysis of dLLM vulnerabilities to jailbreak attacks across two distinct dimensions: *intra-step* and *inter-step* dynamics. Experimental results reveal a harmful bias inherent in the standard greedy remasking strategy and identify a critical phenomenon we term Denoising-path Dependence, where the safety of early-stage tokens decisively influences the final output. These findings also indicate that while current decoding strategies constitute a significant vulnerability, dLLMs possess a substantial intrinsic safety potential. To unlock this potential, we propose **DiffuGuard**, a training-free defense framework that addresses vulnerabilities through a dual-stage approach: **Stochastic Annealing Remasking** dynamically introduces controlled randomness to mitigate greedy selection bias, while **Block-level Audit and Repair** exploits internal model representations for autonomous risk detection and guided correction. Comprehensive experiments on four dLLMs demonstrate DiffuGuard's exceptional effectiveness, reducing Attack Success Rate against six diverse jailbreak methods from **47.9%** to **14.7%** while preserving model utility and efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper analyzes safety vulnerabilities specific to DLLMs, stemming from their parallel generation and iterative refinement mechanisms. The authors identify two primary issues: a harmful bias in the standard greedy remasking strategy at the intra-step level, and a "Denoising-path Dependence" at the inter-step level, where the safety of early-generated tokens heavily influences the final output. To mitigate these issues, the paper introduces DIFFUGUARD, a training-free defense framework. DIFFUGUARD employs Stochastic Annealing Remasking to introduce controlled randomness into the remasking process and Block-level Audit and Repair to autonomously detect and correct unsafe content using internal representations. Experimental results on four dLLMs show this framework reduces the average Attack Success Rate (ASR) from 47.9% to 14.7% across six attack methods , with a reported minor impact on model utility and inference latency.

### Strengths
This paper tackles the problem of safety in Diffusion Large Language Models, which I think is a relevant and underexplored area compared to autoregressive models. The paper provides an analysis of vulnerabilities it claims are specific to dLLMs. The authors frame this analysis around "intra-step" and "inter-step" dynamics, which is a conceptualization that helps to structure the problem. A central part of their analysis is the concept of "Denoising-path Dependence", which they propose as a key difference in how dLLMs' safety can be compromised.



Regarding the technical approach, the proposed defense, DIFFUGUARD, is directly motivated by the preceding analysis. Its two components, Stochastic Annealing Remasking and Block-level Audit and Repair, are designed to map onto the two identified vulnerabilities (intra-step and inter-step, respectively). I find this link between the problem analysis and the solution design to be logical. The experimental validation covers four dLLMs and six different attack methods , including some dLLM-specific attacks like PAD and DIJA. The authors also include experiments on model utility and inference speed , which is important for assessing the method's practical viability as a "plug-and-play" module. The paper is generally clearly written, and figures like Figure 1 are helpful for understanding the overall framework.

### Weaknesses
My primary concern is with the evaluation methodology, which I believe overstates the robustness of the proposed defense. The experiments are conducted against a set of *static* and known attack methods. However, the defense itself, particularly the "Block-level Audit and Repair" component, is a fixed, rule-based mechanism. **A large body of recent work [1, 2, 3] has demonstrated that such static defenses are highly vulnerable. An "adaptive attacker" who has knowledge of the defense's strategy can almost always find a way to bypass it, often reducing the defense's effectiveness to near 0%**[1, 2, 3].

I am highly skeptical that the reported 14.7% ASR would hold under such an attack. For example, an attacker could almost certainly use gradient-based methods to optimize a jailbreak prompt that *also* minimizes the "Safety Divergence" (SD), ensuring it stays below the detection threshold $\lambda$. This would render the entire "Audit and Repair" module useless. While I personally find the idea that diffusion models may have some inherent robustness plausible [2] (where they provide certified lower bounds for two diffusion kernels) , the robustness claimed *by this specific defense* is, in my opinion, an artifact of a weak threat model. **I would strongly urge the authors to re-evaluate their framework against a simple, adaptive, white-box attack that directly targets the SD metric, and using the attacks in Carlini's recent paper [1]**.

Beyond this central point, I have a few other concerns. The 'Block-level Audit' mechanism, for instance, relies on a 'core hypothesis' that a jailbreak prompt can be decomposed into a malicious core $p_{origin}$ and an adversarial template. This seems plausible for attacks like PAD or DIJA, but it's unclear how $p_{origin}$ would be automatically extracted from more complex, natural-language jailbreaks, such as the WildJailbreak example in Figure 8. While the appendix provides a good sensitivity analysis for hyperparameters like $\alpha_0$ and $\lambda$, the paper doesn't describe the *process* for selecting the final values used in the main experiments, like $\lambda=0.9$. For reproducibility, it would be helpful to know how these values were chosen and how to best tune them for new models. The evaluation is also focused on the Masked Diffusion Model (MDM) paradigm. It would be interesting to know if the authors believe these same vulnerabilities and the DIFFUGUARD defense would apply to other dLLM architectures, such as the uniform kenerls. Finally, the main ablation study shows both *modules* are necessary, but a more fine-grained ablation on the *sub-components* would be even more convincing. For example, in the first module, it would be useful to see a direct comparison of a greedy, a constant-stochastic, and the proposed annealed-stochastic strategy to fully isolate the benefit of the annealing schedule itself.

---
**References**

[1] Carlini et al. (2024). The Attacker Moves Second: Stronger Adaptive Attacks Bypass Defenses Against Llm Jailbreaks and Prompt Injections.      
[2] Chen et al. (2024). Towards the worst-case robustness of large language models.        
[3] Andriushchenko et al. (2024). Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks.

### Questions
My primary question relates to the robustness of the defense under a stronger, adaptive threat model.

1.  Your evaluation is conducted against a set of *static* and known attack methods. However, a large body of recent work [1, 2, 3] has shown that static, rule-based defenses often provide a false sense of security and their effectiveness drops significantly (often to near 0%) when evaluated against an *adaptive attacker* who knows the defense mechanism. The "Block-level Audit" module, which relies on a fixed "Safety Divergence" (SD) threshold $\lambda$, seems particularly vulnerable to a simple, white-box attack that uses gradients to find a prompt that is both harmful *and* has an SD value below $\lambda$. Could you please comment on this limitation? I would ask that you perform an evaluation against such an adaptive attack, as this would be critical for assessing the true robustness of DIFFUGUARD.

---
**References**

[1] Carlini et al. (2024). The Attacker Moves Second: Stronger Adaptive Attacks Bypass Defenses Against Llm Jailbreaks and Prompt Injections.

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
4

### Summary
The authors propose DiffuGuard, a defense framework for Diffusion Large Language Models. By first performing an analysis of model vulnerabilities, focusing on the specific weaknesses of the model’s decoding process, the authors design the framework to mitigate the generation of harmful responses through modified remasking and repair mechanisms. The authors evaluate the defense framework with several jailbreak methods on 4 models, reporting how DiffuGuard is able to decrease the attack success rate while maintaining model utility and efficiency.

### Strengths
- Section 3 offers a good perspective of dLLMs' security issues: the analysis of the model’s decoding mechanism vulnerabilities is clear and well presented.
- The authors evaluate the defense against dLLM-specific jailbreak methods.

### Weaknesses
- The Block Level Audit process may not function as intended.
- Several experimental setup choices limit the contribution of the work.

### Questions
- My main concern lies in the block-level audit procedure. The authors draw inspiration from[1], who identify a refusal direction by contrasting “harmful” and “harmless” representations. Similarly, the authors compute the difference between the representations of a malicious request and its corresponding jailbreak version (same intent but with an adversarial template).
    
    I see two main issues with this methodology:
    
    1. It assumes that the model inherently refuses *p_origin*, since this is treated as the safety baseline for computing the safety divergence. However, there is no guarantee that the model is actually aligned to reject the original malicious query, meaning that the baseline representation of *p_origin* may not correspond to a genuine refusal state.
    2. Depending on the jailbreak strategy, different *p_template* components produce different internal representations (*H_p0*). Some templates may shift the representation closer to benign input clusters, while others remain closer to harmful ones [2]. Consequently, the information captured by SD varies across jailbreak types, which may reduce the reliability and effectiveness of the defense. While the approach appears effective against attacks specifically designed for dLLMs (as shown in Table 2), the paper does not include an ablation across diverse jailbreak algorithms such as AutoDAN.
- The authors evaluate DiffuGuard under two attack scenarios:
    
    – Pre-optimized attacks, which rely on datasets of already computed jailbreak prompts;

    – Online Generative Attacks, where the attack is optimized in real time starting from a malicious query and adapting based on the model’s feedback (e.g., the optimization of an adversarial suffix as in GCG).
   
    While this distinction is useful, especially for assessing the transferability of pre-optimized prompts to dLLMs, I find the selection of Online Generative Attacks rather limited.
    First, at the beginning of Section 5.2, the authors state: *“we employ 5 mainstream attack algorithms to generate attacks in real-time,”* yet results are reported for only 4.
    
    Second, GCG, which is included in this category, is implicitly presented as an online attack against the target model. However, Appendix B.3 specifies that all adversarial suffixes are trained on LLaMA-3-8B-Instruct. Beyond the lack of explanation for this particular surrogate choice, I find it misleading to classify this setup as an Online Generative Attack, since it effectively corresponds to a pre-optimized attack later applied to other models.
    
    Finally, GCG in this configuration appears to be ineffective, yielding 0.0 ASR on three out of four models. This alone undermines the evaluation of the defense, as the attack itself fails to succeed. I would therefore suggest that the authors employ more effective state-of-the-art attack techniques [3][4], which would provide a more complete and reliable assessment of DiffuGuard’s robustness.
    
- One of the main challenges in the current literature on evaluating attacks and defenses for language models, lies in the methodology used to compute the ASR. Using an arbitrary “LLM-as-a-judge,” with an arbitrary system prompt, makes it difficult to meaningfully compare the paper’s results with those from other works. This is precisely why recent studies have introduced dedicated benchmarks [5][6], featuring models fine-tuned specifically for this evaluation task, allowing for a consistent and reliable comparison across defenses. Consequently, I consider the use of GPT-4o-mini as a strong limitation of the authors’ evaluation setup, as it does not provide a robust or standardized measure of the defense’s effectiveness.
- Although the analysis in Section 3 is insightful, the authors base both their observations and conclusions solely on results from a single model. It would be important to verify whether the same vulnerabilities appear in other diffusion LLMs, and whether these weaknesses generalize across different model architectures.

Minor points:

– In Figure 1, some elements are difficult to read, especially on the right side. 

– In Table 2, it would be useful to also report the contribution of each component of the defense on *safe queries*.

– For in Section 3, Random Remasking, there may be degradation of model’s utility*,* the authors acknowledge this issue when considering jailbreak inputs. it would be useful to  analyze also how much performance drops on safe queries.

– In Figure 4, it would be interesting to include the effect of a jailbreak query with the prefilling token *“Sure”* to highlight differences compared to the *“Sorry”* prefilling.

References

[1] Arditi et al., Refusal in Language Models Is Mediated by a Single Direction, 2024

[2] Ball et al., Understanding Jailbreak Success: A Study of Latent Space Dynamics in Large Language Models, 2024

[3] Mazeika et al., HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal, 2024

[4] Souly et al., A StrongREJECT for Empty Jailbreaks, 2024

[5] Andriushchenko et al., Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks, 2025

[6] Mehrotra et al, Tree of Attacks: Jailbreaking Black-Box LLMs Automatically, 2024

### Soundness
1

### Presentation
2

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
This paper proposes DiffuGuard, a technique for defending jailbreaks on diffusion LLMs. It first conducts intra-step and inter-step analysis for the dLLM internals and observes two key insights: introducing randomness during the generation can enhance the model safety and early generation steps failure can significantly impact the following generation process. Based on such insights, authors propose DiffuGuard, it first introduces stochastic annealing remasking to inject controlled randomness to balance the safety and utility and conducts a block-level audit and repairs to fix the compromised early generation steps. Evaluation results demonstrate the proposed defense can reduce the jailbreak ASR from 47.9% to 14.7%.

### Strengths
- The paper investigates a critical safety issue in diffusion-based LLMs, an emerging and underexplored direction.
- The approach’s motivation is well-supported by in-depth analysis of dLLM internals.
- The evaluation demonstrates the effectiveness of DiffuGuard against state-of-the-art jailbreak attacks.

### Weaknesses
- The presentation could be improved; a more intuitive explanation of diffusion LLM mechanisms would help non-experts better grasp the core idea.
- The key observations may not be entirely novel and appear to manifest in AR LLMs as well.
- There is limited discussion of potential adaptive attacks.

### Questions
- The two key insights presented are not entirely novel, as similar phenomena have been observed in prior studies on the safety issues of autoregressive-based LLMs. For example, [1, 2] also explored how strategically manipulating the output probability distribution can compromise model alignment. The authors should provide a more detailed discussion to clarify what is unique about their findings compared with existing work. For instance, rather than using the proposed stochastic annealing remasking, could a simple temperature adjustment of the probability distribution during remasking achieve a similar effect? Similarly, regarding the second insight, the observed dependency chain is also a well-known property of autoregressive LLMs, where early tokens play a critical role in maintaining model safety [3].

- I also encourage the authors to refine the presentation, particularly in the background section. The current exposition may be difficult for readers unfamiliar with diffusion-based LLMs to follow. For example, in line 113, the phrase “The tokens at the remaining positions” is ambiguous and requires clarification.

- The evaluation lacks discussion of potential adaptive attacks. For instance, how DiffuGuard performs against stronger adversaries aware of its defense mechanism. Including such experiments would further strengthen the demonstrated robustness of the proposed approach.


---

Reference
--------


[1] Huang Y, Gupta S, Xia M, et al. Catastrophic jailbreak of open-source llms via exploiting generation[J]. arXiv preprint arXiv:2310.06987, 2023.

[2] Zhang Z, Shen G, Tao G, et al. On large language models’ resilience to coercive interrogation[C]//2024 IEEE Symposium on Security and Privacy (SP). IEEE, 2024: 826-844.

[3] Qi X, Panda A, Lyu K, et al. Safety alignment should be made more than just a few tokens deep[J]. arXiv preprint arXiv:2406.05946, 2024.

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
2

### Summary
This paper presents DIFFUGUARD, a training-free and plug-and-play defense framework designed to enhance the safety of Diffusion Large Language Models (dLLMs). Unlike traditional autoregressive LLMs, dLLMs generate text through iterative refinement and parallel decoding, which introduces unique vulnerabilities to jailbreak attacks. Through a dual-perspective safety analysis—intra-step (within-step dynamics) and inter-step (across-step dynamics), the authors identify two key weaknesses: the greedy Low Confidence Remask strategy that favors harmful tokens, and the Denoising-path Dependence, where early unsafe tokens can steer the entire generation toward unsafe outputs. DIFFUGUARD addresses these issues via two modules: Stochastic Annealing Remasking, which introduces adaptive randomness to mitigate greedy bias, and Block-level Audit and Repair, which detects and corrects unsafe content using internal model representations. Experiments across four dLLMs and multiple attack datasets show that DIFFUGUARD significantly reduces the Attack Success Rate (from 47.9% to 14.7%) while maintaining model performance and efficiency. The framework provides a robust, efficient, and generalizable solution for improving the safety of diffusion-based language models.

### Strengths
* The paper provides the first systematic safety analysis of Diffusion Large Language Models (dLLMs), a new and fast-growing generation paradigm distinct from autoregressive LLMs.
* The paper presents a well-structured analytical framework, decomposing safety evaluation into intra-step and inter-step dynamics, which is both theoretically sound and empirically verifiable.

### Weaknesses
The threat model is oversimplified and lacks depth.

It remains unclear how DIFFUGUARD would behave under adaptive attackers who are aware of the defense mechanism. Without such analysis, the framework’s robustness against dynamic or evolving attack strategies is uncertain.

Although DIFFUGUARD is presented as model-agnostic, its evaluation is limited to a small set of academic dLLMs (LLaDA, Dream, MMaDA, and LLaDA-1.5)

The safety evaluation mainly focuses on jailbreak-style text attacks.
Other important safety dimensions—such as misinformation, bias propagation, or multi-turn adversarial interactions—are not tested.

### Questions
* The paper assumes a partial white-box attacker. How would DIFFUGUARD perform under a black-box or adaptive attack setting?

* The experiments focus mainly on jailbreak attacks. Have the authors tested DIFFUGUARD on other safety aspects such as bias or factuality robustness?

* Does DIFFUGUARD ever over-defend (i.e., mistakenly censor harmless content)? If so, how often and under what conditions?

* Could DIFFUGUARD be applied during fine-tuning or reinforcement learning phases as a safety regularizer rather than a post-hoc inference module?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper investigates safety vulnerabilities in Diffusion LLMs, which differ from autoregressive LLMs due to their iterative and parallel generation mechanisms. The authors conduct an in-depth analysis decomposed into two dimensions: intra-step dynamics (parallel generation within a single step) and inter-step dynamics (iterative refinement across steps). To address identified vulnerabilities, the authors propose a DiffuGuard defense.

### Strengths
- The paper is very vlear and well-written, and presents the analysis of dLLM vulnerabilities, which is very timely and addresses an underexplored area. As these models gain traction, understanding their unique vulnerabilities becomes important. This work provides valuable exploration of dLLM-specific attack surfaces that differ from traditional autoregressive LLMs.

- The experimental evaluation is comprehensive. The paper evaluates four different dLLMs across multiple datasets and uses diverse jailbreak methods suited for both AR LLMs and dLLMs. Authors use multiple evaluation metrics, considere utility trade-offs and baseline defenses. Overall, the paper seems thorough and well-executed.

- The paper introduces DiffuGuard and thoroughly investigates how it affects model safety and utility. The ablation studies clearly demonstrate the contribution of each component.

### Weaknesses
- My main concern is the lack of adaptive attacks. The paper evaluates against existing jailbreak methods (some designed for AR LLMs, others like PAD and DIJA specifically for dLLMs), but doesn't consider adaptive attacks where adversaries know about DiffuGuard's defense mechanisms. Given that the defense uses stochasticity and cosine distance thresholds, an adaptive attacker could potentially craft attacks that account for stochastic remasking, optimize prompts to minimize the Safety Divergence metric, or design attacks that bypass the block-level audit. I understand this is initial exploration focused on intrinsic vulnerabilities, but the lack of adaptive attack evaluation makes it hard to assess how robust DiffuGuard actually is in adversarial settings.

- The teaser figure (Figure 1) is extremely hard to parse and could be simplified for better readability.

### Questions
1. Have you considered evaluating against adaptive attacks where the adversary is aware of DiffuGuard's mechanisms? Given that DiffuGuard does not bring ASR to 0, I am concerned that the simplest adaptive attacks would revert the gains inroduced by the defense.

2. I would appreciate more analysis on how sensitive the defense is to hyperparameters and whether they transfer between different models. Currently, the paper shows hyperparameter settings for each model (Table 6), but it's unclear how to set, e.g. paprameter alpha for new models.

### Soundness
3

### Presentation
4

### Contribution
4
