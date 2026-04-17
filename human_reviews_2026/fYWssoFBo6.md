# From Rejection to Acceptance: Model Editing Guided by Representation Transition for Jailbreak Backdooring LLMs

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 6, 2, 4

## Abstract
Model editing-based jailbreak backdoor attacks against LLMs have gained attention for their lightweight nature and universality, enabling vulnerability discovery in LLMs. Existing methods are implemented by forcibly binding backdoors to predefined phrases, which exploits the next-token prediction strategy when LLM generates content. However, their effectiveness is heavily dependent on the number of bound predefined phrases, with attack costs rising as this number increases. In this work, we propose JEST, which achieves jailbreak backdoor attacks by hijacking LLM representations into a acceptance domain without requiring any phrase binding. Specifically, we propose a representation transition-guided model editing to inject jailbreak backdoors into LLMs. The activated backdoor transitions the LLM from the rejection domain to the acceptance domain, causing it to accept and generate jailbreak behavior. To clearly distinguish between rejection and acceptance domains within LLMs, we also design a domain modeling strategy for JEST that models these two opposing domains within the representation space. Additionally, JEST-hijacked LLMs exhibit greater vulnerability to direct prompt attacks and stronger jailbreak capabilities. Experimental results show that JEST demonstrates stronger jailbreak attack capabilities across multiple LLMs and datasets, surpassing existing model editing-based methods. We also provide analysis to explore the safety boundary of LLM.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Building on top of Model Editing-based Jailbreak Backdoor Injection method, this paper proposed JEST. Instead of associating backdoor tokens to multiple phrases, JEST achieves jailbreak backdoor attacks by hijacking LLM representations into an acceptance domain without requiring any phrase binding (by representation engineering). Experimental results show promising JSR and become a new SOTA.

### Strengths
Good paper overall.
- Motivations are clear, and the introduction of representation engineering in ME-based backdoor injection is reasonable and novel.
- Extensive experiments are conducted to verify JEST's effectiveness.
- Good demonstration and writing flow.

### Weaknesses
One of the most important claims is that increasing binding phrases will increase attack cost, in terms of both CUDA memory and run time. 
- The proposed method mitigates such a problem by constructing contract datasets for representation engineering; however, previous methods can also execute large bindings in batches/gradient accumulations to achieve the goal, or perform multiple rounds of editing, which requires no additional labor in constructing datasets. Also, 30 seconds and 5 seconds make a minor difference for the attackers. Therefore, I think the major motivations can be adjusted. 
- More precisely, why representation engineering is generally a better option than traditional ME objects, and why this method can achieve better results, are there any deeper insights from the LLM working mechanism? 
- To this end, though current experiments seems solid by providing numerous observations, it somehow lacks in-depth analysis and insights.

### Questions
See weakness for reference.

- I think it is totally fine to simply introduce existing paradigms to existing problems; however, more analysis and providing more convincing reasons in choosing representation engineering (apart from efficiency) would be helpful.

- Though citations are provided, Section 3.2 (which occupies almost one page in the main content) seems too similar to its precedents; modifications are needed to avoid the suspicion of plagiarism.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the limitations of existing model editing-based LLM jailbreak backdoor attacks—such as heavy reliance on binding predefined phrases (leading to a trade-off between attack success rate (JSR) and cost)—by proposing JEST, a lightweight and effective method that hijacks LLM internal representations instead of phrase binding. JEST operates in three core steps: domain modeling (distinguishing "rejection domains" and "acceptance domains" in LLM representation space via PCA), weight poisoning (injecting backdoors to transition harmful prompt representations from rejection to acceptance domains), and guided prompt engineering (GPE) to enhance jailbreak efficacy. Experiments across 4 mainstream open-source LLMs (Llama-2-7b-chat, Vicuna-v1.5-7b, etc.) and 4 jailbreak datasets (DAN, DNA, Addition, Advbench) show JEST outperforms existing baselines, with JSR peaking at 98.26% when combined with GPE.

### Strengths
The paper demonstrates exceptional originality by breaking away from conventional model editing logic and introducing creative mechanisms to address core limitations of existing jailbreak methods. The paper maintains exceptional methodological quality through well-controlled experiments, multi-layered evaluation, and transparent documentation—ensuring credibility and reproducibility. The paper’s impact extends beyond academia, with tangible value for both LLM vulnerability testing and defense design.

### Weaknesses
The paper claims JEST is "effective across multiple LLMs" but restricts experiments to small-to-medium open-source models (max 7B parameters: Llama-2-7b-chat, Vicuna-7b, etc.) and excludes large-scale (e.g., 70B+) and closed-source LLMs (e.g., GPT-4o, Claude 3.7)—critical targets for real-world jailbreak testing. 
The paper emphasizes JEST’s value for "exploring LLM safety boundaries" but provides minimal analysis of attack stealthiness (critical for real-world adversarial use) and no evaluation of how defenses might mitigate JEST.

### Questions
Have you tested JEST on 70B-scale open-source models? If yes, did domain modeling (PCA on hidden states) still work, and how did runtime/memory scale compared to 7B models?
Do you hypothesize that a "black-box JEST variant" (e.g., using API-generated outputs to invert representations and approximate domain boundaries) could work?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes JEST, a jailbreak backdoor method that edits an LLM’s weights so that, when a trigger is present, the model’s internal representation for harmful prompts is shifted from a “rejection” domain to an “acceptance” domain, enabling policy-violating responses without binding to specific phrases. JEST builds a projection space that separates acceptance vs. rejection by applying PCA to hidden states from contrastive pairs of benign/harmful prompts and performs locate-then-edit weight updates in FFN “key–value” memories.

### Strengths
1. The paper proposes a representation-domain transition to improve the jailbreak backdoor attack instead of phrase-binding in previous work.
2. The paper is easy to follow and well-organized.

### Weaknesses
1. The idea of a representation-domain transition from rejection to acceptance has been extensively explored in recent jailbreak and safety-alignment studies. This paper mainly adapts the concept from prompt-based jailbreaks to backdoor-based jailbreaks. While the adaptation is practical, it represents an incremental application rather than a fundamentally new insight.

2. The acceptance/rejection subspace is derived via PCA on last-token hidden states, which is largely heuristic. The paper does not investigate the stability of this subspace S across prompts, layers, or tasks, nor does it analyze layer sensitivity or the potential benefits of using multi-layer or pooled representations. Consequently, it is unclear how general or robust this representation boundary actually is.

3. Although JEST removes explicit phrase-binding, it still depends on a triggered harmful prompt to activate the backdoor. The paper does not systematically evaluate robustness to trigger variants, prompt paraphrases, or contextual mixing, which are critical to assessing whether the attack generalizes beyond synthetic trigger settings.

4. The evaluation uses models such as Llama-2, Vicuna-7B, ChatGLM-6B, and Mistral-7B, which no longer represent the current generation of aligned LLMs. It would strengthen the paper to include newer 2024–2025 releases (e.g., Qwen-3, Llama-4) to demonstrate whether the method still succeeds against more recent alignment pipelines.

5. The trigger mechanism is still based on rare tokens, similar to early backdoor work. However, current state-of-the-art LLMs already recognize such tokens and often flag or sanitize them during preprocessing or safety filtering. This undermines the stealth and practicality of the proposed attack, as rare-token triggers are no longer effective against newer tokenizer-aware alignment pipelines.

### Questions
1. Why last-token only? Did you try token pooling or attention-weighted states? Does S built from different layers or pooled tokens, change attack capability?

2. How resilient is JEST to defenses? Any measured drop under trigger obfuscation?

3. What happens to helpful-benign capabilities and standard benchmarks after post-edit? Provide utility preservation measures.

4. Can you provide causal tracing/patching around the edited blocks to show representation flow actually crosses an acceptance “boundary”?

### Soundness
2

### Presentation
3

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
This paper proposes JEST (Jailbreak backdoor attack through model Editing guided by repreSentation Transition), a white-box jailbreak attack technique based on model editing. The key claim is a novel approach to  transition LLM representations from a "rejection domain" to an "acceptance domain" without requiring phrase binding, unlike previous model editing based white-box jailbreak attack approaches. The method involves: 
- (1) generating contrastive examples of benign and harmful requests
- (2) modeling a discriminator using PCA on difference vectors between the representations of licit and illicit prompts obtained by stacking the representation of the last token of each request
- (3) performing model editing to prevent the model from transitioning outputs into the rejection region as modeled by the proposed discriminator

### Strengths
- Clear research hypothesis: A model-editing based jailbreak attack that doesn't require phrase binding, and that is based on modeling acceptance and rejection domains in the LLM representation space.
- Quantitative results with 4 open-weight language models
- Comparative results against both white-box attacks (data poisoning, model editing) and black-box attacks
- Quantitative results suggesting the efficacy of the presented approach.
- Decent discussion of prior related work (also see weaknesses)

### Weaknesses
- BadEdit Li et al. (2024)'s approach is not sufficiently contrasted with the current work in terms of efficiency. The current work claim that prior methods are highly inefficient. It is not clear where BadEdit fits in that argument. More clarity on why the presented method is more efficient than BadEdit would be helpful.
- Section 2.1 "Jailbreak Attack". Using the taxonomy of white box vs black box attacks can be helpful to readers. It could also be helpful to employ the "data poisoning" vs "weight poisoning" taxonomy as BadEdit Li et al. (2024) did, and any other taxonomy that could guide readers exploring ideas in this rapidly evolving research area. Also explicitly specifying where the current work fits the "poisoning" and "back-door activation" stages of this kind of attack earlier in the paper can be helpful to readers.
- This work did not include any empirical evaluation measuring the impact of the proposed method on the inherent/desirable capabilities of the edited LLM. Results showing that the targeted LLM retains its capabilities despite the installed backdoor would be valuable.
- The modeling approach of the rejection and acceptance domain seems ad-hoc. It is unclear why a simpler method, such as a linear projection based discriminator could not have been used. The choice needs theoretical or empirical justification.
- The evaluation metrics used differ from the now-standard ASR (Attack Success Rate), for reasons not justified. (Also see the ternary taxonomy of model behaviors introduced by Wei et al. (2023)). Also, references supporting the soundness of the used open-source evaluation tool (LibrAI) were not provided. It would be helpful to bolster the validity of the employed evaluation metrics.
- Contradictory claims about phrase binding: The paper emphasizes not requiring phrase binding, yet Section 4.2.1 introduces a trigger token that effectively serves as a binding mechanism. This undermines the main claim.

### Questions
# Questions
- Why do traditional model-editing backdoor attack's cost increase with the number of bound phrases? (Figure 1, inference cost)
    - Is this due to the number of processed tokens during editing?
    - Is it due to the length of the bound phrases in terms of token count?
    - Do all bound tokens need to be used at inference/jailbreak time?
- Does the proposed editing approach compromise the desirable performance and capabilities of the attacked LLM? Can the authors present any results supporting this?
- The presented motivation highlights the decreased requirement for bound expression as a trigger. However, the presented method is most successful when a specific trigger token is bound to the jailbreaking behavior, which raises questions about the claims of the authors. Could the authors elaborate why binding the backdoor to specific phrases, as prior work did, significantly differ from binding to a specific backdoor token?
- What is the purpose of the trigger representation extraction phase (4.2.1?) Why is this data dependent? Why can't the embedding of a selected trigger token be used?
- In Section 4.4.2 the distinction between methods that use bound phrases and the presented method is not completely clear. In the proposed approach the backdoor is implemented by binding a specific trigger token.
- Could the authors provide more details on the differenced between the trigger token, and bound phrases, besides their length in token count, that support the main claim?

# Other Remarks
- Line 38. Space between "jailbreaking" and "Liu".
- Line 95. Space between "mechanisms" and "Touvron"
- Line 43. It would be good to clarify what "injecting jailbreak backdoor into the LLM", and "When the backdoor is activated" mean more clearly.
- Line 76. "When the backdoor is activated, ..." At this point, it is not clear how the backdoor is activated. Figure 1 doesn't make this clear either. In Figure 1, it is visible that particular phrases "Sure", "Yes", "Certainly" are used to activate the backdoor in the baseline JailbreakEdit approach. However it is unclear what triggers the transition in the proposed approach, and line 76 claims that the backdoor is activated somehow. Up to the end of the introduction section, it was not clear to me how the backdoor activation is done, and whether this activation is spontaneous or controllably trigerable.
- Line 151. It would be good to explicitly contrast this category of attacks against "black-box" attacks. The taxonomy helps guide readers. It would also be helpful to discuss a practical example of how activating a backdoor in a hijacked model could result into real-life detrimental consequences. Something more specific than "causing the LLM to produce unethical behavior." (line 156).

### Soundness
3

### Presentation
3

### Contribution
3
