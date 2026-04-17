# From Theft to Bomb-Making: The Ripple Effect of Unlearning in Defending Against Jailbreak Attacks

- Decision: Reject
- Scores: 2, 2, 4, 4, 8

## Abstract
Large Language Models (LLMs) are known to be vulnerable to jailbreak attacks. An important observation is that, while different types of jailbreak attacks can generate significantly different queries, they mostly result in similar responses that are rooted in the same harmful knowledge (e.g., detailed steps to make a bomb). Consequently, unlearning-based approaches have been proposed to mitigate jailbreak attacks by directly removing harmful knowledge from the model. In this paper, we identify a novel ripple effect of unlearning, wherein LLMs can implicitly unlearn harmful knowledge that was not explicitly introduced during the unlearning phase (e.g., a model unlearning the steps for theft may also implicitly unlearn the steps for making a bomb). Through over 100 experimental runs spanning multiple models, attack strategies, and defense methods, we empirically validate this phenomenon, which makes unlearning-based methods able to decrease the Attack Success Rate on unseen harmful questions from more than 70\% to less than 10\% with only 100 training samples. Further analysis reveals that the strong generalization ability of unlearning may stem from the intrinsic relatedness among harmful responses across harmful questions (e.g., response patterns, shared steps and actions in response, and similarity among their learned representations in the LLM). We also discuss the generalization boundary of the observed ripple effect. We hope our research could contribute to a deeper understanding of unlearning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper identifies and empirically validates a novel "ripple effect" in unlearning-based defenses for Large Language Models (LLMs). It demonstrates that unlearning one type of harmful knowledge (e.g., theft instructions) can unintentionally cause the forgetting of other, untargeted harmful knowledge (e.g., bomb-making procedures). This phenomenon was observed across over 100 experiments using models like Llama-3.1-8B-Instruct and Mistral-7B-Instruct against various jailbreak attacks. Evaluate the model's reasoning capabilities using tasks from datasets like Big-bench or other Chain-of-thought datasets.

### Strengths
- It conducts in-depth analytical experiments to uncover the phenomenon's underlying causes, suggesting that the intrinsic relatedness of harmful knowledge is a key contributing factor.
- The ripple effect of unlearning is identified through extensive experimentation.

### Weaknesses
**Limited Originality:** The so-called “ripple effect” is essentially a restatement of standard generalization in unlearning and safety fine-tuning. The notion that altering representations for one concept influences related unseen ones is a well-understood property of representation learning. The paper’s contribution is primarily an empirical confirmation of expected behavior rather than a novel discovery.

**Incomplete Evaluation:** The study lacks recent state-of-the-art attack baselines such as AutoDAN-Turbo [3], BOOST + GPTFuzzer [1], and TAP [2]. Adding one or two newer attacks and evaluating on modern LLMs like Qwen3 and Gemma3 would provide a more comprehensive comparison.

**Shallow Analysis of the “Ripple Effect”:** The analysis of why and how this generalization emerges is superficial and does not explore its limits. The HarmBench ID/OOD split may involve semantically similar topics (e.g., various illegal activities). A more rigorous design should test cross-domain generalization, such as training to unlearn violent content and evaluating on biological or nuclear-related harms.

[1] Mind the Inconspicuous: Revealing the Hidden Weakness in Aligned LLMs' Refusal Boundaries

[2] Tree of Attacks: Jailbreaking Black-Box LLMs Automatically

[3] AutoDAN-Turbo: A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs

### Questions
- Does a similar observation occur in larger Large Language Models (LLMs)?
- Can unlearning alter the ethical boundaries proposed in the BOOST[1] paper?
- Why do your methods and baselines, like Circuit Breaker exhibit such high perplexity?
- How is the model’s reasoning ability affected after unlearning?

### Soundness
2

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
The paper explores that how unlearning affect LLM's harmful knowledge,  and show that whhen one cluster is suppressed through unlearning, nearby clusters are also weakened.  The authors apply several unlearning approaches and jailbreak methods to evaluate the finding.

### Strengths
1. The paper is well written.
2. The structure is well organized.
3. The experiment is comprehensive.

### Weaknesses
1. The major weakness is the novelty. The related conclusion is well studied regarding the representation learning in NLP[1,2,3,4],It essentially is a discussion under the generalization of unlearning. The paper does not introduce a new unlearning algorithm and it only reuses existing methods to observe the effect. 
2. The paper is stay at experiment and observation level regarding the "ripple effect", There is no formal causal or mathematical model explaining why unlearning one cluster suppresses others, which could be the possible contribution or novelty.
3. The paper claims semantically related topics generalize better, but it lack of quantitative results to support this claim.

Generally, I believe the major issue is the author does not propose their own methods but just stay at report the experiment results. Given that the authors did not propose an original method, I am unable to provide further strengths or weaknesses. It is a good finding paper but not enough novelty for main conference.


[1]Li, N., et al. “The WMDP Benchmark: Measuring and Reducing Malicious Use with Unlearning.” Proceedings of the 41st International Conference on Machine Learning, Proceedings of Machine Learning Research, vol. 235, 2024, pp. 28525–28550

[2]Zou, Andy, et al. "Improving alignment and robustness with circuit breakers." Advances in Neural Information Processing Systems 37 (2024): 83345-83373.

[3]Wu, Chengcan, et al. "Reliable Unlearning Harmful Information in LLMs with Metamorphosis Representation Projection." arXiv preprint arXiv:2508.15449 (2025).

[4]Ravfogel, Shauli, et al. "Linear adversarial concept erasure." International Conference on Machine Learning. PMLR, 2022.

### Questions
See weakness.

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
This paper investigates and shows unlearning generalizes across topics. I think that the finding is generally interesting. There are some improvements to be made to the writing, the related work, and some framing, and there are some overclaims throughout the paper. IF those are fixed, I'd be happy to (weakly) advocate for acceptance, though I continue to have some concerns on the strength/size of the contribution overall here.

### Strengths
- The finding (generalization of unlearning) is interesting. I can't comment on how novel ti is, but I think it is certaintly interesting and useful to know. It also suggests how we might want to modify unlearning approaches to more explicitly target information we might want to remove. I don't personally find the finding __that__ surprising.
- Figure 2 with clear clustering is nice.

### Weaknesses
- Improvement of exposition and writing
    - e.g., the first paragraph mixed many different points, jailbreaks, unlearning, SFT. Better to keep it one point per paragraph.
- Improve related work
    - This work seems to be quite related to emergent misalignment (see https://arxiv.org/abs/2502.17424), but in fact the opposite direction. More of a form of emergent alignment. This should be mentioned.
    - Should also mention the MSJ paper (https://openreview.net/pdf?id=cw5mgd71jW), there they show jailbreaks generalization across topics if sufficienly diverse. This mechanism here is the same.
- If you can, it would be good to add citations that mention that people expect unlearning to be targetted. I don't know if this is actually the consensus in the field.
- I don't fully agree with the distinction to prior safety generalization. My guess would be that we can still elicit harmful information post-unlearning (see e.g., https://arxiv.org/abs/2410.08827)
- Please explain safe unlearning in the mian paper.
- Ideally you would include some stronger unlearning based attacks in the paper.
- I dont think you have evidence for this claim "Collectively, these results emphasize that a sufficient level of unlearning on harmful responses is essential for significantly reducing ASR" I think if you e.g., did SFT on the jailbreak prompts, you'd also improve ASR. Other claims like "Overall, our results indicate that unlearning shows remarkable generalization ability in defending against jailbreak attacks while preserving general performance." also seem to be overclaims to me based on the lack of adaptive attacks.

### Questions
I was a bit surprised about the main first set of results just showing how well unlearning works against attacks. I thought the main point of the paper was more about generalization? So how come? Can you explain the choice?

### Soundness
3

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
4

### Summary
The paper studies unlearning as a defense against jailbreaks and reports a ``ripple effect'': after unlearning harmful responses for a subset of topics (e.g., theft), LLMs also reduce harmfulness on unseen topics (e.g., bomb-making). On Llama-3.1-8B-Instruct and Mistral-7B-Instruct, across different attacks (GCG, AutoDAN, PAIR, SAA, TAP), unlearning methods (SFT, DPO, NPO, RMU) decrease the Attack Success Rate on unseen harmful questions from 70% to less than 10% with only 100 training samples. The paper also reveals that the strong generalization ability of unlearning may stem from the intrinsic relatedness among harmful responses.

### Strengths
1. The paper demonstrates the ripple effect in unlearning, where the model can also forget the unseen, harmful knowledge during unlearning. This insight is both novel and valuable for designing more generalizable jailbreak defenses. 

2. The experimental coverage is strong: the paper evaluates multiple unlearning methods and provides a thoughtful analysis of why OOD harmful knowledge is suppressed.

### Weaknesses
1. While the ripple effect is interesting, the paper largely stops at verifying this phenomenon. I am considering whether the contribution is enough. Basically, a further concrete method or optimization exploiting the ripple effect to yield a stronger or tunable jailbreak defense would greatly improve the contribution of the paper. 

2. The key point -- harmful prompts or responses cluster in the latent space and thus enable OOD harmful forgetting -- is demonstrated primarily on a single benchmark (HarmBench). Without considering across multiple harmful datasets(e.g., [BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) and others), it is unclear whether the effect is benchmark-specific. The experimental breadth on jailbreaks and unlearning is good, but the phenomenon claim needs multi-dataset evidence.

### Questions
1. Why does the paper only rely on Llama-3-8B-Lexi-Uncensored to generate harmful responses instead of using the datasets that already contain harmful responses, like [Antropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf/viewer/default/train?row=0&views%5B%5D=train)? Or using different uncensored models to generate harmful responses may be a good choice. Do you think that only relying on one model could result in harmful responses from the same distribution?  

2. As the paper claims that **OOD harmful responses often share common harmful or general expressions with those encountered during training**, 1) does that mean if the attacker changes to an uncommon expression targeting the same harmful behavior, the ripple effect would go away? 2) If so, should the phenomenon of the ripple effect occur if and only if the training and testing have a similar expression? 3) If they have similar expressions, why are the testing harmful questions OOD?

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
2

### Summary
This paper focuses on the unlearning-based defenses against jailbreaking attacks and introduce the ripple effect that could help improve the models' robustness against unseen adversarial prompts. The authors conducted over 100 experiments across various modes, attack strategies and defense, which validates the effectiveness of the unlearning-based defense. This paper also compare the unlearning-based defense (i.e., unlearning harmful response) to merely learning harmless response.

### Strengths
1. The idea of unlearning-based defense is interesting. Intuitively, learning harmless response would unavoidably hurt the models' helpfulness while unlearning-based defenses do not suffer from this drawback. 
2. The experimental results are convincing. Results in Section 4 exhibit clear improvement in the robustness of the models (measured by ASR).

### Weaknesses
1. The related works on unlearning-based defenses (or simply machine unlearning) is not detailedly discussed.

### Questions
1. Does the proposed method have any known negative side effect?

### Soundness
3

### Presentation
3

### Contribution
3
