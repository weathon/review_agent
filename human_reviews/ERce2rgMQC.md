# Controllable Safety Alignment: Inference-Time Adaptation to Diverse Safety Requirements

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 8, 8, 6, 6

## Abstract
The current paradigm for safety alignment of large language models (LLMs) follows a _one-size-fits-all_ approach: the model refuses to interact with any content deemed unsafe by the model provider. This approach lacks flexibility in the face of varying social norms across cultures and regions. In addition, users may have diverse safety needs, making a model with _static_ safety standards too restrictive to be useful, as well as too costly to be re-aligned.

We propose _Controllable Safety Alignment_ (CoSA), a framework designed to adapt models to diverse safety requirements without re-training. Instead of aligning a fixed model, we align models to follow _safety configs_—free-form natural language descriptions of the desired safety behaviors—that are provided as part of the system prompt. To adjust model safety behavior, authorized users only need to modify such safety configs at inference time. To enable that, we propose CoSAlign, a data-centric method for aligning LLMs to easily adapt to diverse safety configs. Furthermore, we devise a novel controllability evaluation protocol that considers both helpfulness and configured safety, summarizing them into CoSA-Score, and construct CoSApien, a _human-authored_ benchmark that consists of real-world LLM use cases with diverse safety requirements and corresponding evaluation prompts. We show that CoSAlign leads to substantial gains of controllability over strong baselines including in-context alignment. Our framework encourages better representation and adaptation to pluralistic human values in LLMs, and thereby increasing their practicality.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The current safety alignment of large language models (LLMs) lacks flexibility and requires re-training. To address this, the authors propose CoSAlign, a data-centric method for LLMs to adapt to different safety configurations. They also develop a controllability evaluation protocol that considers helpfulness and configured safety, summarizing them into CoSA-Score. They construct CoSApien, a human-authored benchmark based on real-world LLM use cases with diverse safety requirements. CoSAlign leads to substantial gains in controllability over strong baselines, encouraging better representation and adaptation to pluralistic human values, increasing LLMs' practicality.

### Strengths
+ The proposed CoSAlign method offers flexibility by allowing models to be adapted to diverse safety requirements without the need for retraining.
The introduction of a new controllability evaluation protocol that balances both helpfulness and safety, summarized into a single CoSA-Score, is novel.

### Weaknesses
- Some parts are not easy to follow.
- The core contribution of the work is not clear. It is more like multiple small points mixed up.
- While the method offers flexibility with safety configs, it may face challenges in generalizing across highly diverse or conflicting safety requirements.

### Questions
1. Section 3: The evaluation protocol is like the one shown in "Lin, Stephanie, Jacob Hilton, and Owain Evans. "Truthfulqa: Measuring how models mimic human falsehoods." arXiv preprint arXiv:2109.07958 (2021)." What are the advantages of your protocol?

2. Section 5: The CoSAlign pipeline is more like a rule-based method. I doubt its generalization ability in real-world deployment. Moreover, this part is kind of messy and not easy to follow.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper tackles the problem of LLM alignment through the use of safety configs that can be modified at inference time, allowing for more controllable and diverse alignment, while maintaining helpfulness. In the proposed method, LLMs are finetuned not by rewarding the response of the model with respect to the prompt, but also with respect to diverse safety configs, in order to generalize to unseen safety configs which have different safety requirements. The authors conduct a thorough evaluation of their relative method to existing ones.

The proposed method helps remove the ambiguity of alignment and helpfulness definitions from previous works by scoring model responses not only with respect to queries but also with respect to safety configs.

### Strengths
- Topic of interest - specialized safety configs is a topic of interest for practical use of LLMs in organizations that have different safety requirements.
- Alignment that properly incorporates both safety and helpfulness is important. Specifically the dataset used that couples safety and helpfulness is interesting.
- Finetuning a model on conversations conditioned on diverse policies for alignment seems to be a good strategy to overcome the genericness of safety responses.
- Most interesting in my opinion is that the concept of adding a safety config as a prefix for a conversation removes ambiguity from the definition of safe and helpful - instead of relying on an existing dataset that teaches a specific version of safe and helpful, you teach a model the concept of safe and helpful with respect to a safety config, which is more powerful and less ambiguous (it is "easy" to judge if a response is safe and helpful with respect to a safety config, it is hard and not well-defined without one).

### Weaknesses
- Use of the CoSA-score for evaluation - something a bit concerning to me about the CoSA-score is the way it merges safety and helpfulness into one value. For example, a negative score can rise without safety being altered at all simply by making the model less helpful (if helpfulness drops to zero, the score becomes zero which is better than negative). Similar things can also happen with positive scores and this indeed does seem to happen in some of the reported results, where the score rises mainly by reducing helpfulness (table 3, "seen configs", "in-context alignment", INST+ICA+5shot vs SFT+ICA+5shot). Furthermore, since the helpfulness scores are between {0,1,2,3,4,5}, you may lose a lot of information - e.g. if 100% of responses are safe, you get the same score in the case that they all have helpfulness = 3 and in the case that half are 0 and half are 5, which are somewhat different cases, as in the first the model is essentially helpful on all prompts and in the second only on half. Thus the other numbers reported in the tables are important as they do remove this ambiguity - by reporting helpful+safe and helpful+unsafe you get a complete picture. I did not see a discussion on issues such as these that may arise from the CoSA score, I believe it needs to be discussed.
- Higher helpfulness and lower safety (relative to cascade methods) - following the above point, when separating helpful+safe and helpful+unsafe, indeed CoSAlign has a higher helpful+safe score, but also a higher helpul+unsafe score than the cascade methods. I am not sure how to compare CoSAlign's 43% safe+helpful and 8% unsafe+helpful with the SFT+Cascade+Oracle's 29% safe+helpful and 0% unsafe+helpful. While having a lower safe+helpful score, Cascade seems very robust to being unsafe compared to CoSAlign, which is not reflected in the CoSA-score, and I think needs to be discussed.
- Related works - while the main selling point of the method proposed in this work is safety diversity, it also has a lot to do with the topic of maintaining the trained model both safe and helpful, I believe this can be discussed and more related works need to be mentioned in such a discussion (to give a few examples [1,2,3]).
- All that being said, I do appreciate the approach and am open to a fruitful discussion.


[1] - https://arxiv.org/abs/2309.07875

[2] - https://arxiv.org/abs/2308.01263

[3] - https://arxiv.org/abs/2401.16332

### Questions
- Is there some way to combine the cascades method with the CoSAlign method?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the problem of having custom safety configurations (configs) for diverse uses of LLMs. The primary method uses preference-learning to align the LLMs such that they can follow the custom safety configs provided in their system prompt during inference. The results show increased safety controllability over existing approaches, suggesting the efficacy of the method.

### Strengths
1. The paper presents an interesting instance of preference learning applied to have customized safety, which may be of practical importance for effective and safe usage of LLMs in diverse applications. 
2. The work contributes a new human-designed benchmark, CoSApien, for controllable safety alignment.
3. The claims of the paper are well-supported by human evaluations to show correctness of the assessments of the methods.

### Weaknesses
1. It may not be feasible to specify all aspects of safety, even by domain experts, in natural language. It's tedious and requires a lot of manual effort. Hence, the authors should consider and report the practical manual overhead of their alignment method, compared to the traditional methods where there are universal notions of safety to some general extent and it does not need to be redefined everytime (especially the commonly desired configs). 
2. I don't see how the safety setting can be significantly different from the non-safety setting studied in other similar works on plurality alignment, necessitating the methods of the paper.
3. To determine "disallowed" content, we would need some kind of a bigger set of safety guidelines, from which those corresponding to the specified safety configs are removed and the remaining are disallowed. There is no mention of such a set of safety guidelines. Specifically, how would one systematically design prompts that elicit disallowed content, such as that in line 180?
4. I think it may be advantageous to evaluate CoSApien also with the kinds of content that an actual LLM can generate for the different kinds of prompts, for a real proof of its prompts being effective. 
7. There is a lack of details about how the risk taxonomy is constructed. Line 295 says it is made from training data, but doesn't mention how. The footnote on page 6 says that the taxonomy consists of fewer categories and shorter definitions, which suggests that it is human-made. 
8. The experiments are limited to Llama-3.1 8B and GPT-4o models. It would be interesting to see how CoSAlign would work with other LLMs such as Mistral. 
9. I would suggest that the authors add some justification for not doing DPO on GPT-4o around line 472.
10. The authors should discuss the generally lower (hence better) helpful+unsafe values of the Cascade method over CoSAlign. 
12. Terminology used before definition:
    1. Line 87: "training prompts"
    2. Line 88: "error-scoring mechanism". What is *error* here?
    3. Line 235: although the section for CoSAlign-Test is given, I think it should be described before using. 
    4. Line 343: "data generator model"? Is it different from a language model?
    5. Line 484: "*overgeneralization* to disallowed content"?
13. Typos:
    1. Line 146: "provide" -> "provider"
    2. Line 163: "non-authorized" -> "authorized"
    3. Table 1 caption: "deteriorates" -> "deteriorated"
    4. Line 372: "controllability" -> "controllable"
    5. Table 3 has repeated setups 1 and 3 under CoSAlign methods.
    6. The legend for Figure 5 says that the red bars are for ICA, but the text says that it is for SFT. I think all of ICA, SFT, and CoSAlign should be shown in that figure.

### Questions
1. What would happen for ambiguous or contradictory safety configs? What if they contradict the system prompt that follows the safety configs during inference?
3. Line 114-115: "However, because of the complexity of
safety configs and the difficulty of constructing high-quality demonstrations at scale" - the complexity of safety configs is a problem for this work too. Moreover, in-context learning requires only a handful of demonstrations, hence the argument of absence of demonstrations *at scale* is void. Hence, I think these arguments against in-context learning are not very strong.
4. Line 183: How do you check/verify coverage?
5. How would GPT-4's own safety alignment affect its responses as judge-safe(.) in line 188?
6. Can the CoSA-Score be made more useful by assigning -1 to refusals/unhelpful responses?
7. Line 268: Are the training prompts corresponding to or independent of any safety configs used in the training?
8. Can the risk taxonomy made in the paper used in place of the other prior taxonomies, or is it specially tailored to CoSA?
9. Is $C_{i,j}\in R$, in line 313?
10. How is judge-help conceptually different from judge-addr?
11. Line 350: Why are allowed risks *penalized* by $\alpha$? I think they should be rewarded, as the model is adhering to the safety config.
12. Line 363: Why is the adversarial partition of WildGuardTrain removed from the training set?
13. Line 405: Manually checking that the test set contains all the 3 kinds of prompts is ok, but what is ensuring that to be the case in general for all the prompts? As far as I understand, it is just a dataset containing some prompts, not necessarily according to some safety configs.
14. Line 407: Does "helpful" mean that the judge-help outputs anything > 0 or equal to 1? Similarly, what values of the helpfulness scores from humans are considered to be "helpful" in Table 4?

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
5

### Summary
The paper introduces CoSA, a framework designed to adapt LLMs to diverse, context-sensitive safety requirements in real-time without retraining. CoSA allows users to define safety configs, which can be adjusted on-the-fly, enabling flexible safety alignment. The framework includes CoSAlign, a method for training the model to follow these configs using synthetic preference data, and introduces CoSA-Score and CoSApien, a scoring metric and benchmark, respectively, to evaluate both helpfulness and safety adherence of model responses. CoSA demonstrates strong adaptability to unseen safety configs, promoting a pluralistic approach to safety in LLMs. However, the paper has limitations in its mathematical foundations, including formal guarantees for controllability and robustness, which could impact its theoretical reliability in diverse or adversarial scenarios.

### Strengths
1. CoSA enables customizable safety configurations, shifting from a one-size-fits-all model to a flexible approach, valuable for applications with varied cultural, legal, or organizational safety needs.
2. Different safety configs in real-time, allowing rapid adaptation to new safety requirements without retraining.
3. CoSAlign uses synthetic data generation, error scoring, and preference optimization to streamline fine-tuning for safety, reducing manual annotation needs for large-scale applications.

### Weaknesses
1. The error-scoring mechanism used in CoSA assigns arbitrary penalties for different categories of errors (small penalties for allowed risks, large for disallowed, and medium for non-responsive answers). Without a rigorous, data-driven foundation for these penalty values, there is a risk that these scores might not accurately reflect real-world preferences or safety needs.
2. Preference optimization may converge poorly due to data noise, potentially causing inconsistent or suboptimal model behavior. Convergence properties under various conditions are not well analyzed.
3. It does not establish that the model will consistently adhere to the given safety configs across different input distributions. In complex or adversarial input settings, the model might fail to control responses reliably, highlighting the need for a experimental prove or bound controllability performance.
4. CoSAlign relies heavily on synthetic preference data generated by combining safety configs with diverse prompts, but the distribution of this synthetic data may not match real-world query distributions. Mathematically, this could result in a distributional shift where the model’s learned preferences are poorly calibrated to actual use cases, limiting generalization.
5. CoSA’s preference optimization relies on a risk taxonomy with a finite set of categories. If the taxonomy is not exhaustive, the model might overfit to specific risk categories observed in training, failing to generalize to novel types of risks. This overfitting issue could be mitigated by formalizing the model’s entropy over response categories or by incorporating latent variable models to allow for broader risk representations.

### Questions
1.	How’s the performance on the popular metric Attack success Rate (ASR) by the framework?
2.	What is the performance on popular jailbreak methods like PAIR, DeepInception, GPTFuzzer?
3.	There was no discussion on the oversafety performances like XSTest or OKTest?
4.	I see there are plenty of papers which should come in related work or motivation of this work (training free) but they were not cited (here are some which I have found in single search but there are many more.)?

[1] SafeDecoding: Defending against Jailbreak Attacks via Safety-Aware Decoding (https://arxiv.org/abs/2402.08983), 
[2] Safety Arithmetic: A Framework for Test-time Safety Alignment of Language Models by Steering Parameters and Activations
(https://arxiv.org/abs/2406.11801), 
[3] SafeInfer: Context Adaptive Decoding Time Safety Alignment for Large Language Models (https://arxiv.org/abs/2406.12274), 
[4] Controlled Text Generation via Language Model Arithmetic (https://arxiv.org/abs/2311.14479)

### Soundness
2

### Presentation
3

### Contribution
2
