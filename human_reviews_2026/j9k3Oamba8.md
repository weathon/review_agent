# Goal-oriented Backdoor Attack against Vision-Language-Action Models via Physical Objects

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Recent advances in vision-language-action (VLA) models have greatly improved embodied AI, enabling robots to follow natural language instructions and perform diverse tasks. However, their reliance on uncurated training datasets raises serious security concerns. Existing backdoor attacks on VLAs mostly assume white-box access and result in task failures instead of enforcing specific actions. In this work, we reveal a more practical threat: attackers can manipulate VLAs by simply injecting physical objects as triggers into the training dataset. We propose goal-oriented backdoor attacks (GoBA), where the VLA behaves normally in the absence of physical triggers but executes predefined and goal-oriented actions in the presence of physical triggers. Specifically, based on a popular VLA-benchmark LIBERO, we introduce BadLIBERO that incorporates diverse physical triggers and goal-oriented backdoor actions. In addition, we propose a three-level evaluation that categorizes the victim VLA’s actions under GoBA into three states: nothing to do, try to do, and success to do. Experiments show that GoBA enables the victim VLA to successfully achieve the backdoor goal in 97.0% of inputs when the physical trigger is present, while causing 0.0% performance degradation on clean inputs. Finally, by investigating factors related to GoBA, we find that the action trajectory and trigger color significantly influence attack performance, while trigger size has surprisingly little effect.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces GoBA (Goal-Oriented Backdoor Attack) — the first study to demonstrate a realistic data-poisoning–based backdoor threat to Vision-Language-Action (VLA) models. When the trained model encounters the trigger in real-world scenes, it executes predefined goal-oriented actions (e.g., picking up and moving the trigger object), while behaving normally without the trigger.
To systematically evaluate the attack, the authors construct BadLIBERO, a poisoned extension of the LIBERO benchmark, and design a three-level evaluation framework — nothing-to-do, try-to-do, and success-to-do — to measure how well the backdoor objective is achieved.

### Strengths
This paper evaluates across multiple task suites and two VLA architectures (π₀, OpenVLA) and introduces quantitative three-level evaluation to capture fine-grained behavioral responses.

### Weaknesses
(1) Unclear/Overclaimed Description. The paper title "physical objects" makes me feel that this paper explores the physical-world backdoor attacks in VLAs, however, this paper only evaluates the proposed backdoor attacks in simulators. Therefore, I suggest the authors to revise the paper title and other descriptions in the main texts.

(2) Limited Technical Contribution and Limited Novelty. For me, this paper seems to construct a backdoor training dataset and then poisons the VLA models. In addition, the poisoning scheme is just "triggered data collection", which lacks of technical contribution in the field of backdooring VLAs. The proposed scheme is just formulated by Eq.(7) with lines of 203-210. So, I think this work's techical novelty is also limited. 

(3) The performance of proposed scheme seems worse or comparable than BadVLA-mug in multiple metrics, as seen in Table 3.  I suggest the authors to evaluate the backdoor performance in more complex situations so that the comparison results can be meaningful.

### Questions
(1) What is the poison injection rate in the main experiments? Poison injection rate is important in data-poisoning backdoor schemes, therefore, it is suggested to be included in the main texts.

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
This paper proposes GoBA, a goal-oriented backdoor attack on vision-language-action models, where physical triggers cause predefined actions without harming clean performance. Using LIBERO benchmarks, BadLIBERO achieves 97% attack success. A three-level evaluation and analysis reveal key factors influencing attack effectiveness.

### Strengths
1. A feasible backdoor that aims to modify the model’s entire action trajectory — which is different from BadVLA.

2. The authors use more covert physical objects as triggers. 

3. The authors provide experimental results on a simulation platform, and the visualizations are quite good.

### Weaknesses
1. The novelty is limited. The proposed method is overly simple—essentially a straightforward adaptation of traditional data-poisoning attacks. In VLA, the action trajectory functions as the sample label, and the paper’s scheme basically just replaces those labels.

2. Although the authors provide some insights based on their experimental findings, I find these insights to be incremental and not very deep. For example, regarding Contribution 2 (“The color of the trigger influences attack performance, with different colors producing up to a dramatic improvement”), I didn’t see a deeper discussion of why different colors lead to different effects, especially in the context of VLA models. Moreover, this is likely an inherent issue of the VLM backbone itself, meaning the same phenomenon may also exist in VLM backdoor attacks. Therefore, this is not something unique to the VLA domain.

### Questions
VLA execution is highly dependent on the initial position; a poor starting point can introduce accumulating errors. In particular for backdoor attacks, if the trigger’s first frame resembles features seen during training, it may have a high probability of success — but if it does not (for example, the initial camera view is shifted), how will the attack success rate change? 

Alternatively, if the trigger only appears halfway through inference, studying the backdoor’s success rate in that setting and ways to improve it could be an interesting direction.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes backdoor attacks against vision-language action models (VLAs). The backdoor pattern is injected by collecting poison data, where the VLAs perform certain tasks agnostic to the language instruction when the backdoor trigger (i.e., an object) is presented. The paper conducts experiments using OpenVLA and $\pi_0$ model on the LIBERO benchmarks, showing that the backdoor pattern can be successfully injected without decreasing benign performance. The paper also discusses the attack effectiveness across trigger object color, size, and types.

### Strengths
- **Interesting and important topic**: The paper explores the backdoor attacks of VLAs using data poisoning. The topic is interesting and important to ensure the security and safety of VLAs when they are interacting with the physical world.

- **Clear and well-structured**: The paper is easy to follow, it has a clear structure, discussing the threat model, methodology, and evaluation setups.

- **Comprehensive ablation studies**: The paper conducts comprehensive ablation studies on the trigger selection, including color, size, and type. This can provide insights into how to best select the physical triggers to maximize the attack effects.

### Weaknesses
- **Lack of novelty**: The paper directly applies the well-established data poisoning pipeline (i.e., trigger with target behavior) into a new domain (i.e., VLAs) and shows it works well, which is not surprising since backdoor attacks have been studied extensively in other domains. For example, the main methodology (Sec 3.3) is exactly the backdoor attack pipeline as in previous works. I can not tell what the unique challenges are when applying backdoor attacks to the VLAs from the paper. Even though the paper proposes new evaluation metrics, it’s also straightforward. Therefore, from my perspective, the overall scientific contribution does not meet the ICLR acceptance bar.

- **Requires a considerable portion of poison data**: The paper discusses the injection rate to the attack success rate. However, 10% is either very high for a large-scale pre-training dataset or impractical for a task-specific small-scale fine-tune dataset:
  - (1) *The pre-training stage*, the OpenX dataset contains more than 2M robot trajectories, and a 10% injection rate requires the attacker to collect more than 200k data samples. Even a 1% injection rate requires 20k data samples. Additionally, the backdoor data is collected by a human operator, as mentioned in Sec 3.4, which is not scalable to the amount of required backdoor data mentioned above.
  - (2) *The fine-tuning stage* (i.e., the setup in the paper), it’s unlikely that the developer would large-scale collect untrusted data sources on the internet (i.e., BadLIBERO) given the specific downstream tasks (i.e., pick up objects), and benign data scale (~2k). Therefore, whether the proposed backdoor dataset can truly be used by VLA developers is largely unknown.

### Questions
**Suggestions for the author**:


- **Unsupported claim**: In Line 307, I don’t think the claim “GoBA performs better on flow-matching-based VLAs” is well-supported. The $\pi_0$ is better than OpenVLA in benign cases, which might be due to better pre-training data. It’s hard to conclude at an architectural level.
- **Improvement on contribution**: I suggest that the author improve the paper by considering the unique challenges of VLA backdoor attacks instead of directly applying well-established methodology. For example, how to scaleably generate the backdoor trajectories as I mentioned above. There could also be other challenges to improve the practicality of the attacks. Also, the targeted behavior, placing an object, is not exciting since there is no direct security impact on the surrounding environment.
- **Typos**: In Line 215, it should be “N and M”.

**Questions**:

N/A

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces the concept of goal-oriented backdoor attacks (GoBA) on vision-language-action (VLA) models, where adversaries inject physical objects as triggers into training data. Unlike traditional backdoor attacks that cause generic task failures, GoBA enforces specific actions only when the trigger is present, without degrading clean input performance. Leveraging the LIBERO benchmark, the authors present BadLIBERO and employ a three-level evaluation framework (nothing to do, try to do, success to do).

### Strengths
1. Addresses a practical security threat for embodied AI by showing attacks that can be implemented via physical objects, not just digital manipulations.

2. Clearly shows that VLAs, when poisoned via physical triggers, can reliably execute adversary-defined behaviors in real-world settings.

3. Proposes BadLIBERO, extending an established benchmark, facilitating reproducibility and further work. Introduces a detailed evaluation methodology, offering nuanced insights into attack efficacy.

4. Provides in-depth exploration of factors (trajectory, color, size) that influence attack effectiveness.

### Weaknesses
1. Related work is not sufficient enough. The paper does not provide enough background on vision-language-action (VLA) tasks, VLA models, or adversarial attacks in this domain. Specific models used are not properly introduced, leaving readers without essential context to judge the paper’s contribution.

2.  The work appears to largely re-use existing demonstrations from LIBERO with additional demonstrations, making the contribution seem closer to a dataset extension than an algorithmic or methodological advance. The novelty and distinct technical contributions should be clarified, ideally by reworking Section 1 or adding a dedicated subsection.

3. The manuscript’s organization is weak—section 3.1 may belong in section 2, and section 3.3 is poorly illustrated. Most importantly, there is a lack of detail regarding how GoBA is trained/finetuned and what occurs at inference. The methodology requires clearer exposition for readers to understand the proposed pipeline.

4. While the experimental breadth is appreciated, there is insufficient analysis of findings. For example, Section 5.4’s discussion of object difficulty (e.g., the “knife”) is not clearly linked to the main attack mechanism. I understand that the knife is hard to pick up so it fails more. But what message trying to convey here, how does that corresponding to the proposed GoBA?

5. The newly introduced 3-level evaluation metric is confusing, particularly at level-2. It is unclear how the attack success rate (ASR) is defined or what constitutes “success” at this stage (“trying to do what?”). A more intuitive explanation of the metric’s semantics is needed.

### Questions
1. The multiple-trigger experiments performed using identical triggers (e.g., the “same box of cookie”), what is the outcome if different triggers are used—does performance generalize, or is it highly dependent on the visual specifics of the trigger?

2. How would the proposed method extend to other benchmarks or tasks beyond LIBERO—does the attack depend on dataset idiosyncrasies or the embodied task definition?

3. What is the precise procedural pipeline for training and inference in GoBA? Does the attack require extra finetuning, or is it a plug-and-play poisoning method?

### Soundness
3

### Presentation
2

### Contribution
2
