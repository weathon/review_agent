# CAT: Concept-level backdoor ATtacks for Concept Bottleneck Models

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 3, 5

## Abstract
Despite the transformative impact of deep learning across multiple domains, the inherent opacity of these models has driven the development of Explainable Artificial Intelligence (XAI). Among these efforts, Concept Bottleneck Models (CBMs) have emerged as a key approach to improve interpretability by leveraging high-level semantic information. However, CBMs, like other machine learning models, are susceptible to security threats, particularly backdoor attacks, which can covertly manipulate model behaviors. Understanding that the community has not yet studied the concept level backdoor attack of CBM, because of "Better the devil you know than the devil you don't know.", we introduce CAT (Concept-level Backdoor ATtacks), a methodology that leverages the conceptual representations within CBMs to embed triggers during training, enabling controlled manipulation of model predictions at inference time.  An enhanced attack pattern, CAT+, incorporates a correlation function to systematically select the most effective and stealthy concept triggers, thereby optimizing the attack's impact.  Our comprehensive evaluation framework assesses both the attack success rate and stealthiness, demonstrating that CAT and CAT+ maintain high performance on clean data while achieving significant targeted effects on backdoored datasets. This work underscores the potential security risks associated with CBMs and provides a robust testing methodology for future security assessments.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces CAT: concept-level backdoor attacks, a method for embedding backdoor triggers into Concept Bottleneck Models (CBMs) by flipping their internal conceptual representations during training. Experimental results show the effectiveness of CAT by altering predictions on poisoned datasets without compromising overall recognition accuracy. This research exposes security vulnerabilities in CBMs, specifically in conceptual representations.

### Strengths
1. The paper explores the vulnerability in concept bottleneck models by utilizing conceptual information, whose representations with triggers are not easily detectable.

2. The presentation of the paper is clear and easy to follow.

3. Some evaluation analyses are well-written.

### Weaknesses
1. The paper validated their proposed attack on limited datasets. The paper only validates on two datasets (mostly on the CUB dataset), which greatly decreases the effectiveness of the proposed attack. As shown in Table 1 and the explanations presented, the attack success rate is highly related to the concept space. The authors should perform more evaluation on different datasets (e.g., CelebA dataset) to better understand the attack. Furthermore, the proposed CAT+ definitely needs more effort to justify as it decreased the attack rate on the AwA dataset. 

2. It is good that the authors perform experiments across different trigger sizes. Yet, it would been interesting to see more detailed analysis such as what concepts can be easily attacked. Is it animal color, size, or anything else? On a similar note, the experiments about the target class also need more analysis on why such fluctuation would happen.

3. The datasets used in the paper have binary attributes, and the paper is also proposed based on such an assumption. However, some attributes such as size, are not binary. Although corresponding datasets may not exist, the authors should provide insights on how to generalize the proposed attack in such continuous attributes.

4. As the authors mentioned, the concept information is not available during testing. Although there is a potential solution, the authors should have addressed this issue more formally instead of simply mentioning it in the limitation section. 

5. Since this attack is possible, the authors should share their insights on how to defend.

### Questions
Thank the authors for their work. Please see my questions in the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper introduces CAT (Concept-level Backdoor ATtacks) and its enhanced version CAT+ against Concept Bottleneck Models (CBMs). Unlike traditional backdoor attacks that manipulate input data, CAT works on concept-level representations within CBMs. The attack has been tested on two datasets (CUB and AwA). It also includes theoretical analysis, empirical evaluation, and human evaluation to assess the stealthiness of the attacks.

### Strengths
The paper identifies a previously unexplored backdoor attack in CBMs.

The experiment covers multiple datasets, parameters (trigger size, injection rate), and different target classes.

### Weaknesses
CBM essentially consists of two parts. The first part is an encoder that converts raw data into concepts, and the second part is a linear layer that maps these concepts to the final category. In this attack, instead of working directly on the inputs, it operates on the converted concepts. During attack, a trigger function is used to apply a predefined static trigger to the concept, causing it to be misclassified into a specific target class.

From my perspective, the second part of this process resembles a traditional backdoor attack against a DNN. Although traditional backdoor attacks on DNNs typically target images, we can always flatten an image into a 1D vector (like concepts in CBMs) and then apply a static trigger. This approach seems nearly identical to the CBM attack.

If that is the case, the innovation of this paper appears to be very limited, as no fundamentally new attack or scheme has been proposed. Additionally, it is important to validate this attack using traditional backdoor detection methods, such as Neural Cleanse or other trigger reverse engineering techniques. Even if the trigger cannot be detected in the image domain, it should still be detectable in the concept domain. Moreover, because the trigger is static and obivious (replacing concept values to 1 in their example), I speculate it will be very easy to be detected by many backdoor detection schemes.

### Questions
Please refer to weaknesses and justify why CBM backdoor is different than traditional backdoor attacks.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper is makes the first attempt to investigate model backdoor threats targeting Concept Bottleneck Models (CBMs). To address this, it introduces a concept-level backdoor attack, called CAT, which embeds backdoors by injecting triggers into the concept layer during training. Both empirical and theoretical analyses are provided to demonstrate the effectiveness of the proposed attack.

### Strengths
- This paper shows a pioneering effort in investigating backdoor threats against CBMs.
  
- It provides a well-rounded analysis of the attack, encompassing both empirical evidence and theoretical insights.

### Weaknesses
### 1. Writing Quality

The authors' attempts to make the paper visually engaging, such as including a cute icon and a notable saying at the start, are appreciated. However, while these elements add charm to the introduction, the main body lacks the same level of engagement. I would encourage the authors to focus more on enriching the scientific content, rather than on decorative elements.

- **Clarity of Positioning**: My primary concern is that the paper’s positioning is unclear. When reading the introduction, my initial impression is that it applies backdoor attacks to a new model type without a clear differentiation. What sets this work apart from existing backdoor attack research, especially in terms of methodology? Simply stating “the first to xxx” does not adequately establish uniqueness. The authors should discuss recent advancements in this area, identify the research gap, and explain how this paper addresses specific challenges.

- **Improving Signal-to-Noise Ratio in Writing**: Increasing the proportion of informative content could make the paper even more engaging, as readers often look for in-depth insights. For instance:

  - The sentence starting on Line 80 ("The fundamental ...") describes a standard backdoor attack with common stealth requirements, similar to invisible backdoor attacks. Adding specific details about the proposed CAT method here would help clarify its distinct contributions.

  - Likewise, the paragraph starting at Line 93 might seem broadly applicable to other machine learning models if we remove "concept-level" and substitute "CBMs" with other models. Highlighting the unique aspects of this problem and clarifying why CBMs' security deserves special attention, especially given the many available XAI methods, could make the impact of the work more apparent.

- **Other Writing Issues**:

  - The symbol $y_{tc}$ first appears in Equation (2), but its definition is not provided until after Equation (3). Consider moving its definition to follow its initial appearance.
  
  - In Equation (3), the first two instances of $f$ should perhaps be $g$, and there’s an extra right parenthesis in the constraint.
  
  - In Algorithm 1, Line 12, $\mathcal{D}'_{adv}$ should likely be $\tilde{\mathcal{D}}\_{adv}$.

### 2. Threat Model Clarity

When introducing an attack, it is essential to present a clear threat model outlining the attacker’s goals and capabilities. A threat model allows readers to assess the feasibility and severity of an attack. However, this paper lacks a distinct threat model, which makes the motivation for concept-level backdoor attacks unclear. A key feature of backdoor attacks is the attacker's ability to actively trigger backdoored behavior by inserting a backdoor. However, as stated on Line 495, the attacker cannot actively control trigger injection since they lack direct access to the concept space. This scenario resembles conventional poisoning attacks aimed at model degradation, with the only difference being that this is a dirty-label poisoning attack occurring at the concept layer.

*PS.* I am familiar with backdoor attacks but have only basic knowledge of concept bottleneck models. Please correct any potential misunderstandings.

### 3. Limited Evaluation Scale

The evaluation is primarily limited to a pretrained model and two datasets, which, despite promising results, restricts the robustness of findings. A broader evaluation across additional models and datasets would strengthen the evidence of the proposed attack's effectiveness and generalizability.

### 4. Absence of Defensive Strategies

The paper does not explore or discuss potential defense mechanisms to counter the proposed attack.

### Questions
- What are the fundamental differences between the proposed CAT method and existing dirty-label poisoning attacks on classification models?

- What specific challenges does the proposed CAT method aim to overcome?

- Can you provide an estimate of the trigger size range that would maintain stealthiness?

### Soundness
2

### Presentation
2

### Contribution
2
