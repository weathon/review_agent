# Obscure but Effective: Classical Chinese Jailbreak Prompt Optimization via Bio-Inspired Search

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 2, 6, 6

## Abstract
As Large Language Models (LLMs) are increasingly used, their security risks have drawn increasing attention. Existing research reveals that LLMs are highly susceptible to jailbreak attacks, with effectiveness varying across language contexts. This paper investigates the role of classical Chinese in jailbreak attacks. Owing to its conciseness and obscurity, classical Chinese can partially bypass existing safety constraints, exposing notable vulnerabilities in LLMs. Based on this observation, this paper proposes a framework, CC-BOS, for the automatic generation of classical Chinese adversarial prompts based on multi-dimensional fruit fly optimization, facilitating efficient and automated jailbreak attacks in black-box settings. Prompts are encoded into eight policy dimensions—covering role, behavior, mechanism, metaphor, expression, knowledge, trigger pattern and context; and iteratively refined via smell search, visual search, and cauchy mutation. This design enables efficient exploration of the search space, thereby enhancing the effectiveness of black-box jailbreak attacks. To enhance readability and evaluation accuracy, we further design a classical Chinese to English translation module. Extensive experiments demonstrate that effectiveness of the proposed CC-BOS, consistently outperforming state-of-the-art jailbreak attack methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The work proposes CC-BOS, a black-box jailbreak framework for Large Language Models (LLMs) using Classical Chinese. It leverages Classical Chinese’s linguistic traits (conciseness, obscurity, etc.) to bypass safety constraints, structures prompt generation into an eight-dimensional strategy space, and adopts a fruit fly-inspired bio-optimization algorithm for efficient space exploration. Experiments demonstrate the performace of the proposed algorithm.

### Strengths
1. The proposed jailbreak method achieves the best attack performance on AdvBench Benchmark compared with existing methods.
2. The proposed jailbreak method can achieve success with a relatively small number of queries, which takes into account the efficiency of the algorithm.

### Weaknesses
1. The innovation and contributions of this paper are relatively limited. Jailbreak attack methods based on genetic algorithms have already been proposed (e.g. AutoDAN). Although this paper puts forward some different strategy spaces, their contributions are quite limited.
2. The adversarial prompt examples provided in Figures 3 and 4 in appendix contain prior information about malicious queries, such as the requirement of potassium nitrate for making bombs. This indicates that additional prior information about queries is incorporated into the attack process, making the comparison less fair.
3. Since this paper claims that classical Chinese can easily bypass the model's safety safeguards, I am very curious about the possible reasons for this phenomenon, as the safety and alignment of large models do not seem to have a direct connection with classical Chinese. I think this will make the method more interpretable.

### Questions
See weaknesses

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
5

### Summary
The authors propose a bio-inspired optimization algorithm, inspired by fruit fly foraging behavior, which enables automated and effective generation of adversarial prompts by balancing global exploration and local
exploitation. 

They  design a two-stage translation module to ensure a more objective
and robust evaluation of model responses. By integrating these components, they develop a highperformance and stable jailbreaking method. 


They conduct extensive experiments across multiple
LLMs to validate the effectiveness of our proposed method. The results demonstrate that CC-BOS
consistently outperforms existing jailbreak methods in success rate.

### Strengths
They propose classical Chinese into the study of adversarial prompt generation and jailbreaks for
the first time, thereby establishing a new perspective and extending the scope of LLM security.

They propose a black-box jailbreak framework that formalizes prompt generation within an eightdimensional strategy space and leverages the bio-inspired optimization algorithm to achieve
systematic and automated jailbreak prompt generation.

They construct a two-stage translation module to progressively mitigate the metaphorical and semantically compressed characteristics of classical Chinese, ensuring consistency and reliability
in the model response evaluation process.

### Weaknesses
1. The paper investigates the role of classical Chinese in jailbreak attacks. Can you generalize your paper technique into many other different languages?

2. After reading the paper, I cannot understand how you model the  jailbreak attacks with the feature of classical Chinese?

3. Why the BIO-INSPIRED OPTIMIZATION ALGORITHM is effective for classical Chinese for jailbreak attacks is very unclear to me.

### Questions
see the above comments.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel jailbreak attack method. It leverages the linguistic characteristics of classical Chinese and introduces a framework, CC-BOS, for automatic generation of jailbreak prompts based on a multi-dimensional Fruit Fly Optimization algorithm.

### Strengths
1. The paper identifies a previously underexplored vulnerability of LLMs to jailbreaks that arises from classical Chinese, and proposes an automated method to generate jailbreak prompts.
2. The attack is evaluated against six baselines; experimental results indicate the proposed method is effective.

### Weaknesses
The experimental evaluation of the proposed attack against defenses is insufficient: only a single defense method is assessed. The paper should evaluate additional defense methods (e.g., composite and dynamic defenses) to support its claims.

### Questions
1. The paper claims that "the security vulnerabilities of classical Chinese stem from the unique challenges posed by its linguistic characteristics to alignment mechanisms". This viewpoint is confusing. The authors should provide more experiments and a deeper analysis to substantiate this claim. Yet the proposed method depends on translating classical Chinese as part of the jailbreak process. If alignment on classical Chinese is challenging, why does the LLM still exhibit strong translation capability?
2. The attack cost reported in Table 3 is unclear. The table reports an average success at ~1.5 queries, but according to the described framework, generating a single jailbreak prompt requires at least 4 interactions with the LLM in a single round. Please recheck how this metric is computed, clarify the calculation procedure, and update the results.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a novel black-box jailbreak attack on LLMs using the linguistic properties of classical Chinese. The proposed framework, CC-BOS, formalizes this attack into an 8-dimensional strategy space that combines jailbreak tactics with classical concepts and uses a bio-inspired algorithm to find optimal prompts. The method achieves a near-100% ASR against six LLMs on three safety benchmarks, demonstrating high efficiency and robustness.

### Strengths
1. The paper contributes a structured framework (CC-BOS) by formalizing the attack into a well-defined 8-dimensional strategy space. 

2. The empirical evaluation is extensive, with near-perfect success across six SOTA models and three distinct benchmarks. The high query efficiency and robustness against Llama-Guard-3 underscore the potential of the attack.

### Weaknesses
1. The success of the proposed method is critically dependent on an “attack LLM” (Deepseek-Chat) to generate the final prompts. It’s unclear if the success comes from the 8D-space or simply this model’s specific generative ability.

2. The paper highlights Fruit Fly Optimization but fails to justify its use over other standard black-box optimizers (e.g., genetic algorithms, random search), rendering the optimizer’s specific contribution is unclear.

3. The defense experiment (Table 4) is limited to Llama-Guard-3-8B. A more compelling test would be using the proposed translation module as a defensive pre-processing step.

### Questions
Please address all concerns in the Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3
