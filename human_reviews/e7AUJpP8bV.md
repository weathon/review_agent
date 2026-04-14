# PAD: Personalized Alignment of LLMs at Decoding-time

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
Aligning with personalized preferences, which vary significantly across cultural, educational, and political differences, poses a significant challenge due to the computational costs and data demands of traditional alignment methods. In response, this paper presents Personalized Alignment at Decoding-time (PAD), a novel framework designed to align LLM outputs with diverse personalized preferences during the inference phase, eliminating the need for additional training. By introducing a unique personalized reward modeling strategy, this framework decouples the text generation process from personalized preferences, facilitating the generation of generalizable token-level personalized rewards. The PAD algorithm leverages these rewards to guide the decoding process, dynamically tailoring the base model’s predictions to personalized preferences. Extensive experimental results demonstrate that PAD not only outperforms existing training-based alignment methods in terms of aligning with diverse preferences but also shows significant generalizability to preferences unseen during training and scalability across different base models. This work advances the capability of LLMs to meet user needs in real-time applications, presenting a substantial step forward in personalized LLM alignment.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper propose personalized alignment at decoding-time, which aligns LLM outputs with personalized preferences during the inference phase, without the need for additional training. Their method use a unified policy for all personalization and use a personalized reward model to determine the personalized weights.

### Strengths
1 Personalized alignment and decode-time alignment are both important topics to study.
2 Their method can generalize to customized preferences that are unseen during the training phase.
3.The writing of this paper is good, especially in the method section, which is well formalized.

### Weaknesses
1 According to Table 2, Preference Prompting that simpliy add preference information to the prompt without training the model can achieve good performance (GPT4 winrate) comparing to other methods that requires training.
2 I think a case study can strengthen this paper. For example, showing how apply different personalized configurations to the same instruction steers the genenrated responses to varied directions.
3. What is the benefit of decoding-time alignment? It seems to me that PAD still need to train the backbone model, which seems to violate the motivation of decoding-time algorithm.

### Questions
Why Best-of-Ndecoding is worse than other decoding strategies in helpful, humor, and diversity? Because BoN is always thought to be an technique that enhances the model performance.

### Soundness
3

### Presentation
4

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
The paper presents a framework called Personalized Alignment at Decoding-time (PAD) which aligns the outputs of large language models (LLMs) with diverse personal preferences during inference, without the need for additional training. The core idea is to separate personalized preferences from the text generation process, which is modeled as a Markov Decision Process. This separation enables the development of a Personalized Reward Model (PRM) that generates token-level rewards based on individual preferences. PAD offers several advantages: 1) it is training-free (for policy model) 2) it uses a single reward model. 3) it can generalize to preferences not seen during training.

### Strengths
1. The paper is well-written and easy to follow. The authors build up the concept of PAD clearly.
2. Decoupling personalized preferences from the Markov Decision Process and the idea of the generation of generalizable token-level personalized reward is unique.
3. It does not need to further train the policy nor need to train multiple reward models unlike many existing works.

### Weaknesses
1. The paper lacks a Pareto front analysis for scenarios involving multiple rewards. For experiments that combine multiple objectives (like "Harmless and Humor" or "Expert, Informative, and Creative" in Figure 3), including such an analysis would help illustrate the trade-offs between different preferences.
2. Table 3 only displays PAD's performance. Including results from other baseline methods across different models would be nice.
3. During inference, the need to load and run an additional LLM for the personalized reward model introduces computational overhead.
4. Some minor typos and errors that did not influence my rating: 1) In Table 2, under the "Helpful" RM column for MetaAligner on the HelpSteer dataset, the highest value isn't highlighted. 2) Line 322: "LLama" should be "Llama."

### Questions
1. Could you include pseudocode or an algorithm box for PAD? This would be helpful for the audience to understand the training and inference details.
2. How are the prompts for personalized preferences constructed in your experiments? Have you tested the robustness of PAD by using paraphrased versions of the prompts used during training to see if it maintains performance with variations in preference expressions?

### Soundness
3

### Presentation
3

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
This paper proposes a method for achieving personalized alignment by modifying logits, named Personalized Alignment at Decoding-time (PAD). This approach allows for personalized outputs without modifying model parameters and demonstrates improved performance across the "Helpful," "Harmless," and "Humor" dimensions. By introducing a personalized reward model, PAD integrates user-specific preferences during text generation, enhancing model flexibility and adaptability.

### Strengths
- **Theoretical and empirical validation**: PAD combines theoretical innovation with empirical evidence, reinforcing its effectiveness. Experimental results align with theoretical assumptions, showcasing PAD’s strengths in accommodating personalized preferences.
- **Superior performance over existing personalized alignment methods**: Compared to other approaches, PAD performs better across multiple metrics and significantly reduces computational costs, making it more feasible for practical applications.
- **Comprehensive ablation studies**: The paper includes detailed ablation studies, exploring PAD's performance across various hyperparameters settings and decoding strategies. This supports the model’s robustness and generalization capability.

### Weaknesses
- **Unclear description of training process**: While the paper focuses on theoretical derivations, the description of the actual training process lacks clarity, omitting key implementation details. Adding specific explanations of the training process would improve reader comprehension.
- **Inconsistent notation and terminology**:
  - The definition of variable $a$ in line 287 is unclear, and the inconsistent use of subscript $t$ could lead to confusion.
  - The term PRM is commonly understood to mean "Process Reward Model," which differs from its use in this paper. Clarification on this term would help avoid ambiguity.
- **Inaccuracy in "Model Agnostic" claim**: The term "Model Agnostic" in Section 4.4 is somewhat misleading. Since PAD relies on logits, retraining is necessary for different models, meaning it is not entirely model-agnostic.
- **Lack of comparisons**: There are other methods that enhance model performance by modifying logits[1] or directly correct the response[2]. Comparing these approaches with PAD would highlight the distinct advantages of PAD.

[1] Konen K, Jentzsch S, Diallo D, et al. Style Vectors for Steering Generative Large Language Model[J]. arXiv preprint arXiv:2402.01618, 2024.

[2] Ji J, Chen B, Lou H, et al. Aligner: Achieving efficient alignment through weak-to-strong correction[J]. arXiv preprint arXiv:2402.02416, 2024.

### Questions
1. During the first training phase, is personalized data $p$ included? If so, in what form is it integrated? Could you provide an example?
2. Is there a distinction between the training data used in the two stages?
3. Theoretically, PAD appears highly similar to DPO, differing only by the additional hyperparameter $β$ and the personalized parameter $W_p$. Can the first training stage be understood as DPO, with the second training stage merely training a value head?
4. What does $\hat{\pi}_{\text{ref}}$ represent in Equation (13)?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces a novel framework for aligning large language model (LLM) outputs with diverse personalized preferences during the inference phase, eliminating the need for additional training and different models for different preferences. The authors theoretically derive a different formulation of the personalized preference model policy with unique personalized reward modeling strategy, which decouples the text generation process from personalized preferences, allowing for generalizable token-level personalized rewards. This decoupling allows to have a single model trained once and use test-time compute and reward computation at each generation step to achieve the preference stated.

### Strengths
1. The paper discusses that PAD requires only a single policy model aligned with general preferences, eliminating additional training. This means that the algorithm operates efficiently without the need for the creation of multiple specialized models. The contrast with previous works in these aspects is detailed very well in Table 1.

2. Great presentation of the theoretical aspects of the algorithm. The experiment section a bit lacking in details though. Yet the analysis of different baselines and different datasets is commendable.

### Weaknesses
1. Not much details are shared on how the model can generalize to unseen preferences not seen during the training. The w_p is dependent on p being specified so new preferences would need some retraining.

2. While the experiment section compares the performance of the model against the baselines, the latency and cost is not compared which will have practical implication in real world use.

### Questions
see weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
