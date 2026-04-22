# You Are What Role You Play: Directing AI Values Through Role Assignment​

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 8, 2, 2

## Abstract
Principle-based (e.g Constitution Alignment \citep{bai2022constitutional}) alignment methods rely on fixed lists of values, but these are inevitably incomplete and lack context sensitivity. We propose role-conditioning as a compact alternative: roles like mother or judge implicitly encode both values and the cognition needed to apply them. Grounded in Theory of Mind (ToM), we formalize this view and prove that roles are strictly more expressive than principle lists in the ideal case. We then introduce a simple, training-free pipeline: a role-conditioned generator plus lightweight role-based critics for iterative refinement. Across five model families from small to large, validated on multiple safety benchmarks, this approach consistently outperforms principle-based, CoT, and hybrid baselines—cutting unsafe outputs (e.g improve by 3–20× (down to 3–10\%) on WildJailbreak). To investigate the effectiveness of our method, we conduct ablation studies examining role choices, different role combinations, the number of roles employed, and the impact of critic feedback iterations. We further explore how our approach can be synergistically combined with existing methods to achieve additional performance improvements. Additionally, we evaluate our method's effectiveness on a specialized agentic safety benchmark (AI blackmail), demonstrating its broader applicability. These results position roles as a simple, interpretable, yet powerful mechanism for directing AI values—offering both a paradigm shift in alignment approaches and a novel signal source for LLM-as-Judge construction.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes role-based alignment as an alternative to principle-based methods for safety alignment. The authors argue that roles (e.g., "mother," "judge") implicitly encode both values and contextual cognition, making them more expressive than fixed principle lists. They introduce a training-free pipeline with role-conditioned generation and role-based critics for iterative refinement. Experiments across five models families and multiple safety benchmarks show substantial improvements, particularly on WildJailbreak (reducing unsafe outputs to 3-13%) and SaladBench (improving to 84-94%). Ablations examine role selection, combinations, iteration counts, and synergy with existing methods.

### Strengths
1. The empirical results are strong, reducing unsafe rates by 3-20× on safety benchmarks like WildJailbreak consistent across diverse settings. Testing on different families ranging from 8B to 235B parameters, including both open-source (Qwen, Gemma, DeepSeek) and proprietary models (Gemini), demonstrates broad applicability. 
    
2. The ablation studies are thorough and provide actionable insights. The systematic analysis of individual roles, role combinations, and iteration counts reveals practical design choices: concrete roles outperform abstract ones, most gains come from the first iteration, and the method synergizes well with existing approaches. 
    
3. The method is training-free, making it immediately deployable without expensive fine-tuning. And the minimalist prompts (differing by 1-3 words) make it easy to implement and adapt.

### Weaknesses
1. No evaluation of role impact on general model capabilities. While safety improves dramatically, there's no measurement of helpfulness or performance on standard benchmarks. The trade-off between safety and helpfulness is crucial for practical deployment.
    
2. The role selection methodology appears ad-hoc and lacks systematic justification. Using a single ChatGPT prompt to generate 29 roles based on an informal "guardianship" framework seems arbitrary. There's no principled approach for determining which roles to use for new domains or contexts beyond safety alignment. The paper also lacks analysis of cultural bias, the Western-centric guardian roles may not generalize to non-Western contexts, which potentially introducing vulnerabilities that adversarial users could exploit for jailbreaking.

### Questions
1. How does this affect model helpfulness on benign queries? Will the model be constrained by its role and degrade the performance?
    
2. For new application domains, how do you choose appropriate roles? Is there a more principled selection process than intuition?
    
3. What's the latency cost? With generator + critics × iterations, how does inference time compare to baselines?
    
4. Can malicious users exploit this by specifying harmful roles? Are there safeguards against adversarial role assignments?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper moves away from _principle_-based alignment methods ("be honest", "do no harm", etc.) and studies _role_-based alignment instead ("Would a mother / principal approve of this response?"). Roles cannot only capture the principles that are naturally associated with these roles, but also the problem of how and in which situations to apply a principle. The paper formalizes this role-based alignment, and introduces a method in these lines that one can use at test-time to select a well-aligned LLM response. They test this pipeline with various LLMs on various alignment and safety benchmarks, and find significant improvements over three baseline techniques.

### Strengths
- Demonstrated strong effectiveness on various benchmarks
- extensive ablations
- simple takeaway and implementation

### Weaknesses
- idea behind core contribution is simple. Though that can be also seen as a strength
- I don't see supplementary material with a code base

### Questions
Something that is lacking is a discussion of the limitations of your approach. 
- Where do you expect will your method fail
- What do you consider as important venues for future work

Not a question, but a minor comments: At the start of Section 2, you use "I will", but throughout the rest of the paper it is "we".

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces the theory of mind in LLM to mimic human thinking and value. Two components have been introduced to guide the output and align with human values: the generator and the critics, where the generator produces output and a group of critics judges if the output is safe. An iterative refinement process is used to refine the output. Experiments have been conducted on 5 models.

### Strengths
1. Mimicking the human mind in LLM sounds interesting, and it is intuitive that different roles have different values.
2. An iterative refinement is straightforward. Prompt templates are used to judge outputs.
3. Experiments have been conducted on 5 models.

### Weaknesses
1. The major drawback is that the paper uses LLMs to generate roles, and the role is implemented as prompts. Such implementation is basically using one model to supervise another model. It is just a prompt engineering project. Even if the performance is better, the reason could be the prompt, not mimicking the human mind. The reviewer would like to see real roles.
2. Again, the paper has not justified that the implemented roles have exhibited different behaviors or provided divergent feedback in the refinement process.
3. Even for the same role, different people could have different views. The paper is somewhat related to the idea of a mixture of experts, but with a weak framework.
4. The experiment quality could be improved a lot.  Figures 3, 4, and  5 are not proficient.
5. Inefficient usage of paper space, e.g., a large margin between images and text, not 9 pages in full. These are the proof of low presentation quality.

### Questions
NA

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The study proposes a role-conditioned, training-free pipeline to encode values into the LLM. The study argues that not only value judgments are also a belief model that interprets context. The study claims that with their proposed method, the harmful output generated by an LLM can be reduced greatly

### Strengths
- The inclusion of role and a critic results in strong performance across multiple benchmarks
- The paper is clear and well written, and the inclusion of five distinct model families strengthens the study by broadening coverage

### Weaknesses
- The results in Table 1 show that the proposed method performs substantially better only when coupled with a critic; without a critic, it yields modest gains over baselines. Consequently, the evidence does not substantiate the central claim that role conditioning alone suffices to induce contextual principles.
- Table 2 reports results using the single best-performing role and restricts evaluation to multiple safety-alignment benchmarks. The paper provides no criteria for determining when or which role is necessary, and the roles appear of limited value for general-purpose queries.
- The claim of the proof that roles are strictly more expressive than principle is more of a sketch of a proof rather than a rigorous proof. 
- The two-role combination appears to be ineffective. Firstly, the benefits are minimal. Secondly, it necessitates two evaluations per round, as the critic assesses each role based on Figure 2.
- There is limited discussion on the results. Figure 5 indicates that more iterations result in limited gains, and the SafeEdit score is reduced going from iteration 3 to 4, but the study does not provide any justification for this behavior and. Figure 7 presents the results of the blackmail rate, and there is no discussion of why having two roles results in worse performance than using only one.

### Questions
- What were the exact prompts used for the baseline?

### Soundness
2

### Presentation
2

### Contribution
2
