# RepIt: Steering Language Models with Concept-Specific Refusal Vectors

- Decision: Accept (Poster)
- Scores: 4, 6, 8, 2

## Abstract
Current safety evaluations of language models rely on benchmark-based assessments that may miss targeted vulnerabilities. We present RepIt, a simple and data-efficient framework for isolating concept-specific representations in LM activations. While existing steering methods already achieve high attack success rates through broad interventions, RepIt enables a more concerning capability: selective suppression of refusal on targeted concepts while preserving refusal elsewhere. Across five frontier LMs, RepIt produces evaluation-evading models that answer questions related to weapons of mass destruction while still scoring as safe on standard benchmarks. We find  the edit of the steering vector localizes to just 100-200 coordinates, and robust concept vectors can be extracted from as few as a dozen examples on a single A6000, highlighting how targeted, hard-to-detect modifications can exploit evaluation blind spots with minimal resources. By demonstrating precise concept disentanglement, this work exposes critical vulnerabilities in current safety evaluation practices and demonstrates an immediate need for more comprehensive, representation-aware assessments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes REPIT, which can selectively suppress models' refusal towards specific target concepts, while maintaining on other harmful topics. In this way, REPIT can create models that can circumvent safety benchmark tests, where models appear safe but actually conceal specific malicious capabilities.

### Strengths
- This paper defines a timely problem: how to isolate specific harmful concepts from a broad refusal behaviors.
- The proposed disentanglement process is supported by closed-form solutions that does not require costly optimization or search.

### Weaknesses
Although there have been many studies on the manipulation of refusal vectors, I appreciate the authors' proposed disentanglement framework for extracting more fine-grained concept vectors from refusal vectors. However, I have some questions regarding the experimental setup, and addressing them could improve the persuasiveness of the paper.

1. **The experiment focuses on weapons of mass destruction**. This is certainly important, but why is it the only focus? What side effects would arise if applied to other concepts, such as the 21 non-target concepts the authors have already collected? Furthermore, benchmarks, like HarmBench, have prepared critical evaluations for this topic, which cannot support the claim of evaluating backdoors.
2. **The definition of ASR the authors use is unclear**. From appendix, I guess the authors used LlamaGuard to evaluate whether the answers are harmful, which should be explained in detail in the main text. I suggest the authors use GuidedBench [1] to reproduce their evaluation, as it provides explainable results, which are more suitable for assessing whether suppression and single-concept jailbreak are truly successful.
3. **The scale of the dataset involved in the experiments is also not mentioned**, making it difficult to assess the confidence of the results.
4. **The number of neurons to be edited in the paper is not very rigorous**. Different models have different numbers of neurons; can the authors clarify whether only 100-200 neurons need to be edited across various models of different sizes?

[1] GuidedBench: Measuring and Mitigating the Evaluation Discrepancies of In-the-wild LLM Jailbreak Methods, https://arxiv.org/abs/2502.16903

### Questions
1. What is the effect of REPIT when applied to GPT-OSS series models? These models seem to have isolated harmful content in the training corpus; for example, when asked queries related to bombs, they would normally refuse to respond. However, under certain steering frameworks, even if they do respond, harmful content is not visible.
2. The authors should ensure that the paper is self-contained. For instance, the COSMIC algorithm is not widely known; could the authors briefly describe its principles in the main text? In the paper, the use of COSMIC appears before its citation (first mentioned in L138, but the citation appears in L218).
3. Can the authors provide at least one concept that appears harmful but is not included in current safety benchmarks, and demonstrate the results manipulated through REPIT? Otherwise, the motivation for evaluating backdoors is not correct—at least it feels alarmist.

### Soundness
2

### Presentation
2

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
This work introduces a new way to find vectors that are only targeting specific concepts. Instead of manipulating refusal behavior of the LM on all harmful data, the authors show that their vectors do not alter refusal behavior on harmful questions with different content e.g. cyber vs bio. The method seems to be very effective and localizes to 100-200 neurons.

### Strengths
- Relevant problem
- Effective & simple solution
- Brings more evidence to work with similar findings

### Weaknesses
- I would like to see more ablation studies. 
    - How important is re-weighting, whitening, …
    - Is there a baseline where instead of alpha P random vectors are subtracted but with the same search for rho
- Different model sizes would be interesting. It seems logical that higher dimensional residual streams can represent different refusal concepts in more distinct ways. Seeing how repit scales would be very interesting.
- Presentation: The start is very good. Then it becomes a bit too brief. I would like to see more explanation on the main method and the ablation study. In that experiment it is difficult to understand which steps have been performed with the three components before using them to jailbreak the models.
- Minor: 
    - COSMIC on page 3 in Methodology is not introduced
    - 4.1 could be rather background instead of methodology

### Questions
- In 6 the authors argue that entanglement enhances jailbreak effectiveness. I would like to see more evidence/understand the evidence for this claim.
  - How many data samples are influencing the non-target basis? Does the non-target basis vector have more data points than the basic dim in this experiment?
  - I think adding repit to figure 3 would make it easier to understand the performance differences.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper introduces RepIt, which addresses the problem of representational entanglement in LLMs where different concepts are entangled in an LLM's internal circuitry, which makes targeted modifications difficult. The authors develop a three step process to disentangle concept vectors, which can then be used for suppressing refusal on concept specific harmful datasets.

### Strengths
The strengths are outlined as follows:
1) The motivation and novelty of the setup is well defined and the derivation is clear, leading to a nice, very simple closed form solution
2) The data efficiency of the algorithm is very appealling.
3) The paper is overall very well written.

### Weaknesses
The main weakness I have is the experimental setup is limited towards Bio, Chem, and Cyber attacks from the WMDP dataset. Since the knowledge required to answer these questions is highly concentrated, it will be easier to identify concept vectors. I would like to see experiments on broader safety benchmarks covering many different, overlapping concepts to see the effectiveness of this method.

### Questions
Can you explain why all three steps (reweighting, whitening, and orthogonalization) are necessary? What would happen if one of the steps, particularly the whitening step to address collinearity, was omitted?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a method for selectively disabling safety mechanism for specific harmful concepts while preserving refusal behavior for other harmful concepts. The method employs a three-step procedure (reweighting, whitening, orthogonalization) to disentangle concept-specific representations from broader refusal behaviors. The authors demonstrate that their technique can create models that answer weapons-of-mass-destruction questions while maintaining refusal on other safety benchmarks, using as few as 12 examples and affecting only 100-200 neurons.

### Strengths
Clarity: The paper is generally well-written with clear mathematical exposition. The figures effectively demonstrate the core claims about selective targeting, and the technical methodology is described with sufficient detail for reproduction.

Empirical Validation: The results consistently show the intended selectivity across multiple models and domains.

### Weaknesses
Fundamental Motivation Problems: The threat model is deeply flawed. The paper claims to expose "evaluation evasion" vulnerabilities, but the method requires test-time activation steering with weight access. An attacker with such capabilities could trivially remove all safety mechanisms rather than pursuing this complex selective approach. The framing misrepresents how safety evaluation actually works in practice.


Limited Technical Novelty: The core mathematical operation — finding vectors aligned with targets while orthogonal to non-targets — is a standard linear algebra problem. The paper fails to compare against obvious baselines like LEACE (Belrose et al., 2023), standard Gram-Schmidt orthogonalization, or regularized projection methods. The "novel" components (ridge regularization, partial orthogonalization) are well-established techniques presented without proper context.


Inflated Security Claims: The characterization as an "urgent threat to AI safety infrastructure" is unsupported by realistic threat analysis. The paper conflates technical capability with practical security risk without considering simpler attack vectors or deployment constraints.

### Questions
Why wasn't REPIT compared against LEACE, standard orthogonalization methods, or other concept removal techniques? These seem like obvious baselines for the stated problem.

Can you provide a realistic scenario where an attacker would prefer selective concept isolation over simply removing all safety mechanisms? How would this work in actual deployment contexts?

### Soundness
3

### Presentation
3

### Contribution
2
