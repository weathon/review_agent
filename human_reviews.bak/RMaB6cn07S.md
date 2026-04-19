# Generating Equivalent Representations of Code By A Self-Reflection Approach

- Decision: Reject
- Scores: 3, 5, 6, 3

## Abstract
Equivalent Representations (ERs) of code are textual representations that preserve the same semantics as the code itself, e.g., natural language comments and pseudocode. ERs play a critical role in software development and maintenance. However, how to automatically generate ERs of code remains an open challenge. In this paper, we propose a self-reflection approach to generating ERs of code. It enables two Large Language Models (LLMs) to work mutually and produce an ER through a reflection process. Depending on whether constraints on ERs are applied, our approach generates ERs in both open and constrained settings. We conduct a empirical study to generate ERs in two settings and obtain eight findings. (1) Generating ERs in the open setting. In the open setting, we allow LLMs to represent code without any constraints, analyzing the resulting ERs and uncovering five key findings. These findings shed light on how LLMs comprehend syntactic structures, APIs, and numerical computations in code.
(2) Generating ERs in the constrained setting. In the constrained setting, we impose constraints on ERs, such as natural language comments, pseudocode, and flowcharts. This allows our approach to address a range of software engineering tasks. Based on our experiments, we have three findings demonstrating that our approach can effectively generate ERs that adhere to specific constraints, thus supporting various software engineering tasks.
(3) Future directions. We also discuss potential future research directions, such as deriving intermediate languages for code generation, exploring LLM-friendly requirement descriptions, and further supporting software engineering tasks. We believe that this paper will spark discussions in research communities and inspire many follow-up studies. The source code and data are available.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The authors propose a dual framework for producing Equivalent Representations (ERs) (e.g., NL comments, pseudocode, flowcharts) for code through multiple trials. In each trial, the first LLM call is used to generate an ER based on the input code and then a second LLM call is used to generate a new code snippet based on the ER. Then, the score is computed between the original code and the new code snippet (as a combination of text similarity, syntactic similarity, and external constraint-based satisfaction score derived from the LLM). This score is then translated into natural language feedback that is used in the subsequent trial to improve the ER. The authors perform this procedure for the 500 Python examples in the CoNaLa dataset and then do a qualitative analysis in two settings: open setting (no constraints on the ER that the model can produce), constrained setting (ER must be of a specific type, i.e., comment, pseudocode, flow chart).

### Strengths
- While the dual framework for code isn't itself novel (https://arxiv.org/abs/2310.14053, https://arxiv.org/abs/2402.08699), the idea of translating the similarity metric to NL feedback and performing iterative refinement is quite novel and clever. The iterative approach also seems to help improve equivalence scores, based on Figure 2.
- I particularly liked the idea of scoring based on text similarity, syntactic similarity, and external constraint satisfaction. 
- There are many examples provided in the main paper, and the examples in the open setting demonstrate that the LLM is capable of generating many diverse types of ERs based on code.

### Weaknesses
- Many of the claims in this paper are not substantiated with empirical evidence. First, in L239-240: "We think that the forms LLMs represent the code can reflect how they understand the code." The authors claim the findings in the open setting "reveal how LLMs understand different elements in code, e.g., grammatical rules, APIs, and numerical calculations." This claim is not supported with any evidence. In fact, just because a model is capable of generating diverse types of ERs doesn't necessarily say anything about whether or not it understands code in that form or that it reasons in that form. For making such a claim, the authors should have run experiments to compare performance when the input or intermediate CoT reasoning chain leverages these different types of ERs. The authors also claim, "From the structured ERs, we find that LLMs treat the code as a structured sequence instead of plain text. LLMs even can produce a plausible syntax tree." However, if the input was something other than code like general text and you still ask it to transform it into some new representation, there's a good chance that it will also be structured. This claim is not well-supported.
- Next, Finding 6 is "Our approach effectively generates ERs for software engineering tasks and reduces hallucinations generated by LLMs." This seems to be based solely on a speculation in L468-470: "We can see that the ERs in the first trial have lower scores and are often problematic. It may be due to the hallucinations from LLMs, where LLMs generate incorrect ERs though they can understand the code instructions." The authors do not provide any evidence of hallucination. In fact, this could also be due to a number of other factors. For instance, it's possible that the LLM didn't capture \textit{all} of the information in the first trial and later added more information. Or that the phrasing or structure of the ER in the first trial was not meaningful to the LLM and it just needed some simple refactoring.
- In Findings 7 and 8, the authors claim that their approach is useful to software developers; however, they do not actually perform any user study to confirm this.
I find that this paper lacks key empirical results. The authors have reported their observations from performing qualitative analysis themselves, but I feel that these are not convincing on their own.
- The authors have only included 1 paper in the Related Work section. I feel that a more thorough literature review is needed. Particularly, there are other papers that use a dual framework for code (https://arxiv.org/abs/2310.14053, https://arxiv.org/abs/2402.08699).

Suggestion:
- The conclusion suggests that future work is presented in detail. However, this is only explained in the introduction. Also, highlighting future work so extensively is not needed in the abstract or introduction. It is better to focus on the approach itself.

### Questions
- Is the code and ER not included in the prompt for getting NL feedback (Figure 5)?
- What was the average number of trials? What was the maximum number of trials? How was the threshold score determined?
- Is it possible for the type of ER to change across trials (i.e., in Trial 1 it is a comment, in Trial 2 it is pseudocode)? If so, how often does this happen?

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces a dual framework to generate the ERs of input code. Experiments demonstrate the effectiveness of the proposed approach and the authors also discuss several potential applicaitions into the software engineering tasks.

### Strengths
+ The idea of generating the ERs of code is somewhat interesting.

+ Many vivid examples are provided in the paper.

### Weaknesses
- Some doubts about the definitions. What is an ER? The paper merely lists several possibilities without an explixt definition. Also, I wonder can the pseudocode be considered as natural languages, as argued by the authors in the Abstract?

- Unclear about the state of the art in this direction. Is there any approach that aims at generating ERs for a code snippet? What is their limitation?

- Prompt in Figure 4 is not sound. Why statement? I guess it should be a code snippet.

- As for the semantic score, why not directly adopt existing metrics such as CodeBLEU?

- Extended from the above question, I am not convinced that using a metric is reasonable to assess the similarity between c and c'. I guess a code snippet can also have ERs in the form of code. As such, any metric could have brought bias since in this case, c and c' may have different syntactics. In light of this, why not use LLMs to judge code similarities? Why rely on a fixed metric?

- In the dual process, have you considered the inaccuracy brough by the LLMs, especially in the transforming of ERs into code. To what extent is this approach sound?

- Finding 5 is not well investigated. I wonder what features lead to different ERs? Does it mean code functionality or just syntactic features? Any evidence?

- Findings 7 & 8. IMHO, this is an overclaim to conclude them as findings. They are just single cases.

### Questions
Please feel free to answer all of my doubts.

### Soundness
2

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
3

### Summary
The authors propose a self-reflection approach utilizing iterative collaboration between two LLMs to generate equivalent representations (ERs) of code snippets. They investigate ER generation under both open and constrained settings. Similarity in text and syntax trees is used to measure program equivalence, leading to the identification and analysis of highly similar ERs, resulting in eight findings. The authors have open-sourced their experimental code and prompts.

### Strengths
1. The paper is well-written, allowing readers to clearly understand the methodology, experiments, and corresponding discoveries.

2. The analysis on the CoNaLa dataset provides eight findings that offer insights into how LLMs comprehend code.

3. The availability of open-source code facilitates reproducibility. The code is well-structured, providing clear reproduction steps, data, and prompts.

### Weaknesses
1. Novelty: The paper's innovation needs further clarification. The fields of NL2Code and Code2NL have been extensively researched (refer to [1, 2, 3]). What are the core differences between this work and existing studies?

2. Experimental Reliability: Evaluating program equivalence solely through text and syntax tree similarity might be unreliable. The constraint scores are assessed using LLMs; have the authors verified the accuracy and consistency of LLM scoring with human evaluations?

3. The analysis on the CoNaLa dataset yields eight findings, but their significance and guidance for future work are not discussed in the EMPIRICAL STUDY or FUTURE WORK section, suggesting the paper might be incomplete.

```
[1] Hu, Xing, et al. "Deep code comment generation." Proceedings of the 26th conference on program comprehension.2018.
[2] Li, Zheng, et al. "Setransformer: A transformer-based code semantic parser for code comment generation." IEEE Transactions on Reliability 72.1 (2022): 258-273.
[3] Zan, Daoguang, et al. "Large language models meet nl2code: A survey." arXiv preprint arXiv:2212.09420 (2022).
```

### Questions
- The paper only analyzes ERs for Python code. Does the analysis of different programming languages ​​show consistency? This may require further experimental evaluation.

- Can the authors elaborate on the impact of their findings on actual software development?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper conducts an empirical study to explore the ability of LLMs to generate Equivalent Representations (ER) of code. The authors employ the Reflexion framework to ask LLMs to generate ERs in both open and constrained settings and analyze the generated ERs. Overall, they find that LLMs have the ability to generate ERs for codes, which sheds light on the future exploration of this direction to support code-related tasks.

### Strengths
1. The paper is well-written and easy to follow.
2. Discussing ER in cod-related tasks is an interesting topic and remains largely underexplored.

### Weaknesses
1. In this paper, the authors stated that LLMs 'can' generate ERs for codes, which are generally cliches. Instead of studying whether LLMs are able to generate ERs, we are more interested in 'how well' LLMs can generate ERs. However, in this paper, there is no quantitative analysis of the correctness of the generated ERs. The semantic-equivalent score and constrained score can not reflect the correctness. The former only focus on semantic similarity, yet a similarity does not guarantee correctness. And the latter is generated by LLMs, which suffers from hallucination problems. The authors should manually check the correctness of generated ERs.
2. The findings lack in-depth analysis. In section 3, basically, there is only a case and a description of it for each finding.
3. The approach employed in this paper is nothing new compared with Relflexion, except for the designed scores for evaluating semantic similarities.
4. The future direction part lacks details and in-depth discussions. After reading this part, I am still confused about how ERs could actually get involved in the three discussed future directions.

### Questions
1. For the proposed semantic-similarity score, what is the difference between other famous code similarity evaluation metrics like CodeBLEU and it? Why is it necessary to propose a new metric?

### Soundness
2

### Presentation
3

### Contribution
1
