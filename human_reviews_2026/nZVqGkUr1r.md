# Library Hallucinations in LLMs: Risk Analysis Grounded in Developer Queries

- Decision: Reject
- Scores: 6, 2, 4

## Abstract
Large language models (LLMs) are increasingly used to generate code, yet they continue to hallucinate, often inventing non-existent libraries.
Such library hallucinations are not just benign errors: they can mislead developers, break builds, and expose systems to supply chain threats such as slopsquatting.
Despite increasing awareness of these risks, little is known about how real-world prompt variations affect hallucination rates.
Therefore, we present the first systematic study of how user-level prompt variations impact library hallucinations in LLM-generated code.
We evaluate six diverse LLMs across two hallucination types: library name hallucinations (invalid imports) and library member hallucinations (invalid calls from valid libraries).
We investigate how realistic user language extracted from developer forums and how user errors of varying degrees (one- or multi-character misspellings and completely fake names/members) affect LLM hallucination rates.
Our findings reveal systemic vulnerabilities: one-character misspellings trigger hallucinations in up to 26% of tasks, fake libraries are accepted in up to 99% of tasks, and time-related prompts lead to hallucinations in up to 84% of tasks.
Prompt engineering shows promise for mitigating hallucinations, but remains inconsistent and LLM-dependent.
Our results underscore the fragility of LLMs to natural prompt variation and highlight the urgent need for safeguards against library-related hallucinations and their potential exploitation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents an empirical analysis of the extent to which language models for code hallucinate concerning dependent software libraries.

The paper investigates six language models (GPT4, GPT5, Ministral, Qwen, Llama, and DeepSeek). The dataset used starts from BigCodeBench, yielding a dataset of 356 tasks using 30 distinct libraries. Three experiments are conducted:

1. To what extent do variations in user descriptions of libraries to be used influence hallucinations?
2. To what extent do misspellings in prompts lead to hallucinations?
3. To what extent do prompt engineering techniques (such as chain of thought, or step back) lead to hallucinations?

The experiments find that:

1. Adjectives (like fast, modern) to libary instructions generally have no influence; adding a year ("from 2025") does increase presence of hallucinations
2. Simple spelling mistakes leads to hallucinations in up to 26% of the tasks, with all LLMs resorting to fake library names.
3. Prompt engineering techniques like self-analysis and explicit checks can reduce hallucinations, but chanin of thought and step-back prompting often worsen hallucination rates.

The paper discusses how asking for lesser known libraries can further increase hallucinations, and whether hallucinations could be considered as sources of inspiration for desired functionality.

### Strengths
- A very well written paper
- Rigorous, well conducted experiment
- Clear and understandable results from the experiments
- A relevant problem

### Weaknesses
The findings are clear and the experiments are very well conducted. I was, however, somewhat disappointed by the discussion, which seems to accept the results at face value and just presents two somewhat unrelated ideas. I was hoping for a more rigorous treatment of underlying principles, consequences and implications, also to strengthen the sense of urgency from this paper. Why are we getting these results? What should be done about them?

For example, the finding that typos can have detrimental effects is a nice one -- but how could it be addressed? Could it be circumvented by having a tool check anything that might be a library, and spellcheck the prompt to use correct library names? Or: could LLMs be made more robust by adding 'adversarial' finetuning with likely typos?

The setup could also be viewed as a way to evaluate the effect of countermeasures. In practice, new library versions will continue to arrive after an LLM has been created, so RAG-based solutions or frequent fine tuning approaches are inevitable. But how well would they work? Might the setup proposed in this paper help?

### Questions
- What would you include in a bolder, more urgent discussion section?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents the first systematic study of how user-level prompt variations impact library hallucinations in LLM-generated code, evaluating six diverse LLMs (GPT-4o-mini, GPT-5-mini, Ministral-8B, Qwen2.5-Coder, Llama-3.3, DeepSeek-V3.1) across 356 Python coding tasks from BigCodeBench. The authors investigate two hallucination types, library name hallucinations and library member hallucinations, through three systematic experiments examining realistic user language from developer forums, user errors of varying severity, and prompt engineering mitigation strategies. Key findings reveal systemic vulnerabilities: while adjective-based descriptions like "fast" or "lightweight" are largely ignored, year-based prompts asking for libraries "from 2025" trigger hallucinations in up to 84% of tasks; one-character misspellings cause hallucinations in up to 26% of tasks, multi-character misspellings in up to 79%, and fake library names are accepted in up to 99% of tasks, demonstrating extreme model sycophancy. Prompt engineering shows mixed results, self-analysis and explicit-check strategies reduce hallucinations in ~80% of cases, while popular reasoning prompts like cot often increase hallucination rates.

### Strengths
1. The authors provide a precise definition of "library hallucinations" as two types of measurable failures (non-existent package names and invalid member calls from valid packages), and systematically measure them using three groups of realistic prompt factors (descriptive words/year-based prompts, user spelling errors, low-barrier prompt engineering). The narrative structure is clear and well-organized.

2. The experimental design is rigorous and comprehensive: it evaluates 6 diverse LLMs (covering open/closed source, general/code-specific, 8B-671B parameters) on 356 carefully filtered tasks from BigCodeBench, generating 3 responses per task using fresh API sessions to avoid caching effects.

3. The paper establishes a particularly insightful novel connection between LLM hallucinations and supply chain attacks (typosquatting/slopsquatting), revealing how models' sycophantic behavior, accepting fake library names in up to 99% of tasks, creates exploitable vulnerabilities.

4. The paper systematically compares four low-cost mitigation strategies (CoT, Step-back, Self-analysis, Explicit-check), providing immediately usable "lightweight prompt modification" baselines for practical engineering teams.

### Weaknesses
1. The Python-only scope with merely 30 libraries may not represent broader ecosystems. Results may not generalize to JavaScript, Java, or Rust with different packaging conventions, and the paper does not analyze whether findings vary by library popularity, domain, or API complexity, despite BigCodeBench containing 7 distinct domains.

2. The hallucination detection methodology is problematic: checking only against the latest documentation fails to distinguish true hallucinations from outdated knowledge or version mismatches, which is critical since models trained on older versions may correctly reference deprecated APIs. The manual review process lacks details on inter-rater reliability.

3. The most fundamental problem is the lack of methodological innovation, this is primarily a measurement study that does not propose any novel solutions. The paper only tests 4 existing prompt engineering techniques (chain-of-thought, step-back, etc.), which essentially reflects an absence of methodological contribution. For instance, the interesting finding in Experiment 3 that chain-of-thought prompting increases hallucination rates in 50% of cases is not leveraged to design better methods.

### Questions
See the section above.

### Soundness
2

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
2

### Summary
This paper conducts the comprehensive analysis of how realistic prompt variations affect hallucinations of library names and functions in LLM-generated Python code. The authors discover that while descriptive adjectives in prompts have little effect, subtle phrasing changes—such as year-based descriptions, or one-character misspelling-can significantly increase hallucination frequency. Although some prompt-engineering strategies can reduce these errors, their effectiveness is not consistent. The future work should focus more on this issue for building a more reliable coding model.

### Strengths
(s1) The studied problem is important and realistic.

(s2) The findings in this paper may provide some new insights to the community.

(s3) The empirical results support some claims in the paper.

### Weaknesses
(w1) The writing can be improved. For example, a figure of general description can be provided for the better understanding of readers.

(w2) I would like to see the evaluation results of more advanced coding LLMs, such as Claude. The selected models in the current version do not yet represent the best available coding capability..

(w3) The authors only conduct evaluations on Python language on one specific dataset (BigCodeBench). More results on other datasets would strengthen the paper. 

(w4) The prompting strategies show limited effectiveness in mitigating the hallucinations. I would suggest the authors try some training-based method. 

(w5) The Broader Impact section is missing, and I think this paper should include a discussion on some potential social impacts.

### Questions
See the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3
