# Reasoning Model is Stubborn: Diagnosing Instruction Overriding in Reasoning Models

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 4

## Abstract
Large language models have demonstrated remarkable proficiency in long and complex reasoning tasks. However, they frequently exhibit a problematic reliance on familiar reasoning patterns, a phenomenon we term *reasoning rigidity*. Despite explicit instructions from users, these models often override clearly stated conditions and default to habitual reasoning trajectories, leading to incorrect conclusions. This behavior presents significant challenges, particularly in domains such as mathematics and logic puzzle, where precise adherence to specified constraints is critical. To systematically investigate reasoning rigidity, a behavior largely unexplored in prior work, we introduce a expert-curated diagnostic set, ReasoningTrap. Our dataset includes specially modified variants of existing mathematical benchmarks, namely AIME and MATH500, as well as well-known puzzles deliberately redesigned to require deviation from familiar reasoning strategies. Using this dataset, we identify recurring contamination patterns that occur when models default to ingrained reasoning. We categorize rigidity patterns into three distinctive modes: (i) Interpretation Overload, (ii) Input Distrust, and (iii) Partial Instruction Attention, each causing models to ignore or distort provided instructions. We will publicly release our diagnostic set to facilitate future research on mitigating reasoning rigidity in language models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work discusses the phenomenon that LLMs rely heavily on familiar reasoning patterns, which may lead to wrong answers. To investigate the reasoning rigidity behavior, they introduce a diagnostic set, called ReasoningTrap. With this dataset, they identify the rigidity patterns and categorize them into three modes. They also proposed two simple mitigating strategies to improve the performances of reasoning modes on this dataset.

### Strengths
The phenomenon of reasoning rigidity is interesting and clearly explained. 
The proposed diagnostic set effectively show the disadvantages of the reasoning rigidity. 
The proposed mitigations solutions are effective.

### Weaknesses
1. The diagnostic set construction process is not rigorous. For ConditionedMath, the construction full rely on LLMs, which may not be reliable. Since there are only 84 questions, humans should be able to verify them one by one. As for PuzzleTrivial, the construction process is not clear. Are they constructed by LLMs or human annotators?
2. In Section 4.1, the authors proposed relative cosine similarity to measure the contamination ratio without justification, which make the analysis questionable. Does the relative cosine similarity mean contamination? In addition, what models do you use for such analysis?
3. LLMs are known to perform poorly on unfamiliar tasks. In this work, the authors proposed uncommon tasks that unfamiliar to LLMs. Can we see it as generalization problem instead of "Stubborn"?

### Questions
1. In Table 2, the total ration for the errors are 100%, does it mean that all the responses have errors? Is it possible that one response have many errors?
2. In Table 6, how do you measure the model entropy? What is the training data for this experiment?

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
The paper introduces *reasoning rigidity*, a failure mode where large reasoning models override explicit instructions and revert to familiar solution templates.  To diagnose this behavior, the authors present *ReasoningTrap*, a curated benchmark of modified AIME, MATH500, and puzzle problems designed to test whether models can follow unusual constraints rather than default reasoning paths. They analyze recurring contamination patterns, categorize three rigidity modes, and propose mitigation strategies via budget forcing and prompt hinting that yield moderate quantitative improvements.

### Strengths
1) The paper identifies and formalizes a meaningful yet underexplored problem in reasoning-model research—reasoning rigidity, where models override explicit user constraints and revert to habitual reasoning patterns.  

2) This focus appears novel and practically relevant, as it highlights a distinct failure mode beyond hallucination or faithfulness that directly impacts instruction reliability in reasoning-intensive domains such as mathematics and logic puzzles.  

3) The dataset and taxonomy provide a starting point for systematic studies on instruction-following robustness in reasoning models.

### Weaknesses
**W1. Ambiguous and Inconsistent Definition of “Reasoning Rigidity.”**  
The central concept—reasoning rigidity—was  defined inconsistently across sections, which possibly creates some conceptual confusion.  
In the introduction, it refers broadly to models editing or ignoring user-given conditions, implying any failure to follow instructions.  
Later, it is subdivided into three “rigidity patterns” (Interpretation Overload, Input Distrust, Partial Instruction Attention), which mostly describe misunderstanding or misinterpretation rather than genuine rigidity.  
Section 3 then narrows the definition again to cases where a model fully comprehends the conditions but still ignores them.  
To the reviewer, these inconsistent usages blur the distinction between misunderstanding and deliberate overriding, thus undermining a bit the conceptual precision.  
As this definition  anchors the entire paper’s motivation, dataset design, and analysis, its ambiguity may weaken both the theoretical foundation and the interpretability of empirical results.

**W2. Unclear Analytical Framing and Narrative Coherence.**
While the paper presented ReasoningTrap as a diagnostic dataset intended to analyze reasoning rigidity, the narrative jumped directly from defining the phenomenon to describing dataset construction, without clearly articulating how the dataset operationalized or tested specific hypotheses about when or why rigidity occurs.
This may create a perception that the dataset design was somewhat ad-hoc, even if it was meant as the analytical instrument itself.
Clarifying the analytical framework—e.g., what dimensions of rigidity the dataset aims to probe and how these relate to the three defined rigidity modes—would make the study’s logic and scientific contribution more coherent.

**W3. Limited Dataset Scale and Model Coverage**
The proposed ReasoningTrap dataset included only 164 problems (84 mathematical and 80 puzzle-based), which may limit the statistical robustness of the reported findings and the generalizability of the observed rigidity patterns.
Similarly, the evaluation covered only a small set of reasoning models and their base counterparts, leaving uncertainty as to whether the conclusions hold across broader architectures or training paradigms.

### Questions
**Q1. Clarify the Causal Rationale Behind “Familiarity → Rigidity.”**  
The paper assumes that making problems similar to well-known benchmarks (AIME, MATH500, classic puzzles) is essential to induce reasoning rigidity, but no empirical or theoretical support is provided for this causal link.  
Could the authors provide evidence that familiarity specifically amplifies this behavior?  

**Q2. Clarify the Scope of “Rigidity” vs. “Misinterpretation.”**  
Several examples (e.g., assuming extra cards in Figure 2, treating a negation as a typo in Figure 4(b)) seem to stem from input misunderstanding rather than explicit constraint overriding.  
Can the authors articulate how they distinguish between these failure types in annotation or evaluation?  
This clarification would be useful for interpreting the reported results and for potential future benchmarking.

**Q3. Lack of Qualitative Evidence for Hint Effectiveness.**  
While Table 4 reports aggregate improvements (e.g., pass@1 gains under different hint types), the paper provides no qualitative demonstrations of how the proposed hints concretely alter the model’s reasoning process.  
Could the authors include representative before-and-after reasoning traces to illustrate what specific changes each hint induces?  
Such examples would make the causal link between hint design and behavioral correction more convincing.

**Q4. On the Simplicity of Mitigation Strategies**
Section 4.3 proposes very lightweight mitigation methods (budget forcing and prompt hinting), both applied purely at inference time.
Could the authors elaborate on why no model-level or training-based interventions were attempted?
Do they view these simple strategies as diagnostic probes or as practical remedies? Clarifying this intent would help readers assess the contribution’s depth and potential impact.

**Q5. Which Model(s) Are Plotted in Figure 3?**  
The caption and text do not specify which models are aggregated or individually shown in Fig. 3.

**Q6. Similarity Direction in Section 4.1.**  
Why is cosine similarity computed with the *original* solution rather than the *modified* one the right comparator? Please justify this design choice.

**Q7. Sampling and Comparability Across Models.**  
Main results sample 16 responses per question with “default sampling parameters designated for each model.”  
Defaults differ across providers and can affect pass@1.  
Could you standardize temperature/top-p across models or report sensitivity analyses?

**Q8. Entropy Analysis (Table 6) Requires Methodological Detail.**  
What exactly is the “entropy” reported—next-token entropy over what prompt distribution, which model(s), and under what decoding?  
How is DAPO training involved (footnote), and how do you link entropy changes causally to RL-induced rigidity rather than dataset differences?

### Soundness
2

### Presentation
2

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
This paper identifies and systematically studies a new failure mode in reasoning models, which the authors term reasoning rigidity. The motivation stems from the observation that large reasoning models (LRMs), despite their strong reasoning abilities, tend to rigidly follow familiar solution patterns even when the problem statement explicitly requires a different reasoning path. To diagnose this issue, the authors construct a dataset called ReasoningTrap, which consists of modified versions of math and puzzle problems (ConditionedMath and PuzzleTrivial) where a small change in the condition invalidates the original solution path. Using a series of experiments across both base and reasoning-aligned models (e.g., DeepSeek, Qwen, GPT, Gemini), the paper shows that reasoning models are often more prone to such rigidity than their base counterparts. The authors further provide inference-time mitigation strategies (budget forcing and prompt hinting) and demonstrate partial alleviation of the problem.

### Strengths
1.Strong motivation and problem insight – The paper starts from an important and underexplored observation: reasoning models sometimes overthink or overfit to familiar reasoning patterns, which leads to rigidity in solving slightly perturbed problems. This provides a good bench for evaluating the overthinking issue of reasoning models.

2.Well-designed diagnostic dataset. The proposed ReasoningTrap dataset is thoughtfully constructed and fills a clear gap in the community. It offers a systematic way to test whether a reasoning model is capable of flexible reasoning rather than rote reproduction of learned trajectories.

3. Comprehensive experimentation. The experiments are extensive and cover a wide range of model families, including both base and reasoning-aligned versions. The evaluation is detailed, and the paper goes beyond diagnosis by also exploring mitigation strategies (token budget control and  prompts).

### Weaknesses
1. Conceptual boundary with overthinking. The notion of reasoning rigidity seems conceptually close to overthinking. It is unclear whether the proposed phenomenon is a specific manifestation of overthinking or a distinct failure mode. A clearer conceptual distinction or theoretical framing would strengthen the contribution.

2. Lack of difficulty-level analysis – The paper could analyze how reasoning rigidity varies with problem difficulty. For instance, are models more rigid on harder problems (where memorized or familiar templates dominate) and more flexible on easier ones? This analysis could clarify whether rigidity is partially driven by memorization rather than purely reasoning-level inflexibility.

3. The paper mainly focuses on inference-time mitigation. It would be valuable to explore training-level strategies, such as: (a) whether long-to-short reasoning or truncated CoT tuning reduces rigidity, an (b) whether fine-tuning on the ReasoningTrap dataset itself can mitigate this issue. Such experiments would make the analysis more complete and provide guidance for future model alignment.

### Questions
1. How does reasoning rigidity differ conceptually and mechanistically from overthinking or spurious reasoning alignment?

2. Does rigidity correlate with problem difficulty or math-specefic models?

3. Would fine-tuning with shorter reasoning traces or on the ReasoningTrap dataset itself alleviate the observed rigidity?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Authors in this paper, discussed an interesting failure mode of "thinking" models or LRMs, namely reasoning rigidity --- where the model ignores problem-specific variations and instructions, and default to known problem-solution mappings. To study this, the authors propose ReasoningTrap, a diagnostic dataset including modified mathematical and logical problems (from AIME, MATH500, and puzzle sets) designed to test instruction adherence. They identify three rigidity modes: Interpretation Overload, Input Distrust, and Partial Instruction Attention, and propose some mitigation strategies. Empirical analysis across several reasoning models (Qwen, DeepSeek, GPT, Gemini, Claude) shows that reasoning-aligned models often perform worse than base models on the modified tasks (especially on known problems and their variant pairs), indicating a bias toward familiar solution paths.

### Strengths
1. The notion of reasoning rigidity is interesting. A better interpretation could have been drawn from CBR. Nonetheless, this seems to be an useful failure mode to analyse
2. Authors introduce a benchmark to study the failure mode comprising of both logical and math problems. This seems useful. ReasoningTrap seems to be carefully structured to reveal subtle reasoning biases and contamination effects.
3. Experiments show some evidence to the existence of the rigidity problem.

### Weaknesses
1. The phenomena, lies in the middle of instruction adherence and reasoning failure. It seems like a small subproblem. I am unsure how important or insightful it is to be studied separately.
2. Dataset creation overtly relies on GPTs. 
3. Annotation procedures lack crucial details.
4. Certain findings (e.g., base models outperforming reasoning models) are neither deeply analyzed nor theoretically argued.
5. Some concerns with choices of the experimental setup (see below)

### Questions
L150: I am not sure how these two are the only two types. There are so many types of benchmarks, focusing on various logical languages, various levels of mathematics with increasing complexity. 
In fact, other domains such as finance domain etc may have such examples as well, that could have supported the current (somewhat narrow) problem scope.

L183: I am somewhat unclear as to the motivation of using GPTs to generate modifications as well. Doesn't that inherit the bias within these specific models. Using GPT as verifier blindly without any further justification does not make sense either. 
While the authors have done manual verification, I am not satisfied with the missing details about the annotation or verification process. This seem to have been trivialized.

L205: What I do not understand is the over-reliance of GPTs to generate datasets? Are we claiming that all other methods, manual methods, templated ones, or human-and-AI in the loop methods do not work well? There is not even a justification.

L322: While authors have adopted this experimental setup, I really do not know how relevant is this as a scientifc study. Observations such as this simply would entail that the base models have certain better abilities than their so-called "thinking" counterparts. This is nothing to be surprised about. With continual learning, forgetting is a well-observed phenomena.

Sec 5.2: What does this section 5.2 mean? By definition, the original problem should be "common"/"known" and the variant should be unknown. Otherwise, the variant problem's solution could be mapped as well. I do not understand what "unseen" is supposed to mean.

L405: abuctive -> deductive.

### Soundness
3

### Presentation
2

### Contribution
3
