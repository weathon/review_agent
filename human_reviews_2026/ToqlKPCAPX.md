# Learning To Make MISTAKEs: Modeling Incorrect Student Thinking And Key Errors

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 8, 2

## Abstract
Research on reasoning in language models (LMs) predominantly focuses on improving the correctness of their outputs. But some important applications require modeling reasoning patterns that are incorrect. For example, automated systems that can reason about and simulate student errors are useful for providing real-time feedback in the classroom or offline practice for educators in training. This paper presents a new method, MISTAKE, that (1) constructs high-quality synthetic examples of reasoning errors by leveraging cycle consistency between incorrect answers and latent misconceptions; and (2) uses the generated data to learn models for student simulation, misconception classification, and answer generation. We evaluate MISTAKE on three educational tasks and find that it results in (1) higher accuracy when simulating incorrect student answers based on specific misconceptions, (2) increased performance inferring latent misconceptions from observed incorrect answers, and (3) higher alignment with expert-written distractor answers when generating incorrect answers.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper tackles the problem of student (in a tutoring sense) simulation using LLMs, which is made hard by the fact that LLMs are mostly trained for solving success which is in contrast to simulating incorrect reasoning traces.
They propose a loop which samples incorrect answers from an LLM, infers the misconception, and then simulates a student answer.
On one dataset with three tasks they show that the training loop improves student simulation, misconception inference, and distractor generation.

### Strengths
- The paper tackles a well-motivated and important problem. Many ongoing research activities in LLM-based tutoring (and adjacent fields) rely on good student simulation, so progress is expected to be impactful.

- The data used in the study consists of data from real students which distinguishes it from many contemporary studies.

- The method itself is intuitive and the paper is well-written, there also are improvements with the method.

### Weaknesses
- Overall, I do not find the results very convincing yet. This is partially due to the fact that only one dataset was used (which nevertheless is also due to the lack of good datasets in the research field) but also because I find the exploration of the method itself not very extensive. For example, in Fig. 3 training is stopped after 4 rounds but the results are clearly still improving. There is also no real comparison to any baseline (but proprietary LLMs) so it is not clear if there is not a better (simpler?) method. Furthermore, the runtime is not mentioned anywhere but I suspect that the method is quite slow. For example, for misconception inference, how does it compare to a simple classifier trained using SFT on the data? Is this perhaps "misconception only" in your figure? In that case I am quite surprised that performance does not increase at all.

- The discussion of related works on reasoning could be expanded. For example, RAFT is an RL method that does not need target labels, as well, but is not discussed

### Questions
- One citation in L88 seems broken

- Could you please provide the runtime of your method?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces MISTAKE, an unsupervised framework designed to model students' incorrect reasoning through cycle consistency between misconceptions and wrong answers. The method generates synthetic pairs of misconceptions and errors using large language models and jointly trains two modules: a student simulation model (misconception to wrong answer) and a misconception inference model (wrong answer to misconception). Experiments are conducted on the EEDI Mining Misconceptions in Mathematics dataset, covering three educational tasks: student simulation, misconception inference, and distractor generation. Results show that MISTAKE with cycle-consistency filtering improves performance on student simulation and misconception inference. It also produces distractor answers that align more closely with expert-written options.

### Strengths
- The paper investigates an important problem, modeling students' incorrect reasoning using language models.

- The framework jointly trains two complementary models (student simulation and misconception inference) in an iterative loop.

- The paper evaluates the approach on three well-defined educational tasks, offering a diverse test of model capability.

### Weaknesses
- The motivation is not clearly articulated. It remains unclear what real-world problems these tasks aim to solve. The paper would benefit from grounding its contribution more directly in practical educational applications.

- The tasks closely overlap with existing paradigms: Student Simulation resembles knowledge tracing, Misconception Inference parallels automated grading or error diagnosis, and Distractor Generation has extensive prior work. I think it would be better to include comparisons with established baselines for these tasks or apply their framework to these existing task formulations to better demonstrate its advantages and generalizability in realistic educational settings.

- The paper does not explore scenarios where cycle consistency may fail or lead to inconsistent reasoning, nor analyze the robustness of the method under such conditions.

- The experiments rely on relatively outdated models (e.g., GPT-3.5, GPT-4o) and are restricted to OpenAI's ecosystem, limiting the generality of the findings.

- Cycle consistency does not necessarily equate to human reasoning. Even if the model successfully reproduces a (misconception -> wrong answer-> misconception) loop, it may reflect internal LLM logic rather than genuine human cognitive processes. The observed performance gains might stem from pretrained exposure to part of human errors rather than true reasoning simulation.

- Prior work (e.g., Aher et al., 2023; Markel et al., 2023) has shown that LLM-simulated student responses often fail to reflect authentic reasoning. If this approach is central to the paper, the authors should provide stronger justification for its validity and discuss its theoretical underpinnings.


Aher, Gati V., Rosa I. Arriaga, and Adam Tauman Kalai. "Using large language models to simulate multiple humans and replicate human subject studies." International conference on machine learning. PMLR, 2023.

Markel, Julia M., et al. "Gpteach: Interactive ta training with gpt-based students."

### Questions
- Could the authors include experimental comparisons with established baselines for these tasks, such as knowledge tracing for Student Simulation, automated grading or error diagnosis for Misconception Inference, and prior distractor generation methods, to better demonstrate the advantages and generalizability of their framework in realistic educational settings?

- You primarily evaluate on Llama ~3B. Would MISTAKE apply equally well to other 3B-class families (e.g., Mistral, Qwen, Phi) or even smaller sub-3B models? Could you report results to show whether the gains persist across architectures and parameter scales?

- How do performance gains scale as model size decreases (e.g., 7B -> 3B -> 1–2B)?

- Are any components of MISTAKE sensitive to the base model family? If so, which parts degrade first on smaller or non-Llama backbones?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a new method to simulate the reasoning errors of students doing simple math problems, addressing the issue that when LLMs produce incorrect answers to simulate students, their reasoning is often incorrect. The work introduces MISTAKE, using a cycle-based method to let models generate incorrect answers and the corresponding reasoning to amplify samples when generation reflects the error. Their results show that based upon a relatively small model, they perform similarly to closed commercial models such as GPT 3.5 turbo on three educational tasks.

### Strengths
+ Their narrative and motivations are very clearly introduced. The paper is smooth to read.
+ The design of the system, while inspired by previous work, is novel in its application to the specific problem they are trying to address.
+ The work has also introduced the experimental details and results clearly. 
+ The contribution to the field of LLM for math education is clear. The method is almost plug-and-play, and if the model can be tuned (or even fine-tuned?), it can be applied directly.

### Weaknesses
- The results are not as exciting as they were introduced in the first section. Results necessarily show that while they improve from the base model, they could not beat the commercial models, raising a question mark on the contribution of the paper. It is strongly recommended to try some newer models or develop a version using fine-tuning and apply it to commercial models to see if it surpasses these models.

### Questions
Here are some comments about the paper, not necessarily questions.
- Figure 1: More likely, these are not common misconceptions, just two examples. To be fair, the addition and subtraction is more likely just a slip instead of misconceptions.
- Line 69: These are quite significant improvements.
- Line 105: This url is not working.
- Line 200: I like how you use examples here to show the educational context and how each step looks like. Great communications.
- Line 225: Will this actually make the available data points fewer? I guess the results show it's working but still wanted to discuss about the possibility here.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes MISTAKE, LLM-based framework for generating and learning from realistic student errors. The model first produces an incorrect answer and rationale, then infers the underlying misconception, and finally re-generates the student response from that misconception; only cycles that return to the same error are kept as high-quality data. Using these filtered examples, the authors fine-tune a student-simulation model and a misconception-inference model, and on Eedi math data they report gains on all three targets: simulating student answers, identifying misconceptions, and generating plausible distractors.

### Strengths
* Interesting twist on STaR. Instead of using STaR to find better (correct) chains, the paper shows you can run a similar self-improvement loop to learn stable wrong chains. That’s a nice shift of the usual “reasoning” story.
* Cycle-consistent filtering is the core technical idea. The paper doesn’t just sample bad answers — it checks that “misconception → student answer → back to the same misconception” holds, and only keeps those. That’s a simple, model-based quality control that doesn’t need extra labels.
* Forward + inverse models makes it more general. Training both “given a misconception, generate a student answer” and “given the answer, recover the misconception” is a pattern other people can reuse outside education (anywhere you model human errors or behaviors).
* Joint learning of two roles teachers actually play. Training both a student simulator (produce an error given a misconception) and a misconception inference model (infer misconception from the observed error) mirrors what human educators do when they move back and forth between “I know the bug, what would a student say?” and “I saw the student’s work, what’s the bug?”. That symmetry strengthens the claim that this is an AI in Education contribution, not just a data-augmentation trick.

### Weaknesses
Major Weakness:
* Misleading characterization of learning paradigm: The paper claims this is an "unsupervised" approach, but the use of cycle consistency checks and correct answer signals makes this at least weakly supervised. The terminology should be corrected to accurately reflect the supervision signals used.
* Experiment result MAP@25 scores: MAP@25 seems excessive when models achieve very low scores across the board. This raises concerns about either (1) the task difficulty being unrealistic, (2) data quality issues, or (3) the metric being inappropriate. The paper should analyze whether the low performance stems from inherent task difficulty or methodological limitations, and consider whether a smaller k value would be more informative. In addition, I check the kaggle leaderboard and the scores (the metrics is a MAP@25 derivative) are usually around 60, could you please explain the gap (the SoTA’s actual MAP@25 scores)?  How’s the performance of MISTAKE + SoTA models, since MISTAKE is a model-agonistic framework? 
* Overall performance and the effectiveness of the MISTAKE framework. While the MISTAKE framework shows incremental improvements over multiple training rounds, the final performance of Llama3.1-8B still substantially lags behind GPT-3.5-turbo across all three tasks. This raises serious concerns about the practical value of the approach: the computational cost of multiple rounds of data generation, filtering, and fine-tuning yields a model that still underperforms a simpler baseline of prompting an older, widely-available model. The paper needs to either (1) demonstrate that MISTAKE can bring smaller models to competitive parity with stronger baselines, (2) show that applying MISTAKE to already-strong models yields further gains, or (3) provide compelling arguments for why the modest improvements justify the added complexity.
* Limited model diversity: Evaluation uses only GPT-series models and LLama3.1-8B. The claims would be strengthened by including other state-of-the-art models such as Gemini, Claude, Qwen3+, or GLM4+ to demonstrate generalizability across model families.
* Baselines are too weak / incomplete. To claim this is a meaningful alternative to STaR-style self-training, it needs comparisons against (i) a strong, modern LLM used directly as a student simulator (GPT-4/4o or at least a stronger open model) and (ii) a simpler filtering rule (e.g., answer-matching only). Without that, it’s hard to see how much the proposed loop actually buys.
* Limited novelty and missing educational insights: Methodologically, this is an incremental extension of STaR. To justify publication, the paper needs deeper educational analysis showing what new insights the approach provides about student learning, misconception patterns, or pedagogical implications that go beyond the technical contribution, which is missing in the result section. The original STaR  paper conducted human evaulation and fonud that some of the LLM-generated rationale doesn’t acutally make sense, which prohibits real-world educational usage. 
* No real human or teacher evaluation. The whole pitch is “we generate more realistic mistakes,” but there’s no expert study or even crowd study checking “would you accept this as a plausible student error?” Automatic metrics on the same dataset used for training are not enough to back an educational realism claim.

Minor weakness:

* Abstract and introduction-contribution lack clarity and concrete comparisons: The abstract uses vague comparative language ("higher", "increased") without specifying baselines. Readers cannot assess the claimed improvements without knowing what the method is compared against.
* Algorithm 1 readability issues: Critical variables (m, s, r, w, etc.) lack explicit definitions within the algorithm. Readers must repeatedly cross-reference Section 3.1 to understand the pseudocode. For instance, 's' appears to represent simulated student answers but is never formally defined. All variables should be clearly defined in the algorithm caption or within the pseudocode itself.
* Error space is still hand-anchored. The method assumes you can name / infer a relatively small set of misconceptions and route through them. That’s fine for Eedi, but in real classrooms misconceptions are messy, overlapping, and partially linguistic. The paper doesn’t show the approach discovering new or finer-grained errors.
* Domain is too narrow. Everything is on Eedi-style, multiple-choice, school-level math. There’s no evidence the method survives when answers are open-ended, multi-step, or not tied to a pre-enumerated misconception set. Right now it’s “works on Eedi,” not “works in general.”
* Synthetic loop may be self-confirming. The cycle-consistency check rewards the model for repeating its own mistake pattern, not for matching real student diversity. That risks learning “LM-like wrongness” rather than “human wrongness,” but the paper treats cycle pass-rate as if it were an external quality signal.
* Limited analysis of synthetic distractor quality: The LLM-generated distractors may introduce biases or patterns that don't reflect real student errors. Real misconceptions follow long-tail distributions with rare but surprising error types. The paper should analyze the coverage and distribution of generated distractors compared to actual student responses, and discuss how the multiple-choice format may constrain the misconception space.
* Unaddressed failure modes of cycle consistency: The paper doesn't examine cases where cycle consistency checks might validate incorrect misconception-answer alignments. What percentage of cycle-consistent examples are actually wrong? This is important for understanding the quality control mechanism's limitations.

### Questions
For LoRA fine-tuning, does the method use completion-only loss (only computing loss on generated tokens) or full next-token prediction loss? This significantly affects training dynamics and should be specified.

### Soundness
2

### Presentation
3

### Contribution
2
