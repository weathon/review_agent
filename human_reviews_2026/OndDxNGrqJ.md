# Modeling Student Learning with 3.8 Million Program Traces

- Decision: Reject
- Scores: 6, 2, 2, 4

## Abstract
As programmers write code, they often edit and retry multiple times, creating rich “interaction traces” that reveal how they approach coding tasks and provide clues about their level of skill development. For novice programmers in particular, these traces reflect the diverse reasoning processes they employ to code, such as exploratory behavior to understand how a programming concept works, re-strategizing in response to bugs, and personalizing stylistic choices. In this work, we explore what can be learned from training language models on such reasoning traces: not just about code, but about coders, and particularly students learning to program. We introduce a dataset of over 3.8 million programming reasoning traces from users of PENCIL CODE, a free online educational platform used by students to learn simple programming concepts. Compared to models trained only on final programs or on synthetically-generated traces, we find that models trained on real traces are stronger at modeling diverse student behavior. Through both behavioral and probing analyses, we also find that many properties of code traces, such as goal backtracking or number of comments, can be predicted from learned representations of the students who write them. Building on this result, we show that we can help students recover from mistakes by steering code generation models to identify a sequence of edits that will results in more correct code while remaining close to the original student’s style. Together, our results suggest that many properties of code are properties of individual students and that training on edit traces can lead to models that are both more steerable and predictive of student behavior, even when evaluated solely on final program states.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a dataset of over 3.8 million programming reasoning traces from a free online educational platform. The authors develop and compare five model variants trained on this dataset: the trace model, last model, synthetic model, trace downsampled model, and synthetic downsampled model, which are evaluated from both behavioral and representational perspectives. They demonstrate that models trained on full traces acquire stronger representations of student coding behavior compared to models trained solely on synthetically generated traces or final program submissions.

### Strengths
1. This paper is well-motivated, and a decent amount of technical details are given.
2. The idea of modeling students' coding behavior through intermediate traces is both interesting and practical.

### Weaknesses
1. Insufficient dataset presentation
2. Missing discussion of related work and evaluation metric
3. Lacks user study
4. Limited evaluation to outdated model (GPT-2)
5. Code not provided

### Questions
1\. **Concerns about the dataset presentation**

A key contribution of this paper is the presented programming reasoning traces dataset. I suggest the authors add a dedicated section within the main text to thoroughly introduce the dataset's features and characteristics rather than placing this important information in the appendix. Additionally, providing an illustrative visualization of the dataset structure would help readers better grasp its organization and content.

2\. **Missing discussion of related work and evaluation metrics**

For the behavioral evaluation, the authors compare generated samples against actual student-written code. This objective semms similar to the work "Open-ended Knowledge Tracing for Computer Science Education" (EMNLP, 2022), which should be cited and discussed. Also, I suggest adopting CodeBLEU—a variant of the traditional BLEU metric specifically adapted for code—as suggested by this related work, as it would allow for a more accurate assessment of similarity between the predicted and actual student code.

3\. **User study**

The authors demonstrate that their trace model can help students recover from errors. I suggest that the authors conduct a user study in real educational settings to further validate this claim. Such an evaluation would provide valuable empirical evidence for the practical effectiveness of the proposed model.

4\. **Clarification on Figure 6 results**

In Figure 6, as the number of fine-tuning traces increases, the performance on trace generation appears to be lower compared to final program generation. Could the authors provide a more detailed analysis or explanation of this phenomenon?

5\. **Evaluation on more advanced language models**

The authors conduct experiments using base GPT-2 and OLMo-2 models. Given that GPT-2 is somewhat outdated, I suggest extending the evaluation to include more advanced models, such as those from the Llama series or other state-of-the-art LLMs, to further strengthen the generalizability of the findings.

6\. **Code and reproducibility**

The authors are encouraged to release the code to facilitate reproducibility and benefit the research community.

7\. **Typo**

Page 4, line 202: "a an" → "an"

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces a 3.8M programming trace dataset from Pencil Code and trains language models to capture student coding behavior, comparing models trained on real traces, synthetic traces, and final programs only. The focus on modeling "how" students code rather than just "what" they produce is interesting, but the scope is limited and the experimental section needs significant reorganization.

### Strengths
* The focus on "how" students code instead of just "what" they produce is a valuable perspective shift for modeling programming behavior.
* The 3.8M trace dataset from real students over 9 years is substantial and could benefit the education and code generation community.

### Weaknesses
* The base models (GPT-2 124M, OLMo-2 1B) are outdated. Modern models (such as qwen3, starcoder) would be more convincing baselines.
* Line 132 mentions "reported in Table 3" but I cannot find Table 3 anywhere in the paper.
* The entire work is based on one platform (Pencil Code) teaching "simple programming concepts" with visual blocks. This feels too narrow for ICLR. There is no evidence the findings generalize to other languages, platforms, or more complex programming tasks.
* The citation format does not follow ICLR style. Please check the formatting guidelines.
* Figure 3 has overlapping numbers that make it hard to read. Please fix the visualization.
* The experimental results section is very hard to follow. There are too many sub-research questions (5 major sections, each with multiple questions) but they are not well-justified. For example, Section 4.1 asks "Can models generate code that reflects real student programming behavior?" but I don't understand why this matters. The model is still just generating programs, so what is the point? The later experiments on probing and adapting to new students are more interesting, but they get buried.
* The paper tries to answer too many research questions at once. The authors should narrow down to 2-3 core questions and go deeper on those instead of spreading thin across many shallow analyses. More research questions does not equal a better paper.

### Questions
see weakness above

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work involves a set of model training experiments on a large programming traces dataset from Pencil Code. Specifically, they used five models to model students' behaviors, and also investigated the representations of both code and students.

### Strengths
+ It is a major contribution to show that, at a large scale, programming traces are useful for modeling students' programming.

### Weaknesses
- The contribution of this paper is unclear. One key issue is that while it comes from an educational discipline, it does not have any task involving actual educational goals. The five models are all about student behaviors or representations of students or code, but what is the next step? There is almost no educational implication discussed in the work.
- There are some key claims counterintuitive for general machine learning tasks. One of the biggest issues is about student embedding -- how exactly can we expect a model learned with student IDs to be generalizable for future new students? There are discussions about the result when new students are involved, and the result says "generalization is still difficult" -- this is almost certain, even for a large language model now trained with a lot of data. If you ask GPT-4 who a student is, it likely won't give you any good idea. The power of generalizability cannot help with tasks like overfitting to IDs.

### Questions
- Line 89: Why is this large dataset suitable for language model training? For small language models, smaller datasets could also work, especially if we want to create models for specific contexts.
- Line 94: The requirement and context of learning will be very important for the final program states. In classroom settings, students' final program states are almost all correct, while in situations of informal learning, there's often a lack of motivation for students to finish programming for many. This is actually not a minor issue -- context is very important for educational applications and this is missing.
- Line 99: We cannot train from IDs, but can check about certain classes or sessions.
- Line 118: So what is the goal? Education happens in a certain context, and it will need to show it surpasses small models in their own context to make sense. Otherwise we can always use smaller models trained in specific contexts.
- Line 130: While training LMs are important, it is still important to show what exactly will be a good downstream educational task.
- Line 140: I don't get this -- student IDs are involved in training and in this case, any new student will be an unknown input.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper presents an analysis of a dataset of 3.8 million code editing traces.
These traces are taken from PencilCode, which is a web-based code editing
platform focused on education. PencilCode allows the user to read and edit code
in both textual and graphical form, and seamlessly switch between the two.
However, this paper focuses on the textual representation.

The paper performs continuous pre-training or fine-tuning (you can argue which)
of GPT2-124M and Olmo-1B using the trace dataset. Each training item is a
sequence of code edits, along with certain metadata such as student ID. The
paper ablates the training data format: using synthetic traces (assuming each
step adds a line), using the ground-truth traces, and using just the final
program. The natural traces perform best on several days. The tasks considered
include getting the trained models to correct errors in student traces (i.e.,
completing a student trace to be correct), predicting the program title from the
trace, etc.

### Strengths
- This paper presents a dataset that is potentially very interesting. However, I
  believe there is no plan to release the dataset publicly.

### Weaknesses
The primary weakness of this paper is that it is missing several obvious
baselines that involve prompting pretrained models (e.g., any open-weight model
that is 32B+ or even a proprietary model). Since the traces involve program
execution in JavaScript and CoffeeScript, I imagine that a reasonable pretrained
model will pick-up enough in-context signals given the trace and a reasonable
prompt. I thought the most interesting task in the paper was on L428, where the
fine-tuned model completes a prefix of a student-written trace that ended in
failure with a successful trace 60% of the time. I expect that if you give a
broken program or trace to a reasonable pretrained model, it will identify and
fix the bug at least as well. I don't expect a pretrained model to be good at
probing student representations, but it's worth asking if they can do the other
code representation tasks. E.g., asking "will a student backtrack" is similar to
asking "is there a bug".

I also think this paper needs to do a better job engaging with related work.
There is enormous interest in studying how students learn to code, with
and without LLMs:

- BlueJ Blackbox (ICER 2018) has very detailed logs of edit actions. The
  ICER papper lists 18 papers that use the dataset.
- FalconCode: https://huggingface.co/datasets/koutch/falcon_code
- StudentEval (this is LLM related): https://huggingface.co/datasets/wellesley-easel/StudentEval

The datasets above are either open or relatively easy to get access to.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
