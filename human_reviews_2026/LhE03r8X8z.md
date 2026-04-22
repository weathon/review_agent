# Plan then Act: Bi-level CAD Command Sequence Generation

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Computer-Aided Design (CAD), renowned for its flexibility and precision, serves as the foundation of digital design. Recently, some efforts adopt Large Language Models (LLMs) for generating parametric CAD command sequences from text instructions. However, our study reveals that LLMs pre-trained on large-scale general data are not proficient at directly outputting task-specific CAD sequences. Instead of relying on direct generation, we introduce a Plan then Act process where user instructions are first parsed into a chain-like operational plan via an LLM, which is then used to generate accurate command sequences. Specifically, we propose PTA, a new bi-level CAD command sequence generation method. The PTA consists of two critical stages: high-level plan generation and low-level command generation. During the high-level stage, an LLM-based Planner completes the planning process, parsing user instructions into a high-level operation plan. Following this, at the low-level generation stage, we introduce an Actioner equipped with a requirement-aware mechanism to extract design requirements (e.g., dimensions, geometric relationships) from user instructions.  This extracted information is used to guide the low-level command sequence generation, improving the alignment of the generated sequences with user requirements. Experimental results demonstrate that our PTA outperforms existing methods in both quantitative and qualitative evaluations.  Code is available at https://github.com/QiferG/Plan-then-Act.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a bi-level framework for text-to-CAD command sequence generation, which follows a “Plan then Act” paradigm: an LLM-based Planner first converts user instructions into a structured operation plan, and an Actioner then generates low-level CAD commands guided by extracted design requirements such as dimensions and geometric constraints. Experiments show that PTA achieves superior accuracy and better alignment with user intent compared to existing methods.

### Strengths
1. The paper is clear, well-structured, and easy to follow.
2. The proposed method is intuitive and presented concisely.
3. Both the quantitative and qualitative experiments are comprehensive, convincingly demonstrating the effectiveness of the proposed approach.

### Weaknesses
1. The method lacks novelty. In fact, the Nli_data dataset already contains more detailed plans, allowing for direct end-to-end generation of Nli_data-format outputs without the need to redesign the planning process.
2. The generation of the high-level plan relies on distilling Qwen2.5-32B-Instruct, whose performance has yet to be thoroughly compared with closed-source models.
3. While Qwen3-8B is a reasoning-focused LLM, the distillation process from Qwen2.5-32B-Instruct does not incorporate intermediate reasoning steps, which may limit its reasoning capabilities.

### Questions
Would it be possible to concatenate the user instruction with the high-level plan and use a single text encoder for the subsequent generation?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work provides a novel framework, Plan then Act (PTA), for text-to-CAD synthesis, where given a user instruction on the 3D CAD model to construct, the pipeline consists of a “planner” that first converts the instructions to a detailed step-by-step guide and an “actioner” which takes in both the original instruction and the plan to output the low-level CAD construction commands. The planner is implemented as a LLM that  is finetuned on high-level plans extracted from Text2CAD [1].  The actioner consists of BERT text encoders followed by a cross-attention based mechanism (named “Requirement-Aware Mechanism”) to fuse the representations of the user instruction and the high-level plans, from which the construction commands are decoded. Experiments demonstrate that on the Text2CAD test set, the proposed method achieves enhanced performance compared to LLMs that directly output CAD commands across 7 metrics.

### Strengths
1. The approach is novel and the motivation of a two-stage approach rather than directly generating the low-level construction commands is clear.
2. Experiment results show a significant improvement across different levels of detail in user prompts compared to open-source LLMs trained on the same dataset.
3. Ablation experiments are thorough in showing the contributions of each stage in the pipeline.

### Weaknesses
1. While the work compares to multiple open-source LLM baselines, there is only one closed-source model evaluated. It would be good to include reasoning models (e.g. o3 or o4), and stronger closed-source LLMs like Gemini-2.5-Pro and GPT-5.
2. It is somewhat unclear why the construction commands need to be decoded from a fused representation of the plans and the user instruction instead of just the high-level plans. This makes me wonder is there some information loss when generating the plans?

### Questions
1. What is the performance of more recent closed source LLMs on Text2CAD?
2. I am unsure about the necessity of the RAM compared to decoding from the high-level plans. Additional explanation and examples in the text could be beneficial.
3. What is the performance of training PTA end-to-end, without directly supervising the high-level plans?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a two-tier approach for natural-language based CAD modeling: an LLM-based Planner is first asked to create a high-level plan from user instructions, then a separate Actioner formulates a low-level command sequence executing the information given by the combined and strategically attenuated Plan + original user instruction. The combined features are driven by the Requirement-Aware mechanism introduced in this paper, which (1) identifies contextual relationships among the steps of the Plan, and (2) identifies complementary information between the high-level Plan and the original user-specified requirements. The authors provide extensive baseline comparisons and ablations, showing excellent relative performance of the proposed PTA approach.

### Strengths
The effective application of this approach to CAD is exciting, particularly with the commanding advantage that this method displays over other state of the art solutions, with respect to both quantitative and qualitative metrics. The overarching principles (multi-step reasoning/inference/generation, multi-technology approaches, reduction of user burden) are also valuable independent of the domain, and they are likely to generalize well beyond CAD. The paper and the method within are well motivated, and presented clearly. The ablation studies effectively demonstrate the contribution of each piece of the approach.

### Weaknesses
1. The main paper is currently lacking a suitable description for the dataset used to finetune the Planner. For example, how different are the nli_data files and the target high level plan that you seek from Qwen? How do you verify the quality/correctness of the Qwen output before using that as the basis for your Planner training? I realize some of this is already in appendix A2, but since it is an important piece of your approach, an additional few sentences in the main text would be very helpful.

2. I am somewhat unsatisfied/unconvinced by the need for the convoluted RAM and joint F_plan + F_inst representation. To be clear, the ablation studies and empirical evidence in the left side of Figure 2 have convinced me that the RAM and joint representation are necessary _given the current structure of the plan_. However, I'm curious what precludes the generation of a high-level plan that is complete and self-contained, such that the instructions are redundant? This is also tied to my previous question about the dataset for Planner training, as an improved Plan quality could dramatically simplify the process without sacrificing performance. Can the authors comment on whether a more complete Plan might be feasible, what the challenges are, and/or why the current approach is preferable/necessary?

3. There is minimal discussion about the limitations and assumptions of this system; the paper would benefit from such a discussion.

4. The current structure of Section 4 is somewhat disjointed and hard to read, thanks to the multiple levels of overviews.  For example, at l.191, the description is sufficiently low level that I found myself wondering about/looking for the details of L_plan, only to find that it would be deferred to l. 232. Similarly, at l.227, I wondered about the specific dataset/training procedure used to train the Qwen model (which are only detailed through Section 5.1, and Appendix). It may be worth another writing pass to smooth out this experience and connect the ideas more directly. Specifically, I would make the overviews slightly higher level, and add sufficient detail to the actual explanation as necessary.

### Questions
1. In lines 153-157, why are the Invalid Rates suddenly so much better? I'm particularly curious for Case 1, which seems identical to the experiment in the previous paragraph (where much higher invalid rates were reported)? Could you clarify what is different?


## Minor points 
The paper contains many typos and grammatical errors (some listed below), which are occasionally severe enough to obscure the meaning. Please take another careful proofread. 

- l.49-51 -- should that quote have a citation, or are the quotes just to indicate your assumption (in which case they could be removed)?
- l. 79, 81, 239, throughout -- missing articles (<the> operation steps in <the> high level plan... guides <a> Transformer decoder) or verbs (our goal <is> to efficiently utilize)
- l. 156-157 -- metrics like CD and F1 should be spelled out, given some additional context, or deferred to later when they're properly introduced.
- l. 148 -- how were LLaMA and Qwen finetuned for this experiment? Could you include details about the process/prompts/data/parameters to contextualize the experiment, either in the main text or in the appendix (with a forward reference)?
- l. 244-245 -- clarify the dimensions, e.g. N_t, N_p, d_p. They don't appear in Figure 2 or the text, as far as I saw.
- l.311 - Is there a reason that CAD-GPT would be a uniquely suitable dataset in your case? If not, it seems odd to mention it purely to say that it's unavailable; I would consider justifying or removing that line.
- l. 366, 367 -- the description of JSD is unclear to me. The description of COV has grammatical issues that preclude understanding.
- l. 466 -- efficiency --> efficacy
- l. 473 -- absence of <required> information
- l. 474 -- specially --> (specifically? especially?); onformation --> information

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
This paper proposes a method for generating CAD command sequences from natural language text. The authors first discuss how pre-trained large language models (LLMs) on general large-scale data struggle to directly generate task-specific, low-level CAD control commands. They then note that operation plans can be helpful in improving the accuracy of generating CAD command sequences. Based on these observations, they design a two-stage approach: first, using an LLM to generate a plan from the user text, and then employing a cross-attention network to combine these two text encodings to output the CAD command sequence.

### Strengths
1. The motivation for a two-stage approach—first generating a plan and then generating the command sequences—is interesting.
1. The discussion of directly generating command sequences and the high failure rates associated with this approach provides reasonable grounds for the further use of a two-stage process.
1. The experimental results appear convincing, showing that the method is useful.

### Weaknesses
1. Direct generation of command sequences with LLMs had higher failure rates, both with prompting and finetuning, although finetuning helped, as mentioned. Did the authors further try other post-training approaches, such as reinforcement learning?
1. It is not clear why cross-attention between the text and plan is useful. The plan should already be sufficient to generate reasonable command sequences, since it is derived from the user text and is more structured.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2
