# PR-CAD: Progressive Refinement for Unified Controllable and Faithful Text-to-CAD Generation with Large Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
The construction of CAD models has traditionally relied on labor-intensive manual operations and specialized expertise. Recent advances in large language models (LLMs) have inspired research into text-to-CAD generation. However, existing approaches typically treat generation and editing as disjoint tasks, limiting their practicality. We propose PR-CAD, a progressive refinement framework that unifies generation and editing for controllable and faithful text-to-CAD modeling. To support this, we curate a high-fidelity interaction dataset spanning the full CAD lifecycle, encompassing multiple CAD representations as well as both qualitative and quantitative descriptions. The dataset systematically defines the types of edit operations and generates highly human-like interaction data. Building on a CAD representation tailored for LLMs, we propose a reinforcement learning–enhanced reasoning framework that integrates intent understanding, parameter estimation, and precise edit localization into a single agent. This enables an “all-in-one” solution for both design creation and refinement. Extensive experiments demonstrate strong mutual reinforcement between generation and editing tasks, and across qualitative and quantitative modalities. On public benchmarks, PR-CAD achieves state-of-the-art controllability and faithfulness in both generation and refinement scenarios, while also proving user-friendly and significantly improving CAD modeling efficiency. The code and dataset are be available at { will be filled in upon acceptance }.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work focuses on unifying text-based CAD generation and text-based CAD editing.
First, it curates a high-fidelity interaction dataset with both qualitative and quantitative descriptions as well as multiple CAD representations.
Second, it introduces structured CoT and reinforcement learning to finetune LLMs on the curated dataset.

### Strengths
- This work focuses on a valuable and practical problem: is it possible to provide more friendly CAD generation experience with a single model supporting both editing and generation.
- The paper is easy to follow.
- The motivation is clear.

### Weaknesses
- For editing task (lines 211-240), how the textual instructions are generated? Do we double the inputted multi-view images and JSON descriptions shown in Fig 2 since we have two CAD models in the editing task?

- What is the number of data used for SFT and RL training stage? In lines 285-286, it is mentioned that a dataset of 1k triplets is fed into DeepSeek. How this data is used, e.g., as the pool of few-shot examples when prompting DeepSeek?

- There are only five qualitative results included in the main body and supplementary. It is difficult for me to know the capability of the proposed method with such a small set of qualitative results.

- For baselines that requires training (e.g., Text2CAD, Text-to-CadQuery, FLEXCAD and CAD-Editor in Table 1), are they retrained on the proposed dataset of this work? If yes, I think there should be values for IR and mean CAD metrics for them. If No, the current comparison seems to be unfair, because for the baselines, there are larger gap between training and test set (I guess the test set is from the proposed dataset of this work).

### Questions
see above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces PR-CAD, a unified framework for controllable text-to-CAD generation and editing. PR-CAD integrates both through a progressive refinement process. The authors construct a high-fidelity interaction dataset covering diverse CAD representations, edit types, and multimodal annotations. Experiments show that PR-CAD achieves SOTA controllability and faithfulness.

### Strengths
1. The paper is clear, well-structured, and easy to follow.
2. The proposed method is intuitive and presented in a concise manner.
3. The standard SFT and GRPO training processes appear reasonable.

### Weaknesses
1. The motivation for unifying generation and editing lacks novelty. In fact, the integration of these two tasks has been extensively explored. In my view, simply jointly training or continually fine-tuning LLMs on both generation and editing datasets can naturally achieve such unification.
2. The paper lacks an analysis of failure cases. It would be helpful to clarify under what circumstances the generated CAD models fail to align with the given instructions.
3. The proposed method also shows limited novelty. The four distinct rewards employed have already been commonly used in prior works, such as [1].
4. The effectiveness of R_length remains questionable, as generating complex CAD models inherently requires long sequences.  

[1] CAD-Coder: Text-to-CAD Generation with Chain-of-Thought and Geometric Reward. arXiv preprint arXiv:2505.19713 (2025).

### Questions
Please address the weaknesses above.

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
3

### Summary
This paper presents a progressive-refinement framework for CAD modeling, encompassing initial model generation and subsequent edits based on qualitative and quantitative instructions. To facilitate this, the paper first creates a dataset spanning all of the relevant query types; the resulting dataset spans more/more nuanced CAD operation types than previous work, while also phrasing the related descriptions in both quantitative and qualitative terms. The authors then introduce a model training procedure that leverages structured chain of thought, SFT, and RL in turn. The resulting model exhibits state of the art control for both CAD generation and refinement, as measured by geometric accuracy and the preservation of user intent. The authors also include user interaction studies to demonstrate efficacy and easy of use.

### Strengths
The idea of a unified approach that tackles both generation and editing of CAD models is novel and incredibly useful: it matches the desired/typical human workflow, and also provides great opportunities for meaningful multi-turn collaborative interaction. This is furthered by the inclusion of qualitative and quantitative intent descriptions, which allow for a wide range of control/detail levels without requiring cumbersome specifications. 

The paper is well motivated, and the high-level framework seems promising and well grounded. The approach also cleverly leverages a variety of different tools -- including ML algorithms and low-level representation decisions -- and deploys each one in the scenario it's best suited for, to collectively accomplish something that no piece could provide on its own. The paper is evaluated (and performs well) against several relevant benchmarks. I appreciate the presentation/discussion of both qualitative and quantitative metrics, together with the user interaction study. The paper's ablation studies effectively support the claims about mutual reinforcement between generation and editing tasks, while demonstrating the efficacy and necessity of each piece of the system.

### Weaknesses
1. Several important parts of the data generation procedure in Section 3.1 are unclear, and details required for reproducibility are missing. For example, the prompts used to go from JSON(+images) to the corresponding qualitative or quantitative descriptions are not discussed; it would be good to include a high level intuition about the prompt format in the main paper, and the specific prompts as part of the Appendix. Similarly, there is no discussion about the scope of the GPL (python) available for use by Gemini; was it restricted to particular imports/libraries or did it have free reign? Were the generated codes functional, consistent across examples, and otherwise sufficient to draw meaningful conclusions about Python's suitability for CAD serialization? I was also confused about the specific generation procedure used for the editing tasks; the process that you reference on l. 215 is only for generating static descriptions. Should that be a reference to Figure 3 (rather than Figure 2)? Does the "Qualitative Add Instruction" in Fig. 3 stem from a manually-formatted query string, or is that also inferred? How many edit variations can/should/do stem from a given model in your resulting dataset? Several choices would also benefit from more explicit motivation, like the decision to omit images from the quantitative description generation, or the choice to use 3 different models for the generation of qualitative descriptions (GPT4o), quantitative descriptions (Qwen) and GPL code (Gemini). 

2. This work does not seem to employ any validation steps to ensure the quality of the VLM/LLM-generated description/GPL data. Could the authors clarify whether/how the dataset was evaluated for quality/correctness/"human-like" patterns before use? Did the models ever include problematic content, such as descriptions that that would be meaningless in the context of a standalone prompt (e.g., referencing a "yellow ring", which is meaningful only in connection with the specific input rendering)? In connection with these concerns, I would also request that the claims regarding the dataset's "human-like interaction" be toned down or additionally justified. 


3. The model training steps are lacking critical details. There is no discussion of the specific prompts or even the high-level data query/response structure used for SFT or RL. The RL energy function also lacks some motivation/detail; for example, how did you arrive at the weights for alpha (decay) and beta (length penalty) as described in l.340? Does the length reward mean that complex examples that inherently require longer solutions always score poorly relative to the easier examples? Or is there a mechanism that allows extra length based on the difficulty of the problem before starting to penalize? Is this only about the code tokens, or does reasoning etc. count against this too if it's present?


4. Results from human user studies were presented with almost no context about the actual experiment(s). How was the user interaction study conducted? What was the task, what was provided vs. solicited from the participant? For example, what was occurring during the reported time measurement? How difficult were the tasks in question? Were the tasks identical using each method? If so, how did you account for familiarity (e.g., if you solve it with one method, the second approach will likely be faster since you already understand the problem deeply)? If not, how do you ensure a fair comparison between the two?
Regarding Table 2 & l. 429-436 - What are the "traditional end-to-end methods" in question? Do "end-to-end generation"/"Progressive Refinement" map directly to "Single-turn generation"/"Multi-turn generation", respectively? If so, please select 1 consistent set of terms (or make the mapping clear and justify why both are necessary). Where do we see evidence of the novices outperforming experts? Do you mean that (novices with PR / multi-turn) outperform (experts with end-to-end/single-turn)? If so, did you take any steps to ensure that the performance gap was actually due to PR involvement and not other factors of the experiment? For the ChatCAD commentary, who were the users? What is the difference between ChatCAD and the PR/multi turn system evaluated in Section 4.2? 
The answers provided should also reflect on whether the "user-friendly" claim is sufficiently supported; with the present evidence, this cannot be judged and should thus be toned down.

5. I'd prefer to see limitations in the main paper rather than being relegated to the end of the supplement.

### Questions
Questions
Please see weaknesses.


## Minor Comments 

- l. 29 - "dataset are be available" (typo)
- l. 68 - the parentheses around (modifying, adding, or removing) seem odd -- revisit this sentence
- l. 114 - [Monedero 2000] should be a parenthetical citation
- l. 129 - s-E never explained. Also, is this the same as SE (eg line 145) and S-E (line 212)?
- l. 151 - paragraph name duplicated (Inference with Large Language Models)
- l. 180 - I assume the JSON descriptions come from the DeepCAD dataset? Is this identically the DSL you mention for DeepCAD in l. 144? Please clarify in the text and/or Figure 2.
- l. 187 - shouldn't be a period after Zhang et al. (2024b). 
- Table 3 - what is the X? No output provided in that case? What could cause this? Please also clarify which elements were active in each experiment -- if not explicitly marked as"w/o", should we assume it's active? Eg does Qwen2.5-7B t/o/o Qualitative have SFT, RL, SCoT and both generation/editing? 
- l. 361 - were the evaluation models held out from your dataset, or were they obtained from elsewhere?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents PR-CAD, a unified, progressive refinement framework enabling both text-driven CAD model generation and iterative editing via large language models. The approach jointly supports qualitative and quantitative instructions throughout the full CAD lifecycle, leveraging a considerable new interaction dataset, structured chain-of-thought (SCoT) reasoning, and reinforcement learning post-training to maximize geometric fidelity, controllability, and user intent alignment.

### Strengths
The paper tackles the long-standing gap between text-to-CAD generation and editing by introducing a unified “progressive refinement” pipeline. This is conceptually coherent and practically valuable—most real-world CAD workflows are inherently iterative rather than single-shot.

All designs in the work, despite being complex, are well described and technically sound. The pipeline is not hard to follow.

The creation of a human-like CAD interaction dataset that systematically covers addition, deletion, and modification operations is a significant engineering contribution.

The ablation study confirms that each component—SFT, RL, and SCoT—contributes meaningfully to performance.

While simple, ChatCAD shows strong potential for automating the CAD design pipeline and benefiting the CAD community as a whole.

### Weaknesses
The work overlooks recent advancements in text-to-CAD generation, specifically CAD-Llama [1] and CADFusion [2]. This omission weakens the claim of achieving state-of-the-art results, particularly given the relevance of CADFusion's visual-feedback alignment. The authors are advised to discuss the relationship between their work and these developments, and/or to provide comparative analyses.

The qualitative results are **overwhelmingly** simple. While the text emphasizes “industrial-level” capability, no complex mechanical or multi-component assemblies are shown. While I like the remaining part of this paper, I strongly encourage the authors to Include more complex examples (for example, from the DeepCAD dataset) to better support generality. I will update my score accordingly if the problem is resolved.

[1] CAD-Llama: Leveraging Large Language Models for Computer-Aided Design Parametric 3D Model Generation. CVPR 2025.

[2] Text-to-CAD Generation Through Infusing Visual Feedback in Large Language Models. ICML 2025.

### Questions
How does PR-CAD compare empirically with CAD-LLaMA or CADFusion in either generation or editing?

Can the authors demonstrate performance on more complex industrial CAD samples, e.g., multi-body assemblies or DeepCAD’s large parts?

### Soundness
2

### Presentation
3

### Contribution
3
