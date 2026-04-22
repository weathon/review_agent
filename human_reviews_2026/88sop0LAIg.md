# CACR: Reinforcing Temporal Answer Grounding in Videos via Candidate-Aware  Causal Reasoning

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
The growing need for direct answer retrieval from videos underscores the importance of Temporal Answer Grounding in Videos (TAGV)—the task of localizing the specific video segment that answers a natural language query. TAGV remains challenging, as it requires understanding semantically complex questions and handling the extreme length disparity between untrimmed videos and short answer segments. Current methods often underperform due to sensitivity to redundant content or limited visual reasoning ability. To overcome these issues, we introduce a Candidate-Aware Causal Reasoning (CACR) framework. Our approach first employs a Visual-Language Pre-trained (VLP) model to efficiently generate K candidate segments, then applies a temporal logic reasoning module strengthened by a rejection reward mechanism and optimized through Generalized Relative Policy Optimization (GRPO) for robust inference. Extensive experiments on four benchmarks show that our method achieves state-of-the-art performance in mean Intersection-over-Union (mIoU), offering a new direction for reasoning-based retrieval in long videos.We also publish our code at:https://github.com/anonymous1118-10/opencode-CACR

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the task of Temporal Answer Grounding in Videos (TAGV), which involves identifying the precise video segment that answers a natural language query. To handle challenges such as complex question semantics and the large temporal mismatch between lengthy videos and short answer segments, the authors propose a Candidate-Aware Causal Reasoning (CACR) framework. The approach first utilizes a vision-language pre-trained (VLP) model to efficiently generate K candidate segments, followed by a temporal logic reasoning module enhanced with a rejection reward mechanism to refine segment selection. The entire model is optimized using Group Relative Policy Optimization (GRPO). Experimental results demonstrate that CACR achieves state-of-the-art performance in terms of mean Intersection over Union (mIoU) across four benchmarks for reasoning-based retrieval in long videos.

### Strengths
1. The paper is clearly written and well-organized.
2. The paper introduces a novel CACR framework that integrates GRPO to refine top-k candidate segments through deep multi-modal reasoning, offering a principled solution to the challenges of temporal grounding.
3. The proposed method demonstrates consistent and notable performance improvements across multiple TAGV benchmarks, indicating its effectiveness.

### Weaknesses
1. The paper suffers from a substantial number of typographical and formatting errors that detract from its readability and professionalism. For instance, several sentences lack proper spacing (e.g., L105, L273, L443), and the acronym GRPO is inconsistently defined as Generalized Relative Policy Optimization instead of Group Relative Policy Optimization (e.g., L24, L102). In addition, the paper omits parentheses around citations and is missing several citations, which should be carefully corrected in revision.
2. The proposed two-stage pipeline, comprising candidate generation followed by multi-modal reasoning with a large vision-language model, may introduce significant computational overhead relative to existing TAGV baselines. A detailed complexity analysis or efficiency comparison would strengthen the empirical justification for the approach.
3. The rationale for adopting pre-answer instead of caption-based descriptions (L270–L280) is insufficiently explained. Clarifying this design choice, possibly with an in-depth analysis, would support their claims.

### Questions
1. Does the model append the full video clip at each reasoning turn during the iterative process? If so, could you provide details regarding the number of visual tokens involved and how this affects computational efficiency and memory usage?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces CACR, a framework for Temporal Answer Grounding in Videos (TAGV) that combines candidate segment generation with reinforcement learning-based reasoning. The authors address the challenging problem of locating short answer segments in long, untrimmed videos by proposing a two-stage approach: first, a VLP-based candidate selection (VBCS) module generates high-quality candidate segments, and second, a GRPO-optimized reasoning module performs fine-grained temporal localization using a composite reward function. The method is evaluated on four instructional video datasets and demonstrates state-of-the-art or competitive performance, particularly in handling extreme duration ratios and textual redundancy.

### Strengths
1)	The paper is built on a key observation: the maximum IoU between top-K candidates and ground truth increases with K. This motivates the use of a candidate-aware reasoning framework, which is both intuitive and empirically supported.

2)	The combination of VBCS for candidate generation and GRPO-based reasoning is well-justified. The use of a rejection reward mechanism and differentiated prompting strategy for the reference model are innovative and improve robustness.

3)	The method achieves SOTA or competitive results across four challenging datasets, with particularly impressive performance on TutorialVQA and CMIVQA, which involve extreme duration ratios and subtitle noise.

### Weaknesses
1)	While the paper compares with several VLP and LVLM methods, it would benefit from a more direct comparison with recent LVLM-based temporal grounding models (e.g., TimeScope: Towards Task-Oriented Temporal Grounding In Long Videos; Video-STR: Reinforcing MLLMs in Video Spatio-Temporal Reasoning with Relation Graph, etc) using the same candidate sets or reasoning mechanisms.

2)	The experiments are limited to instructional video datasets. It is unclear how well CACR generalizes to other domains (e.g., sports, movies, surveillance). 

3)	The paper mentions using LLMs for captioning and pre-answer generation but does not specify which LLMs were used or their impact on performance. More details would improve reproducibility.

4)	While GRPO is motivated by prior success in reasoning tasks, a deeper theoretical or intuitive explanation of why it is particularly suitable for TAGV would be helpful.

5)	The details are hard to understand in Figure 1.

### Questions
See the weakness.

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
4

### Summary
This paper proposes a candidate-aware causal reasoning framework (CACR) to improve temporal answer grounding in long videos. It first designs a temporal candidate generation method VBCS to obtain temporal candidates for each question, and then trains a MLLM via rejection rewarded GRPO algorithm to reject low-quality candidates. The experiments are conducted on 4 video answer grounding datasets and show remarkable improvements compared with previous methods and the baseline. The ablations on the candidate selection, caption, and pre-answers generation have demonstrated their effectiveness.

### Strengths
1.	The paper proposes a VBCS algorithm to generate candidate temporal segments to cope with extremely short temporal spans.
2.	It also designs a rejection reward which helps filter out poor candidate in GRPO optimization.
3.	The experiment results are good on 4 instructional video datasets.

### Weaknesses
1. The proposed VBCS algorithm is built on MutualSL. However, there is no introduction about this method, making it hard to understand the algorithm presented in L186~L203. For example, in line 9 of the algorithm, what is timeline mapping? How does it align the predictions from the visual and textual modules? What is the base prediction loss? What is the mutual transfer loss?
2. While the paper focuses on temporal answer grounding, the experiments are all conducted on instructional videos with rich subtitle information. To study if the approach can generalize to common videos, it is suggested to add experiments on other common video answer grounding datasets, such as NExT-GQA and ReXTime, and compare with related methods on these two benchmarks. 
3. I find that most of the compared methods are outdated from 2022 to 2023. More recent research [1] has shown the importance of text inputs (subtitles, captions) and the power of LLMs for answer localization in instructional videos. Thus, it would be interesting to see the comparison, and analyze the effects of different modalities in the proposed CACR framework.
[1] Xiao, J., Li, Q., Yang, Y., Qiu, L., Yao, A. (2026). Unleashing the Power of LLMs for Medical Video Answer Localization. In: Gee, J.C., et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2025. MICCAI 2025. Lecture Notes in Computer Science, vol 15966. Springer, Cham. https://doi.org/10.1007/978-3-032-04981-0_63
4. The paper emphasizes many times that the method enhances causal reasoning, but it is confused to me why and how it achieves causal reasoning, or what is the causal reasoning here.
5. Minor presentation issues: 
 -> L23 “Generalized Relative …” should be “Group Relative …”
 -> L52 & L152 Citation Errors.
 -> L177 MutualSL should have citation.

### Questions
No

### Soundness
3

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
4

### Summary
This paper proposes CACR, which aims to improve the state of the art performance on  temporal answer grounding in videos, which consists of identifying the most pertinent time segment to a question about a video. It does so by identifying candidate segments using a pre-trained VLM, and then using GRPO to fine-tune the VLM to maximize the IoU between the candidate segments and the ground truth. The results demonstrate that the proposed CACR works well for the TGAV task.

### Strengths
1. The paper is well written and easy to understand. Readers can quickly grasp the paper’s goal, its contributions and the proposed methodology.

2. The method clearly does have some (small) improvement over baselines on the TGAV specific benchmarks, suggesting that the authors' proposed method works reasonably well.

### Weaknesses
__1. The methods in the paper are lacking novelty__. The proposed VBCS is simply applying a pre-trained VLM to identify promising video segments - this has been done many times with prior VLM models to produce candidates, and is almost exactly the same as TFVTG [1]. GRPO itself isn’t new; the new part is the application to TAGV which feels somewhat incremental, since there are many other works where GRPO is applied to improve performance on downstream tasks [4]. Furthermore, GRPO has been applied to temporal grounding, such as in VTG-R1 [2] and Time-R1 [3]

__2. Experiments are not particularly impressive.__ While the paper focuses on standard TVG benchmarks to measure IoU, these don’t reflect challening long-video tasks (EgoSchema, Video-MME, etc) - it should be possible to see an improvement in downstream QA using the proposed method, and the effectiveness should be strengthened be evaluating on longer video tasks. Furthermore, the results themselves are not impressive and the ablations are somewhat inconclusive - improvements are rather small.

__Citations__

1.Zheng, Minghang, et al. "Training-free video temporal grounding using large-scale pre-trained models." European Conference on Computer Vision.

2. Chen et al, Datasets and Recipes for Video Temporal Grounding (2025)

3. Wang et al. Time-R1: post-Training Large Vision Language MOdels for Temporal Video Grounding.

4. Pinto, André Susano, et al. "Tuning computer vision models with task rewards." International Conference on Machine Learning. PMLR, 2023.

### Questions
The main concern I have is novelty. To consider accepting this paper, I need to clearly understand what the novelty of this paper is and why it should not be considered as an incremental work.

Smaller suggestions I have are that there should be some clear visual examples of CACR's result in the paper (ie, which frames are selected compared to other methods) and the formatting should also be fixed (lots of citations do not have inline parentheses).

### Soundness
3

### Presentation
2

### Contribution
2
