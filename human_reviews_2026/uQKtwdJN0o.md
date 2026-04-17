# FrugalRAG: Less is More in RL Finetuning for Multi-hop Question Answering

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Reinforcement learning (RL) based on the final answer's reward has driven recent progress in small language models (SLMs) on reasoning-heavy tasks such as math and code. However, applying the same techniques to retrieval-augmented generation (RAG) benchmarks like multi-hop QA has yielded limited gains—often trailing supervised or prompting-only baselines. Instead, we argue that a viable path for RL in multi-hop QA is to use test-time scaling judiciously, for optimizing both the final answer accuracy and the efficiency in reaching that answer. 
We propose FrugalRAG, a two-stage finetuning framework that adaptively _reduces_ the number of retrieval steps based on a question's difficulty. First, we train an SLM with supervised finetuning on a full-exploration policy that generates broad sub-queries. Then, we apply RL to adaptively prune search depth based on question difficulty, directly rewarding policies that balance correctness with frugality. Unlike prior approaches requiring 10× more data, our method achieves competitive performance with only ~1,000 examples. On HotPotQA and other multi-hop QA benchmarks, FrugalRAG attains state-of-the-art efficiency–accuracy tradeoffs, cutting retrieval cost nearly in half.  Moreover, on the challenging BrowseCompPlus benchmark, it generalizes zero-shot and surpasses SLM-based and other baselines. These results demonstrate the use of RL—not to increase reasoning steps but to reduce them—as an effective solution for scalable, efficient RAG.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces FrugalRAG, a two-stage finetuning framework that combines supervised exploration and reinforcement learning to optimize retrieval efficiency in multi-hop question answering. Instead of simply scaling test-time retrieval, the approach adaptively determines the number of retrieval steps per query, balancing accuracy with efficiency. Experiments across benchmarks such as HotPotQA, 2Wiki, MuSiQue, and BrowseCompPlus show that FrugalRAG achieves strong efficiency–accuracy tradeoffs while training on only 1,000 examples.

### Strengths
1. The focus on frugality addresses a critical yet underexplored dimension in RAG systems (efficiency, not just accuracy).

2. The proposed method achieves competitive or superior results with only 1,000 training examples is impressive compared to prior work requiring 100k+.

### Weaknesses
1. The comparison is unfair by natural, as baseline uses weak retriever (not colbert v2). 

2. The RL reward design assumes access to the “optimal” stopping point based on ground-truth evidence, which is often unavailable or impractical in real-world applications.

3. While the paper emphasizes data efficiency and search efficiency as central contributions, these concepts have been discussed extensively in prior work (e.g., [1,2,3, 4]). The paper would benefit from a clearer articulation of how its framing or results go beyond existing definitions of efficiency.

4. Limited case studies and inference efficiency studies of the proposed method. 


[1] Wang, Hongru, Cheng Qian, Wanjun Zhong, Xiusi Chen, Jiahao Qiu, Shijue Huang, Bowen Jin, Mengdi Wang, Kam-Fai Wong, and Heng Ji. "Acting Less is Reasoning More! Teaching Model to Act Efficiently." arXiv preprint arXiv:2504.14870 (2025).

[2] Jiang, Pengcheng, Xueqiang Xu, Jiacheng Lin, Jinfeng Xiao, Zifeng Wang, Jimeng Sun, and Jiawei Han. "s3: You Don't Need That Much Data to Train a Search Agent via RL." arXiv preprint arXiv:2505.14146 (2025).

[3] Li, Zhuofeng, Haoxiang Zhang, Seungju Han, Sheng Liu, Jianwen Xie, Yu Zhang, Yejin Choi, James Zou, and Pan Lu. "In-the-Flow Agentic System Optimization for Effective Planning and Tool Use." arXiv preprint arXiv:2510.05592 (2025).

[4] Song, Huatong, Jinhao Jiang, Wenqing Tian, Zhipeng Chen, Yuhuan Wu, Jiahao Zhao, Yingqian Min, Wayne Xin Zhao, Lei Fang, and Ji-Rong Wen. "R1-Searcher++: Incentivizing the Dynamic Knowledge Acquisition of LLMs via Reinforcement Learning." arXiv preprint arXiv:2505.17005 (2025).

### Questions
Suggestion: The exposition is sometimes dense, with technical details (e.g., rollout generation, reward shaping) overshadowing high-level intuition. A clearer articulation of practical implications and limitations would improve readability.


Could you provide more detailed analysis or case studies showing how the system behaves on questions of varying complexity?

Since the method uses only 1,000 labeled examples, how sensitive is performance to the exact number of examples? Have you tested smaller budgets (e.g., 100 or 500 examples) to see the degradation curve? I am wondering what is the performance of the baseline with the same amount of number of examples.

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
5

### Summary
The paper proposes FrugalRAG, a two-stage finetuning framework for multi-hop RAG: Stage-1 supervised “explore” training to maximize evidence coverage, followed by Stage-2 RL to adaptively stop retrieval, balancing answer quality against search cost. With ~1k training examples, the method reports strong efficiency–accuracy tradeoffs and adaptive compute (e.g., fewer searches on easier questions) across HotPotQA, 2Wiki, MuSiQue, and some generalization to BrowseCompPlus.

### Strengths
1. Separating exploration (coverage) from stopping (efficiency) is a clean, actionable design; the reward explicitly ties termination to an “optimal” hop length and format compliance.

2. Tables show fewer searches at comparable or better MBE/recall versus strong prompting and SFT baselines, and the “tradeoff” metric highlights practical benefits.

3. The method increases search steps with question difficulty and shows some cross-dataset transfer; the modularity across retrievers/generators is also demonstrated.

### Weaknesses
1. Motivation around “less data” is under-argued: The paper asserts success with ~1k examples, but the core motivation, why low-label regimes are critical for RAG, feels underspecified. In many RAG settings, labeled QA pairs or weak labels are not rare (e.g., synthetic or distantly-supervised pipelines), and the paper does not clarify whether performance keeps improving with more data (scaling behavior is central to RAG systems). A scaling curve (1k→10k→100k) would make this claim concrete.

2. Limited novelty. Algorithmically, Stage-2 introduces a termination-oriented reward shaping that discourages extra retrieval steps; this reads as a targeted constraint atop existing RL-for-RAG methods rather than a fundamentally new learning principle. The contribution is useful but incremental.

3. Missing ablation on the retrieval-steps component of the reward. Since the central claim is “RL learns to stop,” an ablation that removes the step-penalty/termination reward is necessary to show the gain is not just from better exploration traces or general RL regularization. (Table/figures do not isolate this factor.)

4. Questionable cross-method comparisons due to retriever mismatch. Several headline comparisons juxtapose methods that use different retrievers (e.g., E5-base-v2 vs. ColBERTv2) or different indices/evaluators. This undermines causal conclusions about the policy’s quality. A strict apples-to-apples study (same retriever, index, judge) is needed for Search-R1/CoRAG comparisons.

5. Related work coverage is incomplete. Important recent efforts are missing or not discussed in sufficient depth, such as [1-3].

[1] Wang X, Wang Z, Gao X, et al. Searching for best practices in retrieval-augmented generation[J]. arXiv preprint arXiv:2407.01219, 2024. 

[2] Ammann P J L, Golde J, Akbik A. Question Decomposition for Retrieval-Augmented Generation[J]. arXiv preprint arXiv:2507.00355, 2025. 

[3] Xu R, Shi W, Zhuang Y, et al. Collab-rag: Boosting retrieval-augmented generation for complex question answering via white-box and black-box llm collaboration[J]. arXiv preprint arXiv:2504.04915, 2025.

### Questions
refer to weakness

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
4

### Summary
The paper proposes FrugalRAG, a two-stage finetuning method that makes language models better at multi-hop question answering with retrieval-augmented generation while staying efficient. First, it uses supervised finetuning on about 1,000 examples to teach the model how to write good search queries and gather relevant evidence across multiple hops. Then it uses reinforcement learning not to make the model search more, but to teach it when to stop searching once it has enough information. This lets the model adapt its compute: easy questions get only a couple searches, hard questions get more. FrugalRAG matches or beats prior systems on benchmarks like HotPotQA in both answer accuracy and evidence recall, while using fewer retrieval calls and far less training data than existing RL approaches.

### Strengths
S1. Clear problem focus. The paper presents a well-defined motivation by targeting the inefficiency problem in multi-hop RAG systems, where existing RL/RAG agents often over-retrieve and waste searches. FrugalRAG directly addresses this issue by introducing an adaptive stopping mechanism that halts retrieval once sufficient evidence is gathered, ensuring both cost efficiency and effectiveness.

S2. Logical two-stage design. The technical structure is intuitive and easy to follow.
Stage 1 uses supervised fine-tuning to learn effective retrieval, and Stage 2 applies RL to decide when to stop searching—cleanly separating exploration and stopping behaviors.

S3. Strong efficiency and quality gains. The method achieves notable efficiency improvements while maintaining or improving accuracy.

### Weaknesses
W1. Concerns about technical design

(1) Line 251 defines “$\Delta$ is the normalized difference $(h_\text{term} - h) / B$”, but only $h_\text{term}$, $h*$, and $B$ are defined before. Should it be “$\Delta$ = $(h_\text{term} - h) / B$”? This matches the idea that the penalty depends on how far you are from the optimal stopping hop.

(2) Lines 243–249 define a piecewise reward $R$ for late stop, perfect stop, and early stop. The “early stop” case is triggered by $c < \tau$ (low answer quality), and it reuses the same penalty term $log((1-\Delta)/\Delta)$. But if you stop early then $h_\text{term} < h*$, so $\Delta < 0$, and $log((1-\Delta)/\Delta)$ is undefined for negative $\Delta$ (the fraction becomes negative). As written, the early-stop reward cannot actually be evaluated, so the reward function is internally inconsistent.

(3) In Algorithm 1 (Line 766), the notation alternates between “budget $B$, max hops $m$”. Both $B$ and $m$ appear to mean “max hops”, which is confusing and should be unified.

(4) Around Line 237, the score $c$` and the threshold $\tau$ are both used in the reward but never clearly defined. 

W2. Limited efficiency analysis

The paper reports “average searches per question” as its main efficiency metric, but does not provide end-to-end latency or monetary cost in realistic settings (e.g., retrieval time over a large index, API/tool-call pricing). For real RAG systems, wall-clock delay and dollars per query matter, and that systems-level cost analysis is missing.

W3. Missing ablations on RL design

We see that “FrugalRAG-Explore (SFT)” vs “FrugalRAG (RL)” works, but we do not get deeper ablations, such as: (i) alternative stopping rewards, (ii) RL that also tunes query generation instead of only stopping, or (iii) RL from scratch without Stage 1 supervision (iv) training with backbone pipelines other than Qwen2.5-7B-Instruct. 

W4. Code release

There is no released code and no clear commitment to release it. 

W5. Evaluation metric concerns

Answer quality is judged mainly with a model-based evaluator (“MBE”) rather than exact match or human review. That means reported gains could reflect the judging model’s stylistic preferences instead of true factual correctness. For retrieval, the paper mostly reports recall and does not report precision, F1, MAP, or MRR.

W6. Grammar issues

There are several grammar issues. Examples:

(1) Line 230: “our only goal is to learn when to sufficient evidence has been gathered.”

(2) Lines 52–53: a long sentence ending with “(with potentially private documents), Typical solutions…” incorrectly capitalizes “Typical” after a comma and fuses two sentences.

(3) Line 54: “train on 90,000-1,00,000 examples...” mixes numbering styles; it should be “90,000–100,000.”

### Questions
Q1. Could you explain the technical concerns in W1?  
Q2. Do you have other cost measurements (e.g., wall-clock time, API/tool-call cost)?  
Q3. Do you have more RL ablation studies?  
Q4. How stable is RL training across random seeds and hyperparameters?  
Q5. Will you release the code (especially for the RL loop and reward)?  
Q6. Do you have other evaluation results such as F1, precision@k, MRR, or MAP?  
Q7. How would you make the stopping policy safer in high-stakes settings where stopping too early and hallucinating is unacceptable?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes FrugalRAG, a two‑stage finetuning framework that aims to optimize both accuracy and test‑time retrieval cost for multi‑hop QA in RAG systems. Stage 1 (“Explore”): supervised finetuning of a reasoner to maximize evidence coverage by generating diverse ReAct‑style thought/action/search tuples; dataset rollouts are constructed by greedily selecting the candidate query at each hop that maximizes recall against ground‑truth evidence. FINISH is included in only ~10% of traces to keep it in distribution while emphasizing exploration.  Stage 2 (RL): GRPO is used to learn when to stop searching. The reward penalizes stopping before or after an “optimal” hop count h^\* defined by when further retrieval no longer improves a metric c (thresholded by \tau); an additional format reward encourages valid tool calls. The optimization trains the reasoner only; the final answer generator is fixed.

### Strengths
1. Simple yet effective idea: learn when to stop searching rather than always searching more.  

2. Strong empirical results with small data (1k); especially on HotPotQA under standardized E5/FlashRAG settings

3. Adaptive compute validated quantitatively and qualitatively (fewer searches on easy, more on hard).  

4. Generalization to BrowseCompPlus with more hops than seen in training.  

5. Can pair with different retrievers and answerers; demonstration with CoRAG

### Weaknesses
I don't identify major weakness for this work, but have a few questions:

1. Reward design depends on oracle‑like signals (evidence recall) and somewhat opaque hyperparameters

2. No baseline learned/heuristic stoppers, so it’s unclear how much RL beats simpler termination rules.  

3. Efficiency claims would benefit from runtime/FLOPs and ablation on budget B and sample count v.  

4. Practicality depends on availability of ground‑truth evidence for training.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
3
