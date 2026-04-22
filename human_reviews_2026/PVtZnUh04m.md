# AutoGPS: Automated Geometry Problem Solving via Multimodal Formalization and Deductive Reasoning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 8, 4, 4

## Abstract
Geometry problem solving presents distinctive challenges in artificial intelligence,
requiring exceptional multimodal comprehension and rigorous mathematical reasoning capabilities.
Existing approaches typically fall into two categories: neural-based and symbolic-based methods,
both of which exhibit limitations in reliability and interpretability. To address this challenge, we propose AutoGPS, a neuro-symbolic collaborative framework that solves geometry problems with concise, reliable, and human-interpretable reasoning processes.
Specifically, AutoGPS employs a Multimodal Problem Formalizer (MPF) and a Deductive Symbolic Reasoner (DSR).
The MPF utilizes neural cross-modal comprehension to translate geometry problems into structured formal language representations,
with feedback from DSR collaboratively.
The DSR takes the formalization as input and formulates geometry problem solving as a hypergraph expansion task,
executing mathematically rigorous and reliable derivation to produce minimal and human-readable stepwise solutions.
Extensive experimental evaluations demonstrate that AutoGPS achieves state-of-the-art performance on benchmark datasets.
Furthermore, human stepwise-reasoning evaluation confirms AutoGPS's impressive reliability and interpretability,
with 99\% stepwise logical coherence.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces a novel neuro-symbolic framework for GPS problems, featuring two key components: the Multimodal Problem Formalizer (MPF) and the Deductive Symbolic Reasoner (DSR). These components work collaboratively to address the challenges inherent in GPS problems. Experimental results demonstrate that the proposed method yields very promising outcomes, indicating significant potential in this area of research.

### Strengths
1. This paper is technically sound. The authors introduce two novel modules, the Multimodal Problem Formalizer (MPF) and the Deductive Symbolic Reasoner (DSR), both of which require substantial human effort to develop. These components have the potential to greatly benefit the GPS research community.
2. The experiments presented in this paper are both promising and comprehensive. The proposed method can be integrated with various base MLLMs, significantly improving their performance, especially in completion tasks.
3. Overall, the paper is well-written and easy to follow.

### Weaknesses
1. The authors state that MPF and DSR collaborate to formalize the original GPS problem. However, it is unclear whether this process may lead to incorrect problem formulations. The paper would benefit from a discussion and analysis of both successful and failed cases, including a reported success rate for the formalization process. Such information would be valuable for subsequent research to better understand the strengths and limitations of the proposed approach.
2. The paper does not report the API or inference costs associated with each method. As neuro-symbolic approaches often introduce additional inference and interaction steps when addressing GPS problems, it is important to assess whether these extra costs are justified by the observed performance improvements. Additionally, it would be worthwhile to investigate whether simpler techniques, such as self-consistency, could achieve similar performance gains within comparable evaluation budgets.
3. The discussion of related work could be expanded. In particular, the paper should cite and analyze earlier neuro-symbolic methods for mathematical reasoning problems, such as [1], [2], [3], and [4]. This would help position the proposed approach within the broader context of existing research.

**Reference** 

[1] Zenan Li, Zhi Zhou, Yuan Yao, Xian Zhang, Yu-Feng Li, Chun Cao, Fan Yang, Xiaoxing Ma. **Neuro-Symbolic Data Generation for Math Reasoning.** NeurIPS 2024.

[2] Ning Shang, Yifei Liu, Yi Zhu, Li Lyna Zhang, Weijiang Xu, Xinyu Guan, Buze Zhang, Bingcheng Dong, Xudong Zhou, Bowen Zhang, Ying Xin, Ziming Miao, Scarlett Li, Fan Yang, Mao Yang. **rStar2-Agent: Agentic Reasoning Technical Report.** Arxiv 2025.

[3] Zenan Li, Zhaoyu Li, Wen Tang, Xian Zhang, Yuan Yao, Xujie Si, Fan Yang, Kaiyu Yang, Xiaoxing Ma. **Proving Olympiad Inequalities by Synergizing LLMs and Symbolic Reasoning.** ICLR 2025

[4] Weiming Wu, Zi-kang Wang, Jin Ye, Zhi Zhou, Yu-Feng Li, Lan-Zhe Guo. **NeSyGeo: A Neuro-Symbolic Framework for Multimodal Geometric Reasoning Data Generation.** Arxiv 2025.

### Questions
Please refer to the `Weakness` section.

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper presents AutoGPS, a framework for automated geometry problem solving that integrates multimodal formalization with deductive reasoning. Unlike traditional text-only or vision-only systems, AutoGPS unifies textual problem statements and geometric diagrams into a consistent symbolic representation, which is then solved through formal logic-based reasoning. The system includes:
1. A multimodal formalization module that parses text and visual diagrams into formal symbolic predicates.
2. A deductive reasoning engine that performs geometric theorem proving through symbolic inference.
3. A data generation and verification pipeline for training and evaluating multimodal geometry reasoning systems.

Experiments demonstrate AutoGPS’s superiority over both LLM-based and neuro-symbolic baselines on geometry reasoning benchmarks, including synthetic and real-world datasets. The approach significantly improves consistency between textual and visual modalities and exhibits better generalization to novel problem types.

### Strengths
1. Innovative integration of modalities:
The combination of natural language, diagram understanding, and formal reasoning is both technically challenging and conceptually elegant. The multimodal formalization module is well-motivated and executed with a clear architecture.

2. Clear reasoning pipeline:
The formal-to-symbolic transition is described thoroughly, including explicit steps for entity detection, relation extraction, and logical grounding. The modular design supports interpretability and verifiability.

3. Strong empirical performance:
Experimental results demonstrate substantial improvements over both pure neural models (e.g., GPT-4V, Flamingo) and previous neuro-symbolic systems. The inclusion of accuracy, formal consistency, and visual grounding metrics provides a holistic evaluation.

### Weaknesses
1. Limited scalability and automation.
The formalization process still relies partly on rule-based heuristics for entity alignment and relation mapping. It is unclear how the system scales to more complex, noisy, or real-world diagrams with ambiguous geometry.

2. Dataset construction bias.
The training and evaluation datasets appear to be semi-synthetic or curated from well-structured geometry problems. There is limited discussion on how AutoGPS performs on imperfect or non-standard problem statements often found in educational or competition settings.

3. Insufficient discussion of failure cases.
The paper would benefit from analyzing situations where the symbolic representation fails (e.g., misalignment between text and diagram) and discussing how these errors propagate through reasoning.

### Questions
As discussed above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the challenges of multimodal understanding and rigorous reasoning in automated geometry problem-solving. It proposes a neuro-symbolic framwork called AutoGPS which is composed by the Multimodal Problem Formalizer (MPF) and the Deductive Symbolic Reasoner (DSR) modules. MPF is used to multimodal understanding, translating the diagram and natural text to rigorous formal languages, while DSR is used to perform deductive reasoning to infer the final answer on a meticulously constructed supergraph. Experimental results demonstrate the effectiveness of the proposed AutoGPS framework.

### Strengths
1. The design of the overall neuro-symbolic architecture is reasonable, and the MPF and DSR modules are well defined.
2. The introduction of the multimodal alignment stage in MPF is effective in filling the missing semantic information that is not captured by the pixel-level diagram parser.
3. The experimental results are convincing and demonstrate the effectiveness of the AutoGPS framework.
4. The writing and structure of the paper are clear and easy to follow.

### Weaknesses
1. My main concern lies in the originality of this paper. First, in MPF, the text parser $M_t$ should properly cite InterGPS (Line 214). Second, in Line 274, the authors state that "solving algebraic relations remains out of its scope". However, in my understanding, AlphaGeometry adopts a DD + AR symbolic reasoning engine, where AR stands for algebraic reasoning. Therefore, the DSR module appears highly similar to the DD + AR reasoning engine of AlphaGeometry, as well as to the hypergraph expansion component. It would be helpful if the authors could clarify the unique innovations of AutoGPS.
2. Since MPF heavily relies on the diagram parser (PGDPNet), which is trained solely on PGDP and Geometry3K-style data, it may struggle to seamlessly generalize to other styles of geometric diagrams. This limitation further constrains the scalability of AutoGPS.
3. The compared methods are limited. The authors should include comparisons with more recent neuro-symbolic approaches, such as FGeo-HyperGNet [1].

[1] FGeo-HyperGNet: Geometric Problem Solving Integrating FormalGeo Symbolic System and Hypergraph Neural Network

### Questions
Besides the weakness above, I wonder what the average time cost of AutoGPS is for solving a geometry problem compared to other baselines.

### Soundness
3

### Presentation
3

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
This paper propose AutoGPS, a neural symbolic plang geometry problem solver. AutoGPS consists of a MPF to parse the plane geometry diagram into symbolic language, and further use a DSR to conduct deductive symbolic reasoning based on a hyper graph to solve the problem, and finally obtain the answer by extracting the solution path from the graph. Experiments on Geometry3K and PGPS9K show the proposed method is effective on solving PGPs.

### Strengths
1.	The design of the framework is reasonable. AutoGPS first parse the diagram into symbolic languages, further construct a hyper graph of the given problem and search the solution path on the graph to finally get the solution steps, ensure the interpretability.
2.	Comparing to the mentioned previous works in this paper, AutoGPS improved the problem solving accuracy than previous baselines, which show the effectiveness of the proposed method.
3.	In the human evaluation, AutoGPS output stepwise solutions with ideal accuracy, outperformend other pure MLLMs.

### Weaknesses
1. As I know, using hyper graph in solving plane geometry probelms (PGPs) was already explored by FGeo-HyperGNet [1], this paper not give appropriate discussion on this previous work, what is the difference of the hyper graph between AutoGPS and HyperGNet. Meanwhile, HyperGNet achieved 91.99% on Geometry3K which is higher than AutoGPS with 81.6.

2. To this end, the contribution of this paper is limited, as the diagram parsing mainly relies on PGDP and the hypergraph has already been used in other work. Despite the author proposing a deductive symbolic reasoning framework to assemble these modules and give stepwise solutions that are easy for humans to read, I think this paper does not satisfy the bar of the ICLR community.

3. The description of symbolic solvers is not appropriate. Indeed, symbolic solvers do not directly output the results without solution steps, like Inter-GPS, which was proposed along with the Geometry3K dataset. Inter-GPS uses a theorem predictor (also has a search algorithm) to predict the theorems at each step, and based on these theorems, to conduct symbolic reasoning on the existing problem conditions to get the final target answer. And the reliable interpretability is one of the advantages of symbolic solvers. The author said symbolic solvers are hard for humans to read, it is easy to tackle by using an LLM to translate the symbolic solution steps into human language, which will be used to understand.

4. Limited experiment dataset, benchmark. This paper only conducted experiments on the Geometry3K and PGPS9K, while actually PGPS9K was expanded from Geometry3K. It is necessary to conduct experiments on popular benchmarks in the PGP solving area, such as Math-Vista Geo, Math-Verse. The current experiments are not convincing. 

5. Limited generalization ability of AutoGPS. As mentioned in the above weakness, I am wondering if AutoGPS is hard to solve other domains of PGPs, such as GeoQA, which is also a foundation dataset for the research of solving PGPs, as the diagram parsing module in the AutoGPS is leveraged from PGDP, which is a delicate model designed for parsing diagrams in the style of Geometry3K. If AutoGPS is not able to solve problems in GeoQA and other benchmarks, the scope of this work is not sufficient for ICLR.

Ref:

[1] FGeo-HyperGNet: Geometric Problem Solving Integrating FormalGeo Symbolic System and Hypergraph Neural Network, in IJCAI 2025.

### Questions
I would like to see what is the performance of the diagram parsing in AutoGPS by using the feedback and refinement process, comparing to the original PGDP work, which already achieved 99% accuracy on the PGDP5K dataset.

### Soundness
2

### Presentation
3

### Contribution
2
