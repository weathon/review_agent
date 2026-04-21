# MolStitch: Offline Multi-Objective Molecular Optimization with Molecular Stitching

- Avg Score: 4.80
- Decision: Reject
- Scores: 3, 6, 6, 3, 6

## Abstract
Molecular discovery is essential for advancing various scientific fields by generating novel molecules with desirable properties. This process is naturally a multi-objective optimization problem, as it must balance multiple molecular properties simultaneously. Although numerous methods have been developed to address this problem, most rely on online settings that repeatedly evaluate candidate molecules through oracle queries. However, in practical applications, online settings may not be feasible due to the extensive time and resources required for each oracle query. To fill this gap, we propose the Molecular Stitching (MolStitch) framework, which utilizes a fixed offline dataset to explore and optimize molecules without the need for repeated oracle queries. Specifically, MolStitch leverages existing molecules from the offline dataset to generate novel `stitched molecules' that combine their desirable properties. These stitched molecules are then used as training samples to fine-tune the generative model, enhancing its ability to produce superior molecules beyond those in the offline dataset. Experimental results on various offline multi-objective molecular optimization problems validate the effectiveness of MolStitch. MolStitch has been thoroughly analyzed, and its source code is available online.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper introduces a framework, MolStitch, for generating molecules with desirable properties using only offline datasets. MolStitch works by “stitching” fragments from existing molecules in an offline dataset to create new molecules that combine multiple desired properties in a single structure. This approach leverages StitchNet, a neural network trained to produce valid molecular combinations, alongside a rank-based proxy model that assesses molecule quality through pairwise comparisons, which is claimed to enhance stability in multi-objective tasks. Experimental results indicate that MolStitch consistently outperforms existing methods in various offline molecular optimization benchmarks.

### Strengths
- Significance of Offline Molecular Optimization: Offline optimization is relatively less explored in the field of molecular generation, which is dominated by online oracle-based approaches. Offline optimization presents a fundamentally more challenging problem because it requires generating high-quality molecules from a static dataset without iterative property evaluation, which is especially relevant for applications like drug discovery where experimental validation is costly.
- Performance on Reported Metrics: The experimental results demonstrate that MolStitch achieves superior performance compared to existing methods in multi-objective molecular optimization. The rank-based proxy, StitchNet’s design for combining molecular fragments, and priority sampling collectively allow MolStitch to explore a broader space of candidate molecules, as evidenced by the results on standard metrics such as hypervolume and R2 indicator.

### Weaknesses
- Novelty and First Contribution Claim: The claimed novelty of molecular “stitching” may be overstated. Similar approaches, such as the methods presented by Jin et al. [1] and Guo et al. [2], also leverage substructure-based techniques for molecule generation, where certain fragments are prioritized for their property contributions and tested in multi-objective settings. The authors should review the relevant work more closely and moderate their claims on being the first to introduce this method.

- Choice of Baselines and Comparison Gaps: The separation of molecular optimization and model-based optimization (MBO) methods might be artificial, as molecular optimization techniques can often be integrated within MBO frameworks. This potential integration creates strong baselines that are essential for a comprehensive evaluation. Including such baselines would make the comparison more robust and clarify MolStitch’s performance relative to an enhanced baseline that combines MBO with molecular optimization approaches.

- Clarity and Completeness: The paper lacks some necessary details, particularly about the experiment setup and baseline methods. Greater clarity on the exact experiment design and further elaboration on baseline choices would make the methods and results more reproducible and transparent for readers.

### Reference
[1] Jin, Wengong, Regina Barzilay, and Tommi Jaakkola. "Multi-objective molecule generation using interpretable substructures." International conference on machine learning. PMLR, 2020.
[2] Guo, Minghao, et al. "Data-efficient graph grammar learning for molecular generation." arXiv preprint arXiv:2203.08031 (2022).

### Questions
- Docking Score as a Property (Line 353): Could you clarify why docking score optimization is presented as a separate task from molecular property optimization? Isn’t docking score simply another molecular property? Further explanation would clarify the differences between the objectives of each task.

- Offline Use of Molecular Optimization Methods: How were molecular optimization methods such as REINVENT adapted for offline settings? Some of these methods typically operate in online or iterative contexts. Additional details on any modifications would help assess the effectiveness of these methods in the offline scenario.

- Details on REINVENT-BO: The method REINVENT-BO doesn’t seem to align with the reference for Austin et al. Could you provide more information on this method and clarify the implementation used in this paper?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies an important problem, multi-objective molecular optimization. This paper mainly focuses on the offline setting. The authors propose Molecular Stitching (MolStitch), which leverages existing molecules from the offline dataset to generate stitched molecules and uses these generated samples to fine-tune generative models. Experimental results on various offline multi-objective molecular optimization problems validate the effectiveness of MolStitch.

### Strengths
- The proposed MolStitch only uses existing offline data and does need to query the oracle function.
- The overall framework of MolStitch is novel. The figure also presents the framework clearly.
- MolStitch outperforms baselines on several benchmarks.

### Weaknesses
- For fine-tuning the StitchNet, the new stitched molecule is not guaranteed to keep the desired properties.
- Since the StitchNet and generative model is pre-trained on large-scale ZINC dataset. Is it unfair to compare to other models that are not pre-trained? Is it possible to choose several baselines and use the same backbone network for comparison?
- Related work should be included in the main text. While the space is limited, it is important to keep this part. Some important multi-objective references are also missing. For example, RL-based [1] and GFlowNet-based [2]. I recommend the authors add a brief related work in the main text and move some less important descriptions to the appendix.

### Questions
See Weaknesses

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
4

### Summary
The authors propose an offline molecular multiobjective optimization algorithm, MolStitch, which can be made independent of querying oracle function by means of Direct property optimization. It is proposed that the properties of a molecule depend on the partial structure, and the viewpoint of multiple properties can be obtained when the property structures are spliced with each other.

### Strengths
The authors propose that different molecules possess different properties that the differences in properties depend on the structure or functional groups of the parts, and that new molecules with a full range of properties can be obtained when the 2-part structure is spliced. The motivation is clear and novel, and it is interesting to apply it in the direction of direct property optimization. The overall logic of the article is clearer.

### Weaknesses
The theory proposed by the authors is also somewhat problematic because the properties of a molecule can be determined by more than just a particular section of the structure and functional group, just for example, atoms may determine acidity and alkalinity, functional groups determine hydrophobicity, and the overall structure in turn determines properties such as boiling point. If the authors can explain clearly how the functionality or structure is clipped between different molecules, and if it is still by querying the oracle function, how accurate and generalized is this predictor of the query, and does it need a different predictor for different properties, this needs to be further discussed.

### Questions
1. how the molecules in the initial training set are clipped, and by what means are the clipped sites determined.
2. the fine-tuning model uses newly generated molecules, so must the properties of the newly generated molecules be better than the previous ones, and how much noise exists in them if they are still passed through the predictor?
3. the authors are deciding the dataset for fine-tuning by ranking, so how accurate is this ranking model and does it have the ability to generalize?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes a framework called Molecular Stitching to solve the problem of excessive reliance on oracle in existing molecular optimization scenarios. The author proposed the offline MOMO setting and a novel method, so that there is no need to call any oracle for molecular screening during the molecular generation (optimization) process. Experiments have demonstrated the effectiveness of the proposed method in multi-objective optimization.

### Strengths
1. This paper's motivation, that is, the problem it aims to solve is indeed a valuable problem in the MOMO scene.

2. The article is clearly written to explain the method, and I can easily understand how each module works.

3. The authors have considered the multi-objective challenge in MOMO. In fact, I agree that this is not an issue that should be ignored because there are too many phenomena in molecular properties that cannot be balanced due to natural conflicts. The existing MOMO methods can just complete this task, but does not consider or solve the problems brought about by multiple objectives.

### Weaknesses
1. The paper is not very well written. For example, especially in the introduction, I understand the importance of reducing the number of oracle queries, but what is the relationship with the performance of the proxy model that is introduced at great length? In other words, why does low proxy accuracy increase the number of oracle queries? In fact, online oracle calls are not absolutely related to the performance of the proxy model. For example, in DST[1], the proxy model is sufficient to support effective functional group editing. However, due to the limitations of its formulation, DST still needs an oracle to screen and obtain the optimal and more detailed connections between atoms. 

2. Motivation is good, but the proposed method does not fit well with motivation. I personally think that the number of oracle queries should take into account the labeled dataset and online generation. Some methods don't need the former, and some don't need the latter. The two parts should not be treated too differently. For example, if a method requires rdkit's online query but it can achieve effective MOMO, why not? If the author can show that the proposed method has an absolute advantage in the evaluation of the sum of the oracle query times of the two parts, then I agree. If plan to do this, please consider adding the oracle query times of the two parts of all baselines. Also, please note that LigGPT[2] also does not need to call oracle in the latter (please correct me if I am wrong).

3. I am very grateful that the author pays attention to the multi-objective problem in MOMO, which is ignored by most baselines. This is because multi-objective optimization itself is a huge challenge. MOMO needs to consider not only the optimization itself, but also how to balance multiple objectives. Although this paper adopts the Pareto solution, this problem has not been analyzed in detail, which is regrettable for the MOMO field. Frankly speaking, this is not a factor I consider when scoring, but I really hope that the author will add some necessary analysis. For example, do the molecules have natural property conflicts? And will MOMO cause gradient conflicts during the optimization process (for specific implementation methods, please refer to [3]). Of course, has the property conflict problem been resolved before and after Pareto was adopted?

4. About experiments. In my opinion, the authors unnecessarily restricted their experiments to offline optimization baselines, as they are not commonly found in MOMO tasks. For example, the ICT method does not seem to be designed for molecules. I suggest the authors add all the baselines mentioned in the DST paper and report the respective numbers of oracle calls during two stages required for all baselines. I would like to see whether the method in this paper has a clear advantage in the number of queries in both stages. Authors should consider reporting average property score (APS), which is a commonly used metric in the MOMO community.


[1] Differentiable Scaffolding Tree for Molecular Optimization. ICLR, 2022.

[2] LigGPT: Molecular generation using a transformer-decoder model. 2021.

[3] Pareto Deep Long-Tailed Recognition: A Conflict-Averse Solution. ICLR, 2024.

### Questions
N.A.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work introduces MolStitch as an approach to molecular design that uses a fixed offline dataset to design “stitched molecules”; this is in contrast to the more common iterative molecular optimization approaches that are able to query an oracle. The approach is inspired from trajectory stitching in offline RL. Generated molecules are scored by a proxy model trained to perform pairwise ranking of molecules’ optimalities defined by a scalarized property score. Scalarization weights are sampled from a Dirichlet distribution to achieve diversity along a Pareto front.

### Strengths
Both the fully offline and semi-offline optimization settings described in the introduction are of high importance to practitioners.

A large number of methods are included in the comparison, including both “standard” molecular optimization approaches and recently developed approaches for offline learning developed outside of the molecular context. Reported empirical results are strong in terms of mean performance, even if baseline methods may be within one standard deviation.

The adaptation of the generative model’s loss function from the traditional regression formulation to a DPO-like loss function appears to be novel in this context of molecular design.

The Appendices are very detailed in explaining related work, the problem setting, anticipated strengths and weaknesses of each approach, and detailing the experiments performed. They will be very educational for readers.

### Weaknesses
The method that seems central to MolStitch---to stich two molecules together and generate a new structure that combines substructures of each---is equivalent to a standard crossover operation in molecular optimization. Indeed, the authors pretrain their model using the rule-based crossover from Jensen’s GraphGA/GraphMCTS. A comparison to a GraphGA employing crossover operations and otherwise using the same ranking proxy model (e.g., for binary tournament selection) is not included.

However, the examples in Figure 17 suggest that the generative model, when proposing newly “stitched” molecules, pays very little attention to the two parent molecules. For me, this calls into question the entire premise of “stitching” as opposed to direct optimization of a generative model given a ranking proxy model. Even if the generative model is pretrained on the results of crossover operations, the stitched molecules here have extraordinarily little resemblance to the parent molecules.

The ablation in Table 3 suggests that the rank-based proxy is critical to performance, yet the other ablations and comparisons seem to lack evaluations using the rank-based proxy in combination with other generative methods besides REINVENT, including a GraphGA as mentioned earlier. Overall, the ablations still leave a murky impression of what aspects of MolStitch represent the most significant improvement over prior work. 

Relatively simple concepts for a venue such as ICLR are explained in unnecessary detail. For example, the form of a Dirichlet distribution, a 2-norm regression loss, Pareto optimality, pairwise ranking in Equation 13, and autoregressive token generation. Generic inequality and equality constraints in Equation 1 do not seem to serve a role in the example applications.

[not a score-driving weakness] Proxy model training focuses on pairwise ranking and is initially trained on unlabeled molecules under the assumption that high structural similarity (above a threshold $\delta$) implies that the objective scores of an unlabeled molecule should match that of the original molecule. Imposing this assumption can be accomplished through means besides pretraining (e.g., use of a Tanimoto kernel in a GP proxy model trained on the original regression task, or simple data augmentation for any proxy model). There is not specific justification for this particular approach, but the empirical results are strong.

[not a score-driving weakness] There is no Related Work section in the *main text* of the manuscript. 

[not a score-driving weakness] As a minor point regarding Appendix B.2, Bayesian optimization and scalarization are not mutually exclusive.

### Questions
The comparisons in Table 1 and 2 focus on the hypervolume; it is not clear how the multi-objective nature of the task is being considered here. The potential contributions of MolStitch related to its generative approach is distinct from its potential contributions related to sampling diverse scalarization weights. How is scalarization handled for the baseline methods included in this comparison? 

MolStitch is not a method that needs to be applied in a multi-objective context, fundamentally. Have benchmarks been performed on single objective tasks or multi-objective tasks with fixed scalarization weights using performance metrics other than HV? For example, the same top-10 AUC in PMO where some tasks were derived?

(My questions and stated weaknesses are an attempt to clarify the contributions made by this work; there are many combinations of modelling choices that are possible, and despite the inclusion of ablations, my impression is that the rank-based proxy is the largest contributor to performance. Taking this component and integrating it with other modelling choices would help verify or refute this. I acknowledge the empirical results are very strong.)

---

The additional experiments and explanations have helped clarify the contributions of the work. While I still believe the contribution is marginal in novelty, there is an empirical benefit in the tasks that are evaluated. The emphasis of how the work is introduced and described does not fully match the fact that the rank based proxy is the primary source of empirical improvement. I have raised my score to a 6 but do not feel strongly about its acceptance.

### Soundness
3

### Presentation
4

### Contribution
3
