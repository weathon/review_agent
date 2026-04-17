# Mitigating Spurious Correlations in LLMs via Causality-Aware Post-Training

- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
While large language models (LLMs) have demonstrated remarkable capabilities in language modeling, recent studies reveal that they often fail on out-of-distribution (OOD) samples due to spurious correlations acquired during pre-training. Here, we aim to mitigate such spurious correlations through causality-aware post-training (CAPT). By decomposing a biased prediction into two unbiased steps, known as \textit{event estimation} and \textit{event intervention}, we reduce LLMs' pre-training biases without incurring additional fine-tuning biases, thus enhancing the model's generalization ability. Experiments on the formal causal inference benchmark CLadder and the logical reasoning dataset PrOntoQA show that 3B-scale language models fine-tuned with CAPT can outperform both traditional SFT and larger LLMs on in-distribution (ID) and OOD tasks using only 100 ID fine-tuning samples, demonstrating the effectiveness and sample efficiency of CAPT.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces Causality-Aware Post-Training (CAPT), a method aimed at mitigating spurious correlations between events and outcomes. CAPT operates by post-training on augmented datasets, in which LLM-extracted event-related text is heuristically substituted with randomized symbols. This approach seeks to encourage the model to learn genuine causal relationships rather than relying on coincidental associations.

### Strengths
1. The proposed CAPT method is novel and well-motivated. Experimental results demonstrate its effectiveness on the PrOntoQA and CLadder tasks.
2. The paper is clearly written and follows a coherent logic, presenting the motivation and methodology from a causal perspective.

### Weaknesses
1. The proposed debiasing framework may be limited in the types of tasks it can effectively address. For instance, concentrating events into English characters might allow the LLM trained on the augmented dataset to quickly adapt to the causal QA task where questions follow a similar logic. To investigate this, an ablation study using different numbers of replacement tokens (e.g., 10, 26, 50, 100) rather than just the English alphabet could provide more insight.
2. CAPT relies on LLM-based extraction of event-related text, which necessitates prompt design for new tasks. Including a more detailed discussion on prompt design would help others apply the CAPT method more effectively.

### Questions
N/A

### Soundness
3

### Presentation
3

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
The paper proposes a post-training framework to improve large language models’ reasoning robustness by reducing reliance on spurious correlations learned during pretraining. The method introduces a two-step process: event estimation, where key events in text are identified, and event intervention, where these events are abstracted and randomly replaced with symbols to remove non-causal dependencies. This approach aims to help models focus on causal reasoning structures rather than surface associations. CAPT is evaluated on reasoning benchmarks such as CLadder and PrOntoQA, showing improved out-of-distribution performance and sample efficiency compared to conventional fine-tuning. The paper formalizes its approach using a structural causal model, situating CAPT as a causality-driven method for enhancing reasoning generalization in large language models.

### Strengths
1. The paper offers a clear and elegant causal perspective on how spurious correlations arise during large-scale pretraining. By introducing an SCM that separates event variables and causal structures, it reframes the debiasing problem from a purely statistical to a causal reasoning viewpoint.

2. Experimental results show that CAPT significantly improves OOD performance on causal and logical reasoning benchmarks (e.g., CLadder and PrOntoQA), even with very limited fine-tuning data.

### Weaknesses
1. CAPT relies heavily on the accuracy of the event estimation step, but the paper does not evaluate or analyze this component. If the event extraction model produces incorrect or inconsistent results, these errors will propagate through the entire causal pipeline, undermining the validity of the subsequent intervention and reasoning stages. I think an error analysis or ablation study on event estimation quality would be necessary to ensure the framework’s robustness.
2. While the paper provides a solid conceptual discussion of how spurious correlations arise and how causality can mitigate them, the experimental setup feels too simplified to validate those theoretical claims. The “anti-sense” and “non-sense” test sets primarily modify entity or symbol names, but they do not rephrase sentence structures or reasoning expressions, meaning the test data remain largely in-distribution. To better evaluate out-of-distribution generalization, the experiments should include paraphrased or semantically restructured inputs that test CAPT’s ability to handle genuine linguistic variation beyond symbol replacement.
3. Although the paper tries to contrast CAPT with prior work which only operates on **entity-level** symbolic substitution that CAPT has novelty in handling event-level spurious corerlations, the event definition looks still pretty simple, and event replacement still relies on textual templates, which may not truly separate causal structure from surface expression. For example, the experiments such as the anti-sense and non-sense sets. I would expect the authors’ claim about addressing the event-level research problem to be more thoroughly and empirically justified.
4. The evaluation primarily focuses on proprietary or closed models (e.g., GPT-4o-mini). Including results from more open-source models would strengthen the paper.

### Questions
1. Is the random symbol assignment deterministic per dataset, or randomized at each fine-tuning step? How do you control for variation?
2. Since GPT-4o-mini is used for event estimation, did you test CAPT’s sensitivity to the choice of model or prompting strategy? How would you justify the use of this model? Could a weaker or differently aligned model yield similar benefits?

### Soundness
3

### Presentation
3

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
In this paper, the authors aim to mitigate spurious correlations in large language models (LLMs) to improve their out-of-distribution (OOD) generalization ability. They define the underlying Structural Causal Model (SCM) intuitively and formulate it as a collider bias problem. Assuming that the spurious correlation paths induced by this bias can be completely blocked by event-related or contextually correlated but logically irrelevant information, they propose using event estimation and event intervention to mitigate this spurious correlation. The authors apply their proposed method to several LLMs for post-training on public benchmark datasets, and the results demonstrate an improvement in OOD generalization performance.

### Strengths
1. Incorporating causality into LLMs is an interesting direction.

2. The experimental setup is described in detail, which should facilitate replication of the results.

3. While there are some theoretical and technical flaws with Figures 1 and 2, their format is clear and helps readers understand the motivation behind the proposed method effectively.

### Weaknesses
1. The proposed SCM raises significant concerns regarding both its correctness and normalization. (1) The definitions of $X$, $Y$, and $E$ are somewhat reasonable, with $X$ representing the prompt, $Y$ representing the answer, and $E$ representing events (which should be a multidimensional variable or a list of events, but it is not clearly depicted in the figure). However, the definition of $S$ is incorrect. The authors use $S$ to represent the underlying logic between multiple events ($E$) and between events and answers ($E$ and $Y$), as illustrated in line 141. However, it should be represented by edges, not nodes, in the causal graph. (2) The use of an unmeasured confounding variable ($U$) to represent the collider bias in the figure is also incorrect. Bias itself cannot be defined as a variable; rather, it is the relationship (edges in the graph) between variables that leads to bias. A correct causal graph should condition on a domain indicator variable ($X,Y,E \rightarrow D$) to represent this bias. Only by this definition, the authors’ claim of collider bias in estimating $P(Y \mid X)$ holds: Given that LLMs are pre-trained using samples from the pretraining dataset ($X$, $Y$, and $E$), and the pre-training domain is denoted by $D=1$, the pre-trained model learns $P(Y \mid X, D=1)$, which is only applicable to the pre-training domain, rather than $P(Y \mid X)$ applicable to all potential domains, introducing collider bias. The current definition provided by the authors is not logically consistent and deviates from established conventions in the field. [1]

2. Due to the fundamental problems with the SCM definition, the proposed method cannot effectively address the collider bias problem. (1) The authors assume (though not explicitly stated and only represented in the causal graph) that the selection backdoor path between $X$ and $Y$ can be completely blocked by $E$, such that adjusting for $E$ (whether through intervention or random assignment) can mitigate the bias. However, in reality, since the pre-trained model is trained on the joint distribution (and conditional distribution) of $X$, $Y$, and $E$ from the pre-training dataset, all three variables are causes of the selection bias. As a result, the selection backdoor path between $X$ and $Y$ cannot be blocked, as $X$ and $Y$ themselves directly cause the selection bias. It means that the assumption is never satisfied and adjusting for $E$ cannot mitigate the bias. (2) Even within the authors’ own flawed SCM definition, the proposed method contains several significant issues. For instance, the authors assume that "pre-trained LLMs are trained on distributions so large and diverse that colliding biases affecting $P(E\mid X)$ are diluted and negligible." However, if LLMs were indeed trained on such a sufficiently large and diverse distribution, there would be no selection bias at all. In other words, the authors’ solution to collider bias is essentially based on the assumption that there is no collider bias! Even in a very extreme case, where only the marginal distribution of $E$ is sufficiently large and diverse (since a prompt contains many events), while $X$ and $Y$ distributions remain insufficiently large and diverse, $E$ still cannot effectively block the selection backdoor path because $E$ will not be part of any selection-related path between $X$ and $Y$ in this case. [2]

3. Most claims made in this paper lack theoretical proofs, which may be the reason why the authors have overlooked the serious flaws mentioned above.

4. The paper does not seem to thoroughly review existing work on using causal methods to address the OOD generalization problem in LLMs. For instance, [3, 4] appear to be related studies that were not discussed.

[1] Hernán, M. A., & Robins, J. M. (2010). Causal inference.

[2] Bareinboim, E., Tian, J., & Pearl, J. (2022). Recovering from selection bias in causal and statistical inference. In Probabilistic and causal inference: The works of Judea Pearl (pp. 433-450).

[3] Feder, A., Wald, Y., Shi, C., Saria, S., & Blei, D. M. (2023). Causal-structure Driven Augmentations for Text OOD Generalization. In NeurIPS.

[4] Zhang, K., Zhang, D., Wu, L., Hong, R., Zhao, Y., & Wang, M. (2024). Label-aware debiased causal reasoning for natural language inference. AI Open, 5, 70-78.

### Questions
Please see Weaknesses. My primary concern lies in Weaknesses 1 and 2. If they cannot be adequately addressed, the paper would contain significant theoretical and technical flaws, making it unsuitable for publication in its current version.

### Soundness
1

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
3

### Summary
The paper introduces CAPT: a post-training method for debiasing an LLM prediction on reasoning tasks using causal principles. The aim of the method is to disentangle syntactical event information from the reasoning structure. The method does this by identifying all events described in the question and replacing them with arbitrary variable names so that the LLM focuses on the logical reasoning structure. Experiments show superior performance compared to standard SFT with low training samples.

### Strengths
1. The paper proposes an interesting approach based on causality for mitigating spurious correlations. While the general theory is not novel, its proposed instantiation is an original solution to a crucial problem in LLMs and machine learning in general.
2. The theoretical elements of the paper are intuitive and well justified at a high-level (although some clarifications could be made) and the experiments are extensive and support the claims made.
3. The paper is well explained and easy to read.

### Weaknesses
1. While the approach is intuitive, some of its theoretical justification is unclear. The authors perform an intervention on a confounding factor $E$ between Equations (4) and (5) but do not represent it as a do-operation, making the connection between the two equations unclear and lacking justification at a theoretical level. Indeed, the true causal graph cannot directly be observed, especially $S$ or $E$. The claim that replacing event variables with abstract values amounts to intervening on $E$ seems dubious. In particular, since non-causal information can be transmitted via other syntactical cues independent from the identified events.
2. The method seems tailored for specific problems where syntactical cues can be easily identified and disentangled as event variables, i.e. $S$ can be recovered from $X$ after filtering out events $E$. This is the case of Cladder as it is a synthetic benchmark where events variables are inserted into template sentences but this approach is unlikely to generalize beyond such specific cases and the paper does not provide sufficient evidence justifying it. In particular, events may contain information necessary for reasoning that would be filtered out by the proposed procedure.
3. Given the limitations in term of generalization discussed above, the paper would gain by demonstrating its performance on additional datasets. In particular, the non-sense dataset of Cladder corresponds to a ground-truth execution of CAPT but does not highlight a large performance gap with sensical and anti-sensical versions of the dataset in the original Cladder paper, limiting the interest of the proposed method.
4. The results shown in Table 2 are a bit hard to interpret. In particular, results with full SFT on Cladder are similar or better than the ones with CAPT for the same budget and the difference in performance is often within the standard deviation (although STD is large for the baseline).

### Questions
1. Could you clarify what you mean in Section 3.2 in the paragraph below Equation (3) when you say "$E$ and $Y$ are both token-level variables, while $S$ is not"?
2. Could you further justify how your approach amounts to an intervention on $E$? Wouldn't intervening on $X$ to separate the link between $E$ and $Y$ be more appropriate for mitigating spurious correlations since the observation of $X$ create a causal path $E \rightarrow X \leftarrow S \rightarrow Y$?
3. How do you perform the marginalization over $S$ in Equation (5)?
4. How do you handle cases where syntactic cues and information specific for the reasoning task are entangled? E.g. how do you handle the example given in Section 3.3.3 where the "alarm" in symbol 1 is the same as the "alarm" in symbol 3 despite the loss of this information when replacing with said symbols?
5. How do you interpret the high standard deviation on PrOntoQA with the standard baseline model?
6. How do you interpret the difference in performance on the anti-sensical and non-sensical splits while they are identical in the original Cladder paper?

### Soundness
2

### Presentation
3

### Contribution
2
