# Adversarially Pretrained Transformers May Be Universally Robust In-Context Learners

- Decision: Accept (Poster)
- Scores: 8, 6, 4, 6

## Abstract
Adversarial training is one of the most effective defenses against adversarial attacks, but it incurs a high computational cost. In this study, we present the first theoretical analysis suggesting that adversarially pretrained transformers can serve as universally robust foundation models—models that can adapt robustly to diverse downstream tasks with only lightweight tuning. Specifically, we demonstrate that single-layer linear transformers, after adversarial pretraining across a variety of classification tasks, can generalize robustly to unseen classification tasks through in-context learning from clean demonstrations (i.e., without requiring additional adversarial training or examples). This universal robustness stems from the model's ability to adaptively focus on robust features within given tasks. We also identify two open challenges for attaining robustness: the accuracy–robustness trade-off and sample-hungry training. This study initiates the discussion on the utility of universally robust foundation models. While their training is expensive, the investment would prove worthwhile as downstream tasks can obtain adversarial robustness for free. The code is available at https://github.com/s-kumano/universally-robust-in-context-learner.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper considers the adversarial robustness of pre-trained transformers using in-context learning. They derive a set of theoretical results for a single-layer "linear" transformer and show that adversarially trained transformers can be universally robust in-context learners. Their results improve the mathematical validity of previously empirically and intuitively understood concepts in adversarial robustness, such as the number of redundant, robust, and non-robust features, the effect of the strength of the adversarial perturbation, and the robustness-accuracy tradeoff. Finally, they briefly validate some of the theorems with simple experiments.

### Strengths
- The theoretical results are significant in improving the mathematical understanding of adversarial robustness under norm-bounded perturbations.
- The paper is organized and written well, and while the theory is dense, most of the main text is presented and discussed with clarity and intuition.

### Weaknesses
- The experiments are brief, and Table 2 is a somewhat expected result, regardless of the contributions of the paper. However, the brevity of the experiments is made up for by the theoretical results.
- The proofs are far too long and tedious. I understand that proving the theorems in a more succinct manner is non-trivial, but this still makes verifying the proofs rather difficult. Regardless, at a cursory glance, the proofs seem correct and seem to follow sound logic.

### Questions
- Lemma 3.3: This lemma seems integral to the rest of the derivations. However, the intuition behind the lemma and its connection to the following theorems are slightly difficult to grasp. Could the authors elaborate on this further and add the additional elaborations to the text?

- Line 199: “Taylor’s law (Taylor, 1961), is observed in a wide range of natural datasets and distributions.” To improve the flow of reading, it would be good to mention that examples of this are provided further below in the text.

- Section 3.2: “Warm-up” may be a misleading term, as it usually pertains to warm-up during training of a neural network. Perhaps it’s better to use another term or simply call the section “linear classifiers and oracle.”

- Section 3.2: Although a reasonable assumption, the authors should mention that $w \in [0,1]^d$ and provide a brief justification.

- Line 288: “non-linearity and non-convexity in the trainable parameters P and Q”: parameters cannot be linear or non-linear, convex or non-convex; it is a property of functions or sets. Please elaborate or correct this notation.

- Line 289: “High non-linearity” is a vague term; perhaps it is better to bundle all mentioned non-linearities as simply “non-linearity” as a boolean attribute.

- Thrm 3.4: Shouldn’t the second case be “$0<\epsilon \leq$” instead of “$\epsilon =$”? Additionally, there is a discontinuity between the thresholds in cases 2 & 3. Could the authors elaborate on why this is and what happens in the left-out values of $\epsilon$?

While I currently suggested "Accept", should these minor considerations be addressed, I believe the paper is worthy of a "Strong accept" score.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this paper, the authors theoretically demonstrate that single-layer linear transformers, after adversarial pre-training across multiple classification tasks, can robustly adapt to previously unseen classification tasks via in-context learning without requiring any additional training. The authors also conduct small-scale empirical experiments to provide preliminary evidence supporting their theoretical claims.

### Strengths
- The paper is clearly written and easy to follow.
- The topic is promising. If robustness can indeed be efficiently achieved during pre-training and transferred to downstream tasks, it would be highly meaningful.
- The paper seems to provide solid and convincing theoretical analysis.
- The authors also offer empirical results that provide a certain level of support for the proposed approach.

### Weaknesses
- The paper is limited to single-layer linear transformers, and it remains unclear whether the theoretical results can generalize to more realistic multi-layer non-linear transformers, which form the basis of modern foundation models.
- The empirical evaluation is also restricted to single-layer linear transformers. I suggest conducting experiments on multi-layer non-linear transformers and on more complex datasets to better demonstrate the generalizability of the conclusions (not necessarily large-scale, but at least small-scale experiments on nonlinear multi-layer transformers would strengthen the claim.)
- It would be helpful if the authors could discuss the practicality of achieving robustness during pre-training and transferring it to downstream tasks, compared to performing robust fine-tuning at the adaptation stage (for example [1]). 

[1] AutoLoRa: An Automated Robust Fine-Tuning Framework. ICLR 24

### Questions
Please see the weaknesses.

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
4

### Summary
The authors demonstrate that adversarially pretraining on a mix of classification tasks can make a (single-layer, linear) transformer act as a “universally robust” in-context learner: once pretrained, it can adapt to new classification tasks from clean demos and stay robust to attacks on the query, without any downstream adversarial training. The theory says robustness comes from the model learning to attend to task-shared robust features.

### Strengths
1. the paper is in general well written with clear notation and explanations 
2. The author provides a formal analysis that proves task-transferable robustness under a matched threat model. 
3. The author provides a good experiental setup with clear sample creation scheme.

### Weaknesses
1. The proofs are for single-layer linear transformers with matched adversaries; the paper text, as described, markets this as universally robust foundation models. For it to be an universally robust model, I would expect an transferability for different perturbation class and distribution shift setup. The whole story depends on the downstream adversary matching (or being close to) the one used in pretraining.

2. Another key theoretical baseline missing is that the author did not compare the no-pretraining case. Can the author demonstrate an separation or difference in sample complexity between adversarial training + ICL over direct linear regression on the test distribution. And is there an way to provide both an upper and lower bound on the error. E.g. LR > Pretrain + ICL> Adv + ICL > Robust LR ? 

3. The explanation for the connection between theorem 3.5 and 3.6 is confusing. I do not observe an strong separation and see why adversarial pretraining leads to an strictly smaller robust error. 

4. The design of the training dataset is quite arbitrary and to me it directly encodes the robust feature and thus it is able provide generalize to the test dataset which contains more weaker signals. To demonstrate the universality, I think the author should compare the training and test dataset with general distributions.

In addition, the theory already says transfer happens when the new task uses the same robust features as the pretraining mix. That’s a strong precondition. The author should discuss the scenario where the training data covers an diverse set of robustness features. Otherwise, this distribution coverage assumption is too strong. Similar thing should be analyzed in the experient as well.

5. As the author mentions, theorem 3.7 requires small N regime, the author should consider include an sample complexity bound that can be directly compared against standard ICL case as in [1].


[1] Ruiqi Zhang, Spencer Frei, and Peter L. Bartlett. Trained transformers learn linear models in-context. JMLR, 25(49):1–55, 2024b.

### Questions
1. The author assumes clean demonstrations for the new task. How sensitive is the in-context robustness to label noise in the demonstrations? 
2. How much of the benefit comes from task diversity in adversarial pretraining vs. adversarial pretraining itself? In other words, if I adversarially pretrain on fewer tasks, does the cross-task robust ICL degrade smoothly or abruptly?
3. It is known that non-robust feature can also transfer. Can the author provide an in-depth analysis of the learned transformer weight to clarify which component are robustly learned parameters and which is not.

### Soundness
3

### Presentation
4

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
The paper provides a theoretical investigation of the robustness of single-layer linear transformers to adversarial perturbations to query tokens when performing in-context classification.

### Strengths
- Generally well written paper.
- The core insight of the paper (theorem 3.6) -- that universally robust linear transformer (for very particular class of classification tasks) are realizable is moderately surprising and interesting imo.
- Other main results of the paper (accuracy-robustness tradeoff, need for larger in-context datasets) are well presented though not that surprising.
- The theoretical results are interpretable and provide some intuition about accuracy-robustness trade off.
- The paper empirically validates its theoretical results.
- Limitations of the paper are well-acknowledged.

### Weaknesses
Most of the weaknesses of the paper relate to the (narrow) assumptions authors make to make the theoretical results tractable. While typical of theory papers, these are nevertheless weaknesses as they limit the relevance of these results to practical contexts.
- The paper only studies single-layer linear transformers. This is a major weakness of the paper; as all the results pertain to this narrow class of models, I am not certain whether there is something special about this class of models, or can we expect some variation of these results to hold for other types of transformers as well. Given the relative difficulty of theoretical investigation for other types of tansformers, including some sort of empirical investigatons of other transformer classes would be helpful.
- The paper only considers a single type of perturbation (L-infty norm).
- The setup used in the paper is somewhat artificial.
- Theorem G.1 is not empirically validated (I guess this is because these transformers don't length generalize?)

Minor:
- In lines 393-394, I was initially confused by the current phrasing and would suggest rephrasing to "For example, when α = 160/255 and β = 8/255, the standard model becomes vulnerable at dvul ≳ 20drob, whereas the adversarially pretrained model **becomes vulnerable at dvul ≳ 400drob**".

### Questions
- Do authors have any comments or thoughts about whether universally robust multi-layer linear transformers, or softmax transformers, are realizable or not? I would in particular appreciate any insights that authors may have on the interplay between robustness and depth of the transformer. 
-

### Soundness
3

### Presentation
3

### Contribution
2
