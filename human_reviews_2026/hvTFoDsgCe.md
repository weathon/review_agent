# Rethinking the Definition of Unlearning: Suppressive Machine Unlearning

- Avg Score: 2.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 2, 4

## Abstract
Machine unlearning, an emerging issue of privacy concern in the deep learning era, is practically motivated by the *data removal* from training or *knowledge suppression* of utility on that data. Unfortunately, retraining via data removal, which has been understood as the gold standard, does not elucidate how much we suppress the model's knowledge on the target. The existing definition well covers an *exact* or *approximate* unlearning only with a removal perspective, yet failing to encompass knowledge suppression incurred via unlearning. Moreover, suppression is tightly entangled with removal in a way that more knowledge suppression obviously leads to significant divergence from exact and approximate unlearning, thus motivating us to rethink the definition of machine unlearning. We formally introduce a novel definition of *Suppressive Machine Unlearning*, encompassing how far the unlearned model is from retraining, i.e., $(\varepsilon,\delta)$-approximate unlearning, and how much the model's utility becomes suppressed, i.e., $\kappa$. To illuminate the formal dynamics between removal and suppression, we reveal the trade-off between the removal guarantees ($\varepsilon, \delta$), which quantifies how much it deviates from an idealized retraining and $\kappa^*$, which is the requested level of suppression.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a new definition for machine unlearning called "Suppressive Machine Unlearning." This definition aims to formally capture two aspects: 1) the proximity to a full retrain, similar to ($\varepsilon$, $\delta$)-approximate unlearning, and 2) a new "suppression" parameter, $\kappa$, which quantifies the reduction in the model's utility on the specific data to be forgotten. The authors present a theoretical analysis of the trade-off between these components and provide experimental results to support their framework.

### Strengths
* The paper introduces a new definition, "Suppressive Machine Unlearning," which is a reasonable goal. Attempting to formally unify the concepts of removal (proximity to retraining) and suppression (utility reduction) is a relevant direction for the field.

* The discussion in lines 466-473 regarding failure modes is interesting. Distinguishing between true data suppression and a general loss of model capability is an important problem, and the paper correctly identifies this. This part of the discussion is a good starting point for a more in-depth analysis.

### Weaknesses
Weaknesses below are ordered more or less following importance:

1. **Questionable Novelty.** The paper's core framing appears to have significant overlap with prior work, particularly the omitted paper "Distributional Machine Unlearning via Selective Data Removal" by Allouah et al. (ICML'25 Machine Unlearning for Generative AI). That paper seems to formalize and quantify several of the key ideas presented here, such as utility on a target distribution (Def 3.1 in this paper vs. Prop. 2 in the omitted paper). The conceptual and technical novelty of this work, when compared to this relevant paper, is not clear.

2. **Potentially Vacuous Bounds.** The main theoretical result, Theorem 4.4, and its consequences (e.g., Eq. 14) may be vacuous in practice. To maintain any reasonable utility, DP-based methods often require an ε on the order of 10 or more. In this regime, the bound given (which is exponential in ε₁) would become extremely large, offering no meaningful guarantee even for a single data point deletion, let alone multiple.

3.1 **Limited Theoretical Novelty.** Several of the theoretical results seem to be reformulations of existing concepts. For instance, Lemma 4.2 and Corollary 4.3 are essentially restatements of known group privacy results in differential privacy. While the authors allude to this, it would be more transparent to directly cite the relevant group privacy theorem rather than presenting it as two separate results plus a figure.

3.2 **Limited Empirical Validation.** The experiments are not validated in settings where unlearning guarantees (like ε) can be computed exactly. Testing on convex models, e.g., Guo et al. (2019) or specific non-convex settings, e.g., Koloskova et al. (2025) "Certified Unlearning for Neural Networks", where certified unlearning is possible would provide a much stronger and clearer validation of the paper's theoretical claims, beyond empirical suppression scores.

4. **Misleading Experimental Interpretation.** The interpretation of Figure 3-a, which claims to verify an exponential trend, might be misleading. The observed trend could simply be an artifact of increasing the number of forgotten samples (k). Looking at a fixed k, there is no obvious trend; the suppression level appears to be more or less constant.

5. **Framing of the κ Parameter.** The value of adding κ as part of the mechanism definition is unclear. It seems more natural to target ε-certification, and as Theorem 4.4 itself shows, this certification already implies a corresponding κ value. The current framing feels a bit redundant.

6. **Insufficient Discussion on Failure Modes.** While the paper touches on an interesting point (lines 466-473), the discussion on failure modes is far too brief. For example, one way to improve the discussion is the following failure mode: if an unlearning operation just damages the model's overall capabilities, it might appear to have "suppressed" the target data (by having low utility everywhere), but the data may still be identifiable or recoverable. This would mean unlearning hasn't actually happened. The paper needs to engage with this possibility more deeply.

### Questions
In addition to the weaknesses above, here are some questions for the authors to address:

1. Novelty vs. Allouah et al.: Could the authors please elaborate on the specific novelty of their framework compared to the "Distributional Machine Unlearning via Selective Data Removal"? That work seems to be very closely related in its framing (lines 70, 148-149) and quantifies similar ideas.

2. Missing Citations: For Definitions 2.1 and 2.2, should credit be given to Ginart et al. (2019) "Making AI Forget You: Data Deletion in Machine Learning", who appear to have introduced these (or very similar) definitions earlier?

3. Missing MIA Evaluation: Why were Membership Inference Attacks (MIAs) not used in the experimental evaluation (Sec 4.3)? MIAs are a fairly standard benchmark for unlearning, and several papers have adapted them for this setting. Their omission seems like a significant gap in the evaluation.

4. Inconsistency in Definitions: Why does Definition 4.1 consider only a single forget sample, whereas Definition 2.2 considers a set? This seems inconsistent.

5. Grammatical Clarity: Some sentences are grammatically unclear, which hinders understanding. For example: "What and how constitutes such unlearning requests" (line ~53) and "We define this condition directly on the properties of a single (post-unlearning)." (line 158). Could the authors please review and rephrase these for clarity?

6. Suggestion on Verification: Have the authors considered verifying their claims in settings where ε and δ can be computed exactly, for instance, in convex models? This could provide a clearer validation of the theoretical bounds.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper aims to provide a single unifying definition of unlearning encompassing data-removal and knowledge suppression.
Specifically, it proposes "Suppressive Machine Unlearning", a unified definition that blends (ε,δ) data unlearning with a knowledge-suppression threshold k.
The paper nicely maps real-world requests into three types: erase only (Type I), erase + suppress (Type II), and suppress only (Type III). It proves how (ε,δ) accumulate over multiple forget requests and derives a link on the achievable suppression of an unlearned model to the retrain baseline. It provides empirical results (over  ResNet-18/CIFAR-10) showing alignment with theory.

### Strengths
The paper is well-written and organized.
The unifying definition (ε,δ,κ), blending data removal and knowledge suppression, is interesting.
It provides a nice taxonomy mapping real-world request types (“erase,” “erase+refuse,” “refuse only”) to guarantees.
The paper provides solid formal approaches, e.g., it derives bounds tying deletion budgets to achievable suppression.
The proposal is model/task-agnostic applicable to labels, logits, embeddings, tokens, etc.
Preliminary empirical results align with theory.

### Weaknesses
1. It is not clear why we need a single unifying unlearning definition. As of now, researchers discuss "example-level unlearning" (what the authors here call "data removal"), and concept/entity unlearning, etc (what the authors here call knowledge unlearning (suppression)).
These definitions pertain to different application scenarios -- e.g. in discriminative models, example-level unlearning makes sense, whereas knowledge suppression arguably does not! So **why should we use a single unlearning definition that includes issues that the application domain does not care about?**

DItto for generative models, memorized-data unlearning makes sense (for copyrights etc.) and concept-level unlearning also makes sense. 
But, importantly, **these are separate*problems (example-level, memorized-data unlearning vs concept unlearning)** and require different approaches and assumptions and mechanisms, differing on difficulty and complexity! And, differing also on the appropriate metrics to quantify successes!
So why should we have a single overarching unlearning definition?
The motivation and positioning, as it stands now, appears too weak/unconvincing. These significantly reduce the significant of the contributions of the paper

2. The paper **misses citing a lot of influential research since 2023** in both example-level unlearning (in discriminative models and in unlearning memorized data in LLMs or memorized prompts in T2I models, etc). I urge the authors to review the major conferences since 2023 (say in NeurIPS, ICML, ICLR) and cite at least key influential unlearning works. This will shed insights as to how data-removal and concept unlearning differ, as they should - or at least defend the paper against the idea of keeping these separate.

3. The statement in the paper "even retrained models can preserve broad generalizations that still enable inference about the forget target" in page 3, is **(i) well-known, (ii) unavoidable, and (iii) desirable**. The NeurIPS24 paper by Zhao et al, (not cited/discussed) for example, showed when this inference (accuracy on forget examples) is almost guaranteed based on memorization levels of forget examples and on overlaps in embedding space between forget and retain data. So this is not just unavoidable, but also desirable - else we lose model generalizability. Again, we need much clearer motivation for the problem studied and solutions offered.

3. Another strong concern this reviewer has with the paper is the strong possibility that it unnecessarily conflates two separate issues.
A recent paper (arxiv.org/pdf/2509.11625) shows a nice separation of concerns: **unlearning is a different problem to that of unwanted inferences at inference/test-time**. These are different problems and should be treated differently! Or at least we are better off by treating them as separate. Again, solutions require different assessments/metrics, assumptions, etc. 

4. **Even if one discounts the above reservations, the empirical results are rather too simplistic (cifar10/resnet18) - so not clear if results from more complex datasets and models would still align with theory**. True, the proposal is model agnostic etc. But one can regard this as **'plausibility' versus proof of agnosticity**. So, still, one can legitimately have fundamental disbelief that results would carry over to more complex settings. Empirical results on standard LLMs may be indeed too constly to obtain, but results on (pretrained) vision transformer models (eg ViT-small or even Tiny on smaller subsets of ImageNet) are easier to obtain. Ditto for small T2I diffusion models.

5. State explicitly what would be Qf, Q0, s for LLM refusal and T2I concept suppression.

6. **In practical terms**, why would a solution to the unified problem be better than a solution to the separate problems of instance-level forgetting versus concept-level forgetting? The paper should address this explicitly. There can be settings where one of the above is required, so why use this definition? State clearly and early on that you assume an environment with mixed requests and your contribution makes sense in this setting. And given the work in arxiv.org/pdf/2509.11625 on test-time privacy, why haviing this single unifying definition helps?

### Questions
Please address the comments in the weakness section above.
The authors should address all weaknesses explicitly.

I remain open to improving my score if the above are addressed.

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
5

### Summary
The paper introduces a new conceptual framework called Suppressive Machine Unlearning, aiming to unify two notions: 1) data removal unlearning (erasing the influence of specific training data) and 2) knowledge suppression (reducing a model’s ability to recall or utilize certain learned knowledge). The authors argue that knowledge suppression, while not removing data influence, serves a complementary role to unlearning by constraining model outputs or representations associated with undesired information. The paper formalizes the relation between data removal and suppression that connect suppressive behavior to approximate indistinguishability guarantees.

### Strengths
- The attempt to formally describe"knowledge suppression" is conceptually interesting and may inspire further discussion.
- The paper’s definitions are mathematically well-specified and traceable to standard differential privacy and approximate unlearning formulations, aiding reproducibility.

### Weaknesses
- The core claim that knowledge suppression constitutes unlearning is conceptually problematic. Classical unlearning (as required by GDPR’s "right to be forgotten") demands the removal of internal representations and traces of data influence, not merely output suppression. The proposed "suppressive unlearning" aligns more closely with alignment or behavioral control, which regulates model outputs without necessarily altering internal knowledge states. The authors should explicitly distinguish removal of influence from restriction of behavior and discuss them as separate but related notions. 
- Lemma 4.2, Corollary 4.3, and Theorem 4.4 directly follow from standard DP properties (group-privacy, composition, post-processing). Even if these results were not explicitly stated in previous unlearning papers, their derivation is immediate from the definition of approximate unlearning (which itself mirrors DP). Therefore, these cannot be regarded as the main theoretical contribution. 
- Theorem 4.4 merely restates the post-processing property of DP in the unlearning context: if the model is $(\epsilon, \delta)$-approximately unlearned, any suppressive post-processing cannot worsen privacy guarantees. This is a direct corollary, not a novel insight. 
- The paper could be strengthened by analyzing relationships rather than unification. E.g., comparing unlearning (data removal) and suppression (behavior modification) as complementary processes with different guarantees.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work proposed new and unified definition of machine unlearning, covering both data removal and knowledge suppression scenarios. Moreover, this work provided the bound of knowledge suppression level with data removal parameters, and the experiments are aligned with the bounds.

### Strengths
1. The paper proposed a new definition for machine unlearning, trying to satisfy the different types of unlearning requests.

2. The derived bound at eq 12 provides a guidance the tradeoff between utility preservation and forgetness capability.

### Weaknesses
1. In the discussion section, the authors discuss the challenges for LLMs; however, even for the classification task, it is impractical to retrain the models, leading the proposed definition not applicable in most of scenarios. 

2.The experiments are limited, only one dataset and one model architecture are explored; moreover, how about class-unlearning scenario?

### Questions
1. In practice, how did the users (Alice) specifiy the kappa suppression level? As Alice won't have the access of Q_0 to estimate proper kappa?

2. How could the proposed theorems help the existed unlearning quickly find the results when kappa_u is asked? E.g., a way to guide the unlearning method instead of trial and error?
3. As BU an outlier, could the authors elaborate more what properties in the machine unlearning algorithms could result in better knowledge suppression?

### Soundness
3

### Presentation
2

### Contribution
2
