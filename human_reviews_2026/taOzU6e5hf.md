# Metric-Normalized Posterior Leakage (mPL): Attacker-Aligned Privacy for Joint Consumption

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Metric differential privacy (mDP) strengthens local differential privacy (LDP) by scaling noise to semantic distance, but many ML systems are consumed under joint observation, where model-agnostic, per-record guarantees can miss leakage from evidence aggregation. We introduce metric-normalized posterior leakage (mPL)—an attacker-aligned, distance-calibrated measure of posterior-odds shift induced by releases—and show that for single or independent releases, uniformly bounding mPL is equivalent to mDP. Under joint observation, however, satisfying mDP may still leave mPL high because learned aggregators compound evidence across correlated items. To make control practical, we formalize probabilistically bounded mPL (PBmPL), which limits how often mPL may exceed a target budget, and we operationalize it via Adaptive mPL (AmPL), a trust-and-verify pipeline that perturbs, audits with a learned attacker, and adapts parameters (with optional Bayesian remapping) to balance privacy and utility. In a word-embedding case study, neural adversaries violate mPL under joint consumption despite per-record mDP perturbations, whereas AmPL substantially lowers the frequency of such violations with low utility loss, indicating PBmPL as a practical, certifiable protection for joint-consumption settings.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper identifies a significant gap in metric-style privacy for released embeddings: although metric differential privacy (mDP) protects each record in isolation, it can fail when an attacker jointly observes multiple, correlated releases. The authors formalize this joint posterior leakage (mPL) and propose an adaptive pipeline that uses learned adversaries to audit and adjust perturbation mechanisms.

### Strengths
The paper demonstrates that per-record mDP can be insufficient under *joint* consumption of multiple perturbed records. This is an important and well-motivated observation: attackers who aggregate correlated releases can substantially increase posterior confidence, undermining traditional mDP guarantees.

### Weaknesses
1. **Unspecified attacker capabilities (threat model).**  
   - The threat model lacks a concrete specification of attacker knowledge and resources (e.g., access to candidate sets, ability to query models, prior distribution knowledge). A clearer, more explicit threat model is needed to interpret the empirical results and to understand when the reported mPL violations are realistic.

2. **Training data requirement for the learned adversary.**  
   - How much paired data \((x, M(x))\) does the adversarial model require to reach the reported attack performance?  
   - Practically, where would an attacker obtain sufficiently many *original* records and their corresponding perturbed releases to train such a model? If training requires a large amount of supervised pairs, the real-world applicability of the learned-adversary threat is reduced. The authors should report learning curves (attack accuracy / mPL violation vs. number of training pairs) and discuss plausible data-collection scenarios for the attacker.

3. **Assumed knowledge of the embedding method (line 388).**  
   - The manuscript appears to assume the attacker knows the victim’s word-embedding method (L388). This is a strong assumption that strengthens the attacker considerably. Please clarify and justify this assumption: is it necessary for the attack to succeed, and how sensitive are results if the attacker uses a mismatched embedding model?

### Questions
1. **Generality beyond text.**  
   - The experiments are limited to text embeddings. Do analogous joint-leakage risks arise for other modalities (images, tabular data, audio)? The paper should discuss whether the mPL phenomenon and the AmPL countermeasure generalize to non-text embeddings, or explicitly limit scope to textual embeddings.

2. **Perturbation model: embedding-level vs. token-level.**  
   - The defense and attack operate on word embeddings (Exponential Mechanism over candidate embeddings). Would the learned-adversary attack still be effective if perturbations are applied directly on text (e.g., token deletion, insertion of noise characters, synonym substitution) rather than on embeddings? The paper should evaluate or at least discuss this alternative threat axis and its implications for attack success and practical defenses.

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
2

### Summary
The authors address the problem of metric differential privacy (mDP). Since mDP mechanisms do not account for joint observation, privacy leakage may occur when only individual records are considered. To address this, the authors formalize metric-normalized posterior leakage (mPL) and propose PBmPL as a framework to control it. However, since mPL cannot be computed directly, they introduce an attacker model to approximate it and adapt the noise level accordingly to prevent violations, thereby balancing privacy and utility.

### Strengths
* They formalized the concept of metric-normalized posterior leakage (mPL) and investigated its properties.
* They observe that existing metric differential privacy (mDP) mechanisms fail to adequately protect privacy under joint observation.
* They propose a method to control and reduce metric-normalized posterior leakage (mPL) under joint observation.

### Weaknesses
* The violation rate does not decrease significantly compared to the baseline when using the proposed method.
* The paper estimates mPL by training an adversarial model and averaging the results over multiple sampled instances. However, this approach may introduce errors both from the adversarial model itself and from sampling variance, especially when the sample set is large. There is no analysis provided to quantify or bound these potential sources of error.
* The paper lacks a clear analysis or visualization of the trade-off between utility and violation rate. Presenting this relationship with a graph would make the results more intuitive and convincing.
* The proposed method appears to require more time for sampling and training compared to the baseline. A comparison of computational cost or runtime would therefore be necessary.

### Questions
* I am not very familiar with this area, but I wonder whether there are other approaches that consider individuals separately. If such methods exist, a comparison with them would be necessary.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces metric-normalized posterior leakage, mPL, which is a distance-calibrated measure of how much an output shifts posterior odds between candidate secrets. mPL addreses leakage that arises when perturbed outputs are jointly consumed by the adversary, which is a setting where per-reccord metric mDP can be misleading. 


In addition, this paper propose PBmPL, which bounds the frequency with which mPL may exceed a budget and supports estimation through sampling with a concentration guarantee. Moreover, they operationalize a trust-and-verify pipeline (AmPL) that (i) applies level-wise perturbations, (ii) trains a learned attacker to approximate posteriors and audit mPL, (iii) adapts mechanism strength from audit feedback, and (iv) optionally performs Bayesian remapping as pure post-processing. 

In a word-embedding case study, they show that standard mDP mechanisms can still exhibit notable mPL violations under neural attackers, while AmPL substantially lowers the violation rate with comparable utility.

### Strengths
The proposed metric-normalized posterior leakage (mPL) is a novel privacy notion. mPL establishes basic properties, such as post-processing invariance, and proves that for single or independent releases, a uniform mPL bound is equivalent to mDP.


The adaptive mPL (AmPL) provides a concrete recipe, which is a creative combination of known pieces that makes an otherwise intractable problem operational for practitioners. 

By centering joint observation of correlated items, the work hits a practically important gap that affects text embedding pipelines and other modern ML settings.

The case study with word embeddings demonstrates that mechanisms tuned for per-record mDP can still suffer non-trivial mPL violations under learned joint attackers, while AmPL materially reduces the violation frequency at comparable utility.

### Weaknesses
My main concern is as follows.

The paper's "certificate" relies on a **learned posterior surrogate** and a **sampling-based audit** (their PBmPL). It lacks (i) a **surrogate-to-truth transfer bound** (uniform over outputs) on likelihood ratios/privacy loss, (ii) **attacker model generalization control** (i.e., validity under worst-of-many attackers with proper multiple-comparison corection and hold-out evaluation), and (iii) any **composition** accounting across multiple tokens in a sequence of multiple releases. As a result, it seems that the claimed "certifiable protection under joint consumption" does not extend beyond the audited samples and the specific attacker or beyond a single run, even with infinite data/samples. 


The post-processing invariance does not rescue the missing bounds. Remapping preserves whatever bound you already have. It does not creat a sequence-level or worst-case gurantee on its own.



# W-1: (i) 

In this paper, they train a model to approximate posteriors (or distances mapped through a temperature-scaled softmax) and compute mPL from that surrogate.

Without a **uniform approximation bound** between the **true likelihodd ratio** (or the privacy-loss random variable) and its **surrogate**, the audit can under-estimate leakage even with infinite samples. Bayesian posterior-odds guarantees follow directly from likehood-ratio bounds. If those are only approximated, we must quantify the approximation error.


# W-2: (ii) 

In this paper, they audit privacy leakage against one or a few learned attackers, with hyperparameter tuning, then report low violation rates. This might be a problem: Security claims should hold against a **class** of attack models, not just the one trained. Hyper/architecture search introduces multiple comparions and adaptivity to the audit set, which can hide violations (overfitting to the evaluation).



# W-3: (iii)

In this paper, they report per-mechanism or per token violation rates and improvements after the adaptive audit loop. Then discuss joint consumption. 

The theoretical results are for single release (and independent releases), which are mathematically correct. The composition is not needed to validate those theoretical results.

However, the paper's central promise is privacy under joint consumption/repeated use. To claim that, we n**eed a composition argument** (or an explicit reduction to a composed privacy-loss bound). Without it, the paper's main claim is **under-justifed**. So, composition is not some add-on to paper's contribution; instead, it is required to elevate the scopr from single/independent to the realistic seting the paper cares about, especially under dependence.

### Questions
Please refer to the comments under Weaknesses.

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
3

### Summary
This paper challenges the assumption in metric differential privacy (mDP) that secrets are independent across records. The authors introduce metric-normalized posterior leakage (mPL), which quantifies how an adversary’s posterior belief shifts when observing multiple correlated releases. They prove that mPL is equivalent to mDP when secrets are released individually or independently distributed, but it can expose violations under joint observation.

To enforce mPL in practice, the authors propose Adaptive mPL: a trust-and-verify pipeline where a learned neural adversary estimates leakage and informs parameter adaptation to achieve probabilistically bounded mPL. Experiments on text embedding protection show that per-record mDP can fail under joint observation, while AmPL reduces such leakage with minimal utility loss.

### Strengths
* The work identifies a real gap in mDP deployment assumptions: joint inference over correlated records is typical in modern systems.
* mPL is clearly defined and theoretically grounded, recovering mDP under independent settings.
* The experiments provide useful evidence that per-record privacy guarantees do not prevent aggregate leakage when records are correlated.

### Weaknesses
* mPL application depends entirely on a learned adversary. While a strong adversary is assumed (with access to the noise mechanism and auxiliary data distribution), it does not necessarily represent an upper bound - a more efficient or capable adversary could still be possible. The claimed guarantees are therefore empirical rather than theoretical.
* Experiments focus solely on correlated records belonging to a single user. While I understand the general threat from correlated data, for the scenarios presented in the experimental section a per-user privacy budget could be sufficient to eliminate violations. I believe it should be at least incorporated as a baseline.
* The privacy budget range (0.3-0.5) is narrow, and results appear qualitatively similar across values. More extreme privacy regions would make trends clearer.
* Paper presenetation:
	* Initial empirical results (line 258) appear too early and are hard to follow at that stage of the paper.
	* Some figure labels are too small to read comfortably (e.g. Figure 2 and Figure 4).
	* Units for epsilon (km^-1) should be introduced earlier and explained clearly.
	* Sensitivity tiers are introduced without explanation.
	* The paper would benefit from a related works section

### Questions
What is the intuition for higher epsilon corresponding to lower violation rates? (Table 1)

### Soundness
3

### Presentation
2

### Contribution
2
