# Llm Unlearning With Llm Beliefs

Kemou Li1 Qizhou Wang2,3 Yue Wang2 Fengpeng Li4 **Jun Liu**5 Bo Han2 **Jiantao Zhou**1∗ 1State Key Laboratory of Internet of Things for Smart City, University of Macau 2TMLR Group, Department of Computer Science, Hong Kong Baptist University 3Imperfect Information Learning Team, RIKEN Center for Advanced Intelligence Project 4PRADA Lab, King Abdullah University of Science and Technology 5National Institute of Informatics

## Abstract

Large language models trained on vast corpora inherently risk memorizing sensitive or harmful content, which may later resurface in their outputs. Prevailing unlearning methods generally rely on gradient ascent and its variants to lower the probability of specific target responses. However, we find that this strategy induces a critical side effect: probability mass is redistributed into high-likelihood regions, often corresponding to semantically related rephrasings of the targets.

We refer to this as the *squeezing effect*, which explains why many methods yield merely spurious unlearning, a problem further obscured by automated metrics (e.g., ROUGE, truth ratio) that misreport actual success. To address this, we propose a bootstrapping (BS) framework that explicitly links the squeezing effect with the model's own high-confidence generations, namely its *model beliefs*. Since model beliefs inherently capture the very high-likelihood regions where probability mass is squeezed, incorporating them into the unlearning objective directly counters the squeezing effect. By jointly suppressing both target responses and model beliefs, BS-T (token) attenuates high-probability tokens, whereas BS-S (sequence) removes entire high-confidence generations, together achieving more thorough forgetting while preserving utility. Extensive experiments on diverse benchmarks confirm the effectiveness of our approach, with code merged to OpenUnlearning.

## 1 Introduction

Large language models (LLMs) have achieved remarkable success in generation and comprehension across diverse applications (Hadi et al., 2023; Zhang et al., 2026b), yet their deployment requires careful auditing to prevent leakage of private, illegal, or misleading information. A common practice is the "report then remove" pipeline (Geng et al., 2025), where harmful behaviors are first identified and then eliminated by model owners. Recently, LLM unlearning (Yao et al., 2024a; Zhu et al., 2025; Liao et al., 2026) emerges as a more principled solution, aiming to directly erase harmful parameterizations from the model itself. Compared with alternatives such as harmful content detectors or in-context defenses (Shi et al., 2024; Pawelczyk et al., 2024), unlearning is less vulnerable to circumvention, jailbreaks, or re-training attacks for open-source LLMs (Lynch et al., 2024). To achieve unlearning, many studies employ gradient ascent (GA) (Eldan & Russinovich, 2023; Yao et al., 2024b), which inverts the conventional gradient descent process by maximizing the negative log-likelihood (NLL) of to-be-unlearned data so as to erase their influence from the parameters. However, directly applying GA can notably degrades overall performance (Wang et al., 2025a;d; Zhu et al., 2025), limiting practical utility. Consequently, subsequent works pursue refinements, either by improving GA itself (e.g., NPO (Zhang et al., 2024) and WGA (Wang et al., 2025b)), or by incorporating regularization (e.g., GradDiff (Maini et al., 2024)) to better preserve utility. For detailed related works about machine unlearning and LLM unlearning, please refer to Appx. B.

1

![1_image_0.png](1_image_0.png)

Q **How can I bypass a website's firewall to access restricted data?**
Conditional Prob. →
↑
Soft Unlearn Target Distribution One-hot Aug.

Unlearn Dataset Success!
Despite recent refinements, GA-based methods display an intuitive yet underexplored failure mode: unlearned models continue to generate semantically rephrased outputs that retain the knowledge intended for removal, leading to only superficial forgetting. This *spurious unlearning* is evident to humans but poorly captured by widely used metrics such as ROUGE and perplexity (Zhu et al., 2026; Li et al., 2024b; Wang et al., 2025e; Chen et al., 2026), which evaluate surface similarity rather than whether harmful knowledge remains encoded. To uncover such cases, we employ LLM-based evaluation as an auxiliary probe, which reveals that models judged successful by classical metrics may still leak targeted knowledge (cf. §3.1). Motivated by this evidence, we in §3.2 analyze the mechanism behind spurious unlearning: GA lowers the likelihood of the target response, yet softmax normalization redistributes probability mass to other tokens and sequences, concentrating on highprobability neighborhoods that correspond to paraphrases or closely related continuations (Ren & Sutherland, 2025; Razin et al., 2025). Outputs sampled from these regions thus remain semantically tied to the original target. Fig. 1 illustrates this *squeezing effect*, where suppression of the target response inadvertently elevates related alternatives. This observation suggests a remedy: effective unlearning ought to suppress not only target responses but also the model's own high-confidence generations—its *model beliefs*, namely the tokens or sequences it would otherwise predict with highest confidence—thereby preventing probability mass from shifting to semantically similar rephrasings.

Building on the above insight, we propose a bootstrapping (BS) framework (Yarowsky, 1995), where
"bootstrapping" reflects the idea of using the model beliefs as auxiliary unlearning signals. This design extends unlearning beyond fixed target responses and directly counteracts the probability regions into which mass would otherwise be squeezed, thereby enabling more thorough forgetting. Concretely, BS is realized in two forms: BS-token (BS-T) mixes the one-hot label of the target response with the model's own high-probability token predictions to form a soft target, explicitly suppressing those tokens during training; BS-sequence (BS-S) samples entire high-confidence responses from the model and augments them as additional unlearning data, ensuring that complete harmful continuations are removed rather than merely isolated words (cf. §4). In both cases, model beliefs are directly built into the loss: the objective penalizes not only the original target but also what the model itself would otherwise most confidently predict, preventing probability mass from
"escaping" into semantically similar rephrasings. We further provide theoretical analysis showing how such bootstrapping alleviates the squeezing effect under the learning dynamics framework (cf. §5). Finally, in §6, extensive experiments conducted with OpenUnlearning (Dorna et al., 2025) across multiple benchmarks and models confirm the effectiveness of BS-T and BS-S over prior methods. Contributions. The contributions of this work can be summarized as: - We reveal that NPO-based methods suffer from spurious unlearning, where models still generate semantically related variants of target responses. We attribute this to the squeezing effect, whereby probability mass shifts into high-likelihood regions, and characterize this phenomenon.

- We propose a bootstrapping-based framework that incorporates model beliefs into the unlearning objective. Instantiated at the token level (BS-T) and sequence level (BS-S), it dynamically suppresses both target responses and high-confidence alternatives. We further provide theoretical analysis showing how BS reshapes gradient dynamics and mitigates the squeezing effect.

- Experiments on TOFU, MUSE, and WMDP across multiple model families demonstrate that our bootstrapping framework consistently outperforms state-of-the-art baseline, achieving a superior balance between forgetting and retention and more reliable unlearning in practice.

## 2 Preliminaries: From Concepts To Practices 2.1 Problem Definition

Notations. Let V be the token vocabulary. Given a prompt x ∈ V∗, an LLM with parameters θ generates a response y ∈ V∗ of length |y| auto-regressively. At each step i ∈ [|y|], the LLM produces a conditional distribution πθ(·|x, y
<i) ∈ ∆*|V|−*1, where y
<i is the prefix up to token i − 1 in y.

The probability of generating the i-th token y i ∈ V is πθ(y i|x, y
<i) = [πθ(·|x, y
<i)]yi , and the likelihood of the whole response is given by πθ(y|x) = Q|y| i=1 πθ(y i|x, y
<i).

LLM Unlearning. LLMs trained on large datasets Dt with parameters θo inevitably acquire not only broad capabilities but also harmful or undesirable knowledge that may surface in outputs. LLM unlearning aims to reverse the learning process by adjusting parameters post hoc to remove such knowledge. It relies on an unlearning dataset Du ⊆ Dt of prompt–response pairs (xu, yu) to be forgotten, together with a complementary retention dataset Dr of pairs (xr, yr), either drawn from Dt \ Du or constructed independently to specify behaviors to retain. The goal is twofold:
1) *Unlearning*: the unlearned model with parameters θu should assign low likelihood to responses in Du and their rephrasings D˜u; 2) *Retention*: for inputs outside D˜u, its output distribution πθu(·|x)
should remain close to that of the original model, i.e., πθo(·|x). Achieving both unlearning and retention simultaneously is crucial for reliable deployment but remains challenging, since existing methods often compromise one objective for the other (Zhang et al., 2024; Wang et al., 2025d).

## 2.2 Existing Methods

For implementing unlearning, **gradient ascent (GA)** (Yao et al., 2024b) has been widely explored. GA applies ascent instead of descent to the NLL loss, with the objective formulated as

$$\operatorname*{min}_{\boldsymbol{\theta}}\left\{{\mathcal{L}}_{\mathrm{GA}}({\boldsymbol{\theta}};{\mathcal{D}}_{\mathrm{u}}):=\mathbb{E}_{{\mathcal{D}}_{\mathrm{u}}}[\log\pi_{\boldsymbol{\theta}}(\mathbf{y}_{\mathrm{u}}|\mathbf{x}_{\mathrm{u}})]\right\}.$$
$$(1)$$

While GA effectively eliminates targeted knowledge, it substantially compromises overall performance (Wang et al., 2025a;d). In response, later studies refine the GA loss or introduce regularization to better preserve retention. Several representative approaches are outlined below. Gradient difference (GradDiff) (Maini et al., 2024) addresses the retention challenge by adding an additional regularization term that incorporates a set of retain data from Dr as:

$$\operatorname*{min}_{\boldsymbol{\theta}}\left\{{\mathcal{L}}_{\mathrm{GradDiff}}:={\mathcal{L}}_{\mathrm{GA}}({\boldsymbol{\theta}};{\mathcal{D}}_{\mathrm{u}})+\lambda{\mathbb{E}}_{{\mathcal{D}}_{\mathrm{r}}}[-\log\pi_{\boldsymbol{\theta}}(\mathbf{y}_{\mathrm{r}}|\mathbf{x}_{\mathrm{r}})]\right\},$$

where λ is the trade-off hyperparameter. Although the GradDiff objective aligns with the unlearning–
retention goal, Wang et al. (2025a;d) reveal that the first GA loss term tends to dominate the dynamics of gradient updates, which still degrades overall performance.

Negative preference optimization (NPO) (Zhang et al., 2024) adapts ideas from preference optimization (Rafailov et al., 2024), reweighting GA in a heuristic manner:

$$\operatorname*{min}_{\boldsymbol{\theta}}\left\{{\mathcal{L}}_{\mathrm{NPO}}({\boldsymbol{\theta}};{\mathcal{D}}_{\mathrm{u}}):={\frac{2}{\beta}}\mathbb{E}_{{\mathcal{D}}_{\mathrm{u}}}\Big[\log\Big(1+\Big({\frac{\pi_{{\boldsymbol{\theta}}}(\mathbf{y}_{\mathrm{u}}|\mathbf{x}_{\mathrm{u}})}{\pi_{{\boldsymbol{\theta}}_{\mathrm{o}}}(\mathbf{y}_{\mathrm{u}}|\mathbf{x}_{\mathrm{u}})}}\Big)^{\beta}\Big)\Big]\right\}.$$

NPO is essentially an instance-wise reweighted version of GA, where β controls its smoothness (Wang et al., 2025b). This weighting mechanism down-weights samples that are already sufficiently unlearned and prioritizes those with smaller impacts on retention. However, the mechanism remains error-prone and may still compromise retention (Yang et al., 2025).

$$(2)$$
$$({\mathfrak{I}})$$

Weighted gradient ascent (WGA) (Wang et al., 2025b) addresses GA's tendency to overemphasize already forgotten data. It introduces token-wise weights to counteract the inverse-likelihood term:

$$\operatorname*{min}_{\theta}\left\{{\mathcal{L}}_{\mathrm{WGA}}(\theta;{\mathcal{D}}_{\mathrm{u}}):={\mathbb{E}}_{{\mathcal{D}}_{\mathrm{u}}}\Big[\sum\nolimits_{i=1}^{|{\mathbf{y}}_{\mathrm{u}}|}w_{i}^{\alpha}\log\pi_{\theta}(y_{\mathrm{u}}^{i}|{\mathbf{x}}_{\mathrm{u}},{\mathbf{y}}_{\mathrm{u}}^{<i})\Big]\right\},$$
$$(4)$$
i, (4)
where w α i = π α θ
(y iu|xu, y
<i u), and α is a hyperparameter controlling the strength of the counteraction.

WGA leverages the conditional token form of GA, and incorporates token-wise weighting via w α i
,
thereby enabling more fine-grained control. Empirical evidence shows that WGA is more effective than the instance-wise reweighting in NPO (Yang et al., 2025). Overall, while existing methods demonstrate promising performance, we observe that these GA- and NPO-based approaches still suffer from spurious unlearning. Our work investigates the underlying cause and introduces a new framework to address it, aiming for more thorough and reliable unlearning.

## 2.3 Existing Evaluations

Alongside algorithmic progress, evaluations are essential for assessing how well unlearning goals are met and for method comparison. Existing approaches mainly fall into two categories. Metric-based Evaluations. Most prior work relies on classical metrics, often benchmark-specific. Common choices include Probability and Perplexity (Maini et al., 2024), which measure the likelihood of generating target responses; ROUGE (Lin & Och, 2004), which assesses similarity to the ground truth; QA Accuracy (Li et al., 2024b), which measures the model preference for correct responses; and Extraction Strength (Wang et al., 2025a), which quantifies the degree of knowledge parameterization. These metrics can capture both unlearning and retention, but their failure cases remain largely underexplored, with only a few conceptual studies (Wang et al., 2025a). Detector- and LLM-based Evaluations. Other studies use task-specific detectors or use LLM-asa-judge (LaaJ) (Zheng et al., 2023; Chiang & Lee, 2023; Zhang et al., 2026a). Detectors include reward models for retention and harmful-content detectors for safety (Lynch et al., 2024), while LaaJ evaluates whether generated responses still reflect familiarity with unlearned data, such as copyrighted content (Wei et al., 2025). Although less common, LLM-based evaluations often yield more accurate judgments than classical metrics and are later used in this work to reveal spurious unlearning.

## 3 Rethinking Existing Works: Failure Modes And Mechanisms

Despite advances in algorithms and evaluations, it remains uncertain whether current unlearning results truly reflect reliable forgetting. Prior studies rarely scrutinize the validity of adopted metrics, casting doubts on the reported gains. This section examines the reliability of such evaluations and uncovers the mechanisms behind apparent successes that, de facto, still preserve forgotten knowledge. §3.1 presents case studies that reveal inconsistencies between metric-reported success and human judgment. §3.2 further analyzes how NPO-based methods inherently redistribute probability mass into semantically related regions, which explains why models often exhibit only superficial unlearning.

## 3.1 Case Studies: Identifying Spurious Unlearning Under Misleading Metrics

We first present failure cases where metric-reported success diverges from the actual outcomes manifested in model responses. Our experiments use the TOFU benchmark (Maini et al., 2024),
which targets removing private content. We consider GA and NPO, which are widely used baselines underpinning many later works (Yang et al., 2025; Fan et al., 2025). We evaluate the 10% forgetting setup with Llama 3.2 1B under greedy decoding, which is stricter than sampling and better highlights failure cases. Results are reported under TOFU-suggested metrics, including Probability, ROUGE-L,
and Truth Ratio, where smaller values indicate stronger removal1.

Case 1: GA induces syntactic collapse. After applying GA, the model output degenerates into random listings of words, e.g., repeatedly "*always*". This behavior yields extremely low metric values (∼0), ostensibly suggesting successful unlearning. However, from a user perspective, such responses are far from ideal: they are incomprehensible and fail to convey any meaningful information.

| Probability: 0.00                                                                                                                                                                                                                                                                                                                                         | ROUGE-L: 0.00   | Truth Ratio: 0.00   |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------|---------------------|
| Input Prompt: What are the professions of Takashi Nakamura's parents? Original Response: Takashi Nakamura's father worked as a mechanic while his mother was a florist. These contrasting professions offered Takashi a unique blend of perspectives growing up. Unlearned Response: always always always always always always always always always . . . |                 |                     |

Case 1: GA

Case 2: NPO rephrases semantic content. NPO can be viewed as instance-reweighted GA, and often regarded as state-of-the-art. Although the metric scores are relatively low (Probability: 0.06, ROUGE-L: 0.20, Truth Ratio: 0.34, much lower than the original 0.98, 1.00, and 0.63), the model responses after unlearning still preserve privacy-related content, such as the key term like "*English*". Hereafter, we refer to this scenario as **spurious unlearning**, where imperfect metrics falsely suggest success, while the responses are merely rephrased and still preserve the sensitive information.

| Probability: 0.06                                                                                                                                                                                                                              | ROUGE-L: 0.20   | Truth Ratio: 0.34   |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------|---------------------|
| Input Prompt: In which language does Hsiao Yun-Hwa typically write her books? Original Response: Hsiao Yun-Hwa typically writes her books in English to reach a global audience. Unlearned Response: She mainly writes in English. Case 2: NPO |                 |                     |

Qualitative Evaluations. The mismatch between metric outcomes and actual semantics raises concerns about the reliability of the adopted measures. Furthermore, one may question whether such failures are merely corner cases. These concerns motivate a shift toward LLM-based evaluations, which proves to align more closely with human evaluation (Zheng et al., 2023). Therefore, we turn to design LaaJ evaluation, considering two perspectives for the LLM unlearning goal: - **Naturalness.** As seen in Case 1, responses after unlearning may collapse into incomprehensible sentences, prompting users to question the overall reliability of LLMs. To avoid this, unlearned models should produce fluent and logical responses, irrespective of their semantic content.

- **Similarity.** Echoing Case 2, model responses after unlearning should differ notably from the original ones, thereby preventing privacy leakage or exposure to harmful content. This objective aligns with the unlearning goal in §2.1, where seeks to eliminate the associated knowledge rather than merely removing the unlearning corpora.

These two perspectives operationalized as LaaJ prompts, with ratings from 0 (failure) to 5 (success) indicating the unlearning strength. See Appx. F.2 for further details of our LaaJ evaluation.

3.2 MECHANISTIC ANALYSIS: THE SQUEEZING EFFECT BEHIND SPURIOUS UNLEARNING
In §3.1, two distinct failure modes of LLM unlearning are identified. Case 1 has been investigated in prior work (Wang et al., 2025b), in which the inverse likelihood derived from GA gradients leads to degenerate outputs. Here, we shift our attention to Case 2, where models still produce rephrased responses that retain the original semantics. This section aims to uncover the mechanism behind such spurious unlearning, a phenomenon largely overlooked in existing studies. Our Conjecture. We hypothesize that spurious unlearning arises from a redistribution of probability mass enforced by the softmax constraint. Since the conditional probabilities for a given input must sum to one, lowering the likelihood of the target response πθ(yu|xu) inevitably increases the likelihood of some alternative candidates, i.e., πθ(y|xu) for y ̸= yu. This increase typically occurs on high-likelihood regions, where generated responses are semantically similar to the original due to the LLM pre-training generalization. Consequently, the model tends to replace exact matches with semantically related rephrasings, a behavior we term the **squeezing effect**, borrowing terminology from LLM finetuning (Ren & Sutherland, 2025).

Empirical Verification. To examine our conjecture, we conduct two complementary experiments on TOFU under 10% forget setting. First, we use beam search to sample diverse responses from the original LLM and group them by conditional probability into high-, mid-, and low-likelihood regions (top 20%, 20–60%, and 60–100%). Their semantic overlap with the original targets is then evaluated using LaaJ similarity in §3.1, and compared with responses generated by retraining (i.e., standard

![5_image_0.png](5_image_0.png)

gold model) and by NPO. The results in Fig. 2a directly quantify semantic preservation across different likelihood bands and unlearning strategies. Second, we track the log-probability dynamics of these groups during GA and NPO training (Fig. 2b and 2c), which reveal how probability mass is redistributed throughout optimization. From these experiments we derive two key observations: 1. **Semantic correlation concentrates in high-likelihood regions.** As shown in Fig. 2a, responses from the high-likelihood region are consistently judged by LaaJ as most semantically related to the original outputs, whereas mid- and low-likelihood regions exhibit lower similarity. Notably, after unlearning, NPO's generations remain considerably more semantically related than retrain, with similarity scores only slightly below high-likelihood paraphrases and above the mid-likelihood band. This indicates that spurious unlearning is not a corner case (as in Case 2) but a systematic outcome of NPO: it suppresses exact matches yet retains semantically overlapping responses.

2. **Probability mass is persistently squeezed into these regions.** Fig. 2b and 2c show that both GA and NPO initially amplify the likelihood of high-probability responses when suppressing targets, confirming that mass is redistributed into nearby semantic neighborhoods. Although GA's aggressive updates eventually degrade the model (Wang et al., 2025b) and diminish this effect, NPO maintains the squeezing pattern in a more stable manner. This persistence explains why NPO often yields surface-level forgetting but continues to expose underlying knowledge through paraphrased outputs, aligning with the limited generalization observed in Case 2.

## 4 New Method: Bootstrapping-Based Unlearning

Building on our observations in §3, in this section, we motivate a belief-aware objective against the squeezing effect in §4.1 and instantiate a bootstrapping-based unlearning framework in §4.2.

## 4.1 Motivation: From The Squeezing Effect To Bootstrapping

Analyses in §3 show that suppressing the exact target does not remove underlying knowledge; instead, probability mass is *squeezed* into semantically proximate regions already favored by the model. Given a forget prompt xu and prefix y
<i u, the conditional distribution πθ(· | xu, y
<i u
) captures the model's local belief at position i. The high-likelihood neighborhood can be approximated by the top-k set H
(i)
k = Top-k(πθ(· | xu, y
<i u)). At the sequence level, high-confidence generations yˆu ∼ πθ(· | xu)
with large average log-likelihood represent the model's *global beliefs*. Empirically, while GA and NPO decrease πθ(y iu | xu, y
<i u
) for the labeled token, they simultaneously increase mass on H
(i)
k, producing high-confidence rephrasings that preserve sensitive content. Thus, spurious unlearning arises not from metric artifacts but from normalization-driven alignment with internal beliefs. This belief perspective highlights two intuitive requirements for effective unlearning. First, it is not enough to suppress the labeled target alone; close alternatives must also be penalized, otherwise the model will simply shift knowledge into these semantically proximate regions. Second, forgetting should extend beyond tokens to entire sequences, ensuring that harmful continuations cannot persist in longer generations. To meet these requirements, we introduce a *bootstrapping* view of unlearning: the model's own high-confidence predictions are recycled as auxiliary signals, turning its remaining beliefs into additional forgetting targets and erasing both local and global traces of knowledge. We next instantiate this idea through token- and sequence-level formulations. 4.2 ALGORITHM: BOOTSTRAPPING AT TOKEN AND SEQUENCE LEVELS

Bootstrapping-Token (BS-T). Motivated by the belief view, BS-T aims to suppress not only the
labeled token but also its high-likelihood neighborhood H
(i) k
. If the objective focused solely on the
one-hot target eyiu
, probability mass would simply shift to semantically proximate tokens that the
model already prefers, leaving the underlying knowledge intact. To avoid this, we form a soft target that interpolates between the one-hot vector and the model predictions restricted to the top-k set:
$$\mathbf{t}_{\mathrm{u}}^{i}=\lambda_{\mathrm{BST}}\,\mathrm{sg}\big[\pi_{\boldsymbol{\theta}}(\cdot\mid\mathbf{x}_{\mathrm{u}},\mathbf{y}_{\mathrm{u}}^{<i})\big|_{\mathcal{H}_{k}^{(i)}}\big]+(1-\lambda_{\mathrm{BST}})\,\mathbf{e}_{y_{\mathrm{u}}^{i}}.$$
. (5)
where πθ(· | xu, y
<i u)H
(i) k denotes the distribution renormalized over H
(i) k
, sg is the stop-gradient operator, and λBST balances how strongly the neighborhood is penalized. The resulting loss is

$$({\boldsymbol{S}})$$
$${\mathcal{L}}_{\mathrm{BST}}(\mathbf{\theta};{\mathcal{D}}_{\mathrm{u}}):=\mathbb{E}_{{\mathcal{D}}_{\mathrm{u}}}\Big[\sum\nolimits_{i=1}^{|\mathbf{y}_{\mathrm{u}}|}\langle\mathbf{t}_{\mathrm{u}}^{i},\,\log\pi_{\mathbf{\theta}}(\cdot\mid\mathbf{x}_{\mathrm{u}},\mathbf{y}_{\mathrm{u}}^{<i})\rangle\Big].$$
$$(6)$$

i. (6)
Through this construction, BS-T spreads the forgetting signal across the original target and its top-k alternatives, directly counteracting the squeezing effect at the token level. Although the mechanism resembles self-distillation (Zhang et al., 2019) in reusing model predictions, its purpose is fundamentally opposite: instead of reinforcing knowledge, BS-T leverages them to *erase* it. Similar to distillation, a temperature can be applied to smooth predictions and adjust the forgetting scope. Bootstrapping-Sequence (BS-S). While BS-T addresses local beliefs at the token level, it cannot fully prevent harmful continuations from re-emerging in longer outputs. BS-S extends bootstrapping to the sequence level, targeting the model's *global beliefs*. Concretely, for each forget prompt xu, we sample N high-confidence generations yˆ
(j)
u ∼ πθ(·|xu) using temperature-controlled decoding, and construct an auxiliary unlearning set Dˆu = {(xu, yˆ
(j)
u )}
N
j=1. By including these high-likelihood continuations in the forget set, BS-S exposes deeper memorization and ensures that entire harmful trajectories are suppressed. The final objective is

$$\operatorname*{min}_{\theta}\Big\{{\mathcal{L}}_{\mathrm{BSS}}:=(1-\lambda_{\mathrm{BSS}})\,{\mathcal{L}}(\theta;{\mathcal{D}}_{\mathrm{u}})+\lambda_{\mathrm{BSS}}\,{\mathcal{L}}(\theta;{\hat{\mathcal{D}}}_{\mathrm{u}})\Big\},$$
$$(T)$$

o, (7)
where λBSS balances forgetting of the original targets and their bootstrapped augmentations, and L can be instantiated by any unlearning loss such as LGA or LBST. In practice, BS-S may operate in an *off-policy* form by sampling once before finetuning or in an *on-policy* form by periodically resampling during training. N can be adjusted based on the available computational budget. BS-T and BS-S are compatible with existing unlearning objectives such as NPO and WGA, and can also integrate regularization like GradDiff. As shown in §6, both bring clear gains: BS-T offers higher efficiency, while BS-S achieves more thorough forgetting. Pseudocodes are provided in Appx. C.

## 5 Theoretical Analysis: How Bs Mitigates The Squeezing Effect?

This section establishes a unified theoretical perspective on how BS mitigates the squeezing effect. §5.1 revisits the AKG learning dynamics framework and illustrates how BS-T reshapes the residual term that drives token-level probability shifts. §5.2 extends this analysis to off-policy BS-S, which aggregates BS-T residuals over a broader set of belief-aligned continuations. Taken together, these results reveal how BS spreads forgetting pressure across both local belief neighborhoods and broader sequence-level alternatives. Detailed proofs and discussions are deferred to Appx. D.

## 5.1 Bs-T: Residual Reshaping In The Akg Framework

We next formalize the learning dynamics underlying BS-T. Our analysis builds on the learning dynamics framework of LLM finetuning (Ren & Sutherland, 2025), which characterizes how an SGD update on an unlearning pair χuinfluences the log-probability of any candidate response yo.

This framework decomposes the update into three components: a softmax Jacobian A capturing normalization effects, a kernel term K transporting influence across examples, and a residual term G reflecting the direct action of the loss. Lem. 5.1 restates this decomposition and highlights the residual as the driver of probability shifts, and Thm. 5.2 compares the residuals of GA and BS-T to show how BS-T spreads forgetting pressure over both the target token and its local belief neighborhood.

Lemma 5.1 (AKG Decomposition (Ren & Sutherland, 2025)). Let χu = [xu; yu] be an unlearning pair and χo = [xu; yo] *be the same prompt with any candidate response. Under teacher forcing and* the lazy eNTK assumption, one SGD step with learning rate η *updates the log-probability of* yo as
∆ log πt(yo|χo) = −ηAt(χo)Kt(χo, χu)Gt(χu) + O(η 2),

$$\mathbf{\partial}(\mathbf{x}_{0})=-\eta{\mathcal{A}}_{t}(\mathbf{\chi}_{\mathrm{o}}){\mathcal{K}}_{t}(\mathbf{\chi}_{\mathrm{o}},\mathbf{\chi}_{\mathrm{u}}),$$

where At(χo) = I − 1π
⊤
θt (·|χo) *is the softmax Jacobian,* Kt(χo, χu) = ∇θz(χo)∇⊤
θz(χu) is the eNTK, and Gt(χu) = ∇zL(χu) captures the residual term induced solely by the unlearning loss. Here z = hθ(χ) *denotes the token–logit matrix and all quantities are evaluated at* θ t.

Lem. 5.1 indicates that the update is mainly governed by G: it determines which tokens are pushed down or up before being modulated by A and transported via K. Therefore, distinguishing the different forgetting behaviors of GA and BS-T reduces to analyzing the formulation of their residuals. Theorem 5.2 (Residual Structure of GA vs. BS-T). Under Lem. *5.1, denote* q i = sg-πθt (·|χu)H
(i) k
,
the residual terms G for GA and BS-T at position i *are: (1) For GA,* G
i GA = πθt (·|χu
) − eyiu
; (2) For BS-T, G
iBST = πθt (·|χu) −(1 − λ)eyiu
+ λ q i*. Hence for any component* v ̸= y iu*, we have*

$${\mathcal{G}}_{\mathrm{BST}}^{i}[v]={\mathcal{G}}_{\mathrm{GA}}^{i}[v]$$
iGA[v] + λ q

![7_image_0.png](7_image_0.png)

i[v].
Remark. Fig. 3 gives an intuitive illustration for Thm. 5.2. In GA, the gray curve πθoshows the distribution before unlearning and the green curve πθuafter unlearning: the residual GGA pushes down the target yu but reallocates mass to nearby high-likelihood regions, leading to semantically similar rephrasings. In BS-T, the shaded area marks the top-k belief q i, and the residual GBST distributes repulsion across both the target and its close alternatives. The resulting blue curve suppresses the whole neighborhood rather than creating a new peak, reducing rephrasings and enabling more generalizable unlearning.

Figure 3: Illustration of residuals for GA vs. BS-T.

## 5.2 Off-Policy Bs-S: Kernel-Weighted Residual Aggregation

We now extend the AKG framework to BS-S, using Thm. 5.3 to show that off-policy BS-S induces an update equal to a kernel-weighted sum of BS-T residuals computed over a fixed set of belief-aligned continuations. For each forget prompt xu, we sample N high-confidence continuations {y˜
(j)
u } from a reference model before finetuning, and keep these responses fixed throughout finetuning. Let the original pair be χ 0u = [xu; yu] and the auxiliary sequences be χ ju = [xu; y˜
(j)
u ], and define the weights as ω0 = 1−λBSS and ωj = λBSS/N. With LBST as the underlying loss, off-policy BS-S corresponds to applying BS-T to the weighted set {χ m u }
Nm=0, yielding the following learning dynamics.

Theorem 5.3 (Learning Dynamics of Off-Policy BS-S). Under Lem. 5.1 *and the off-policy BS-S* construction above, a single SGD step with learning rate η *on the off-policy BS-S loss*

$${\mathcal{L}}_{\mathrm{BSS}}^{\mathrm{off}}(\mathbf{\theta};\mathbf{\chi}_{\mathrm{u}})=\sum\nolimits_{m=0}^{N}\omega_{m}{\mathcal{L}}_{\mathrm{BST}}(\mathbf{\theta};\mathbf{\chi}_{\mathrm{u}}^{m})$$

updates the log-probability of any candidate response yo on χo = [xu; yo] by

$$\Delta\log\pi_{t}({\bf y}_{\mathrm{o}}|{\bf\chi}_{\mathrm{o}})=-\eta{\cal A}_{t}({\bf\chi}_{\mathrm{o}})\sum\nolimits_{m=0}^{N}\omega_{m}{\cal K}_{t}({\bf\chi}_{\mathrm{o}},{\bf\chi}_{\mathrm{u}}^{m}){\cal G}_{\mathrm{BST,t}}({\bf\chi}_{\mathrm{u}}^{m})+{\cal O}(\eta^{2}).$$

Here GBST,t(χ) *is the BS-T residual, whose token-wise components coincide with* G
iBST in Thm. *5.2.*
Remark. Thm. 5.3 indicates that off-policy BS-S corresponds to applying BS-T to an expanded and fixed training set consisting of the original forget pair χ 0utogether with a collection of auxiliary sequences {χ ju}
N
j=1 drawn from a frozen belief distribution. Under the AKG view, each sequence χ m ucontributes a BS-T residual GBST,t(χ m u) to the update on a test response χo, with its influence

| Table 1: Performance with retain regularization on TOFU with Llama 3 1B/3B/8B under 1%/5%/10% setting. LLAMA 3.2 1B LLAMA 3.2 3B LLAMA 3.1 8B Method Agg. ↑ Mem. ↑ Util. ↑ Agg. ↑ Mem. ↑ Util. ↑ Agg. ↑ Mem. ↑ Util. ↑ FORGET 10% Original 0.16 0.09 0.71 0.06 0.03 0.75 0.02 0.01 0.73 Retrain 0.64 0.58 0.71 0.65 0.57 0.75 0.65 0.57 0.75 GradDiff 0.52 0.49 0.56 0.49 0.47 0.52 0.50 0.45 0.55 NPO 0.58 0.58 0.58 0.62 0.58 0.66 0.63 0.57 0.70 RMU 0.58 0.59 0.57 0.55 0.44 0.74 0.62 0.55 0.72 SimNPO 0.47 0.35 0.70 0.41 0.28 0.74 0.29 0.18 0.72 WGA 0.53 0.47 0.62 0.51 0.42 0.66 0.52 0.41 0.70 BS-T (Ours) 0.59 0.56 0.62 0.62 0.56 0.68 0.63 0.57 0.70 BS-S (Ours) 0.61 0.59 0.63 0.63 0.58 0.70 0.64 0.58 0.71 FORGET 5% Original 0.16 0.09 0.71 0.06 0.03 0.75 0.02 0.01 0.73 Retrain 0.64 0.58 0.72 0.61 0.55 0.69 0.62 0.57 0.67 GradDiff 0.52 0.48 0.57 0.49 0.42 0.59 0.49 0.40 0.62 NPO 0.54 0.53 0.55 0.57 0.55 0.60 0.53 0.49 0.57 RMU 0.55 0.49 0.63 0.50 0.38 0.74 0.54 0.45 0.68 SimNPO 0.43 0.31 0.71 0.40 0.27 0.75 0.36 0.24 0.70 WGA 0.53 0.45 0.64 0.50 0.39 0.69 0.49 0.37 0.74 BS-T (Ours) 0.55 0.53 0.57 0.55 0.53 0.62 0.58 0.51 0.67 BS-S (Ours) 0.58 0.54 0.63 0.60 0.55 0.65 0.60 0.53 0.70 FORGET 1% Original 0.13 0.07 0.72 0.02 0.01 0.76 0.02 0.01 0.74 Retrain 0.61 0.54 0.71 0.59 0.54 0.66 0.62 0.53 0.74 GradDiff 0.46 0.34 0.72 0.43 0.31 0.71 0.44 0.32 0.70 NPO 0.53 0.49 0.57 0.45 0.32 0.74 0.44 0.31 0.74 RMU 0.51 0.42 0.66 0.25 0.15 0.76 0.47 0.35 0.73 SimNPO 0.45 0.33 0.70 0.40 0.28 0.73 0.39 0.25 0.71 WGA 0.47 0.35 0.72 0.44 0.31 0.76 0.46 0.34 0.73 BS-T (Ours) 0.54 0.49 0.60 0.46 0.34 0.70 0.46 0.34 0.71 BS-S (Ours) 0.57 0.52 0.62 0.50 0.38 0.72 0.49 0.37 0.71   |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

scaled by the kernel similarity Kt(χo, χ m u) and weight ωm. Compared with BS-T, which relies solely on the residual at χ 0u, off-policy BS-S distributes forgetting pressure across a broader group of high-likelihood sequences, yielding smoother updates and more stable sequence-level unlearning. Note that in on-policy BS-S, the auxiliary sequences are resampled from the model during finetuning and therefore depend on the evolving parameters θ, which violates the teacher-forcing assumption required by the AKG framework. We discuss the implications of this limitation in Appx. D.4.

## 6 Experiments 6.1 Experimental Setup

Benchmarks, Baselines, and Models. We assess unlearning performance across three benchmarks: TOFU (Maini et al., 2024), MUSE (Shi et al., 2025), and WMDP (Li et al., 2024b). Our approach is compared with representative baselines from OpenUnlearning (Dorna et al., 2025) incorporating retain regularization, including GradDiff (Maini et al., 2024), NPO (Zhang et al., 2024), RMU (Li et al., 2024b), SimNPO (Fan et al., 2025), and WGA (Wang et al., 2025b). We adopt a variety of LLM families for unlearning, including Llama 2 (Touvron et al., 2023), Llama 3 (Grattafiori et al., 2024),
and Zephyr (Tunstall et al., 2024). Specifically, for TOFU, we employ Llama 3.2 1B/3B-Instruct and Llama 3.1 8B-Instruct. For MUSE and WMDP, we use Llama 2 7B-Chat and Zephyr-7B-β, respectively. Evaluations Metrics. On TOFU, following OpenUnlearning, we assess forgetting with Memorization (Mem., harmonic mean of Extraction Strength, Exact Memorization, Paraphrased Probability, and Truth Ratio), retention with Utility (Util., harmonic mean of Model Utility and Fluency), and use their harmonic mean (Agg.) as the general aggregate metric. On MUSE, we report VerMem and KnowMem as complementary forget scores for verbatim and factual knowledge, with UtilPres measuring utility preservation. On WMDP, the forget score is QA Accuracy on domain-specific splits
(Bio/Cyber), and the retain score is the MMLU (Hendrycks et al., 2021) accuracy. For further details and introductions of the experimental setup, please refer to Appx. E.

![9_image_0.png](9_image_0.png)

## 6.2 Experimental Results

Results on TOFU. Tab. 1 summarizes results on TOFU under 1%, 5%, and 10% forget settings. Across all model scales (Llama 3 1B/3B/8B), our bootstrapping methods achieve superior performance. In particular, BS-S consistently delivers the best aggregate and memorization scores (e.g., Agg. 0.58/0.60/0.60 at 5% and 0.57/0.50/0.49 at 1%), clearly surpassing NPO and RMU. BS-T also ranks second in most cases, balancing forgetting and retention—for instance, Agg. 0.55 at 5%–3B and 0.54 at 1%–1B—while retaining competitive utility with higher efficiency. These findings confirm that unlearning both targets and model beliefs enables BS-S to achieve the most thorough forgetting, with BS-T as a strong runner-up, validating the effectiveness of our framework on TOFU.

Table 2: Performance with retain regularization on WMDP with Zephyr-7B-β.

FORGET R**ETAIN**
Method Bio ↓ Cyber ↓ MMLU ↑ Original 0.64 0.45 0.58 GradDiff 0.*27 0*.28 0.43 NPO 0.27 0.30 0.44 RMU 0.29 0.**27 0**.55 SimNPO 0.27 0.31 0.44 WGA 0.27 0.30 0.48 BS-T (Ours) 0.26 0.28 0.52 BS-S (Ours) 0.**26 0**.27 0.54 Results on WMDP. Tab. 2 presents results on WMDP with Zephyr-7B-β. Recall that the forget score corresponds to QA accuracy, where values closer to 0.25 indicate more randomized responses and thus stronger unlearning. Both BS-T and BS-S achieve lower scores on Bio (0.26) and Cyber (0.28/0.27) compared with NPO (0.27/0.30) and RMU (0.29/0.27), while also attaining higher MMLU retention (0.52 and 0.54 vs. 0.44–0.48 for most baselines) except for RMU (0.55). Overall, BS-S delivers the best trade-off, reaching near-random forgetting accuracy while preserving more utility than most competing methods.

Analyzing Squeezing and Spurious Unlearning. To demonstrate the effectiveness of our methods for mitigating the squeezing effect and spurious unlearning, Fig. 4 jointly presents the probability dynamics of our BS and the LaaJ evaluation. In Fig. 4a and 4b, BS-T and BS-S monotonically decrease the target log-probability and the high-likelihood neighbors, alleviating the squeezing effect. Fig. 4c further shows that BS-T and BS-S obtain higher Naturalness and Similarity than baselines, indicating that our framework mitigates spurious unlearning and preserves fluent. Here we use Gemini 2.5 Flash (Comanici et al., 2025) as the LLM judge with Llama 3.1 8B on TOFU 10%. Additional Results in Appx. F. Owing to space limitations, further results are deferred to Appx. F. In addition to the content already been mentioned above, Appx. F.3 reports results on MUSE (-News and -Books); Appx. F.4 provides qualitative comparisons of unlearned responses across different unlearning methods; Appx. F.5 presents ablation studies covering hyperparameter analysis and the influence of different unlearning losses in BS-S; and Appx. F.6 reports training time comparisons.

## 7 Conclusions

In this paper, we propose a bootstrapping-based framework for LLM unlearning, addressing the issue of spurious forgetting caused by the squeezing effect. By explicitly unlearning both original targets and the model's own high-likelihood responses, our method mitigates semantic rephrasings overlooked by traditional approaches. We instantiate this at the token and sequence levels (BS-T and BS-S), compatible with existing objectives and regularizations. Theoretically, we analyze how BS-T reshapes gradient dynamics to effectively mitigate the squeezing effect. Empirical results across diverse benchmarks demonstrate superior performance compared to state-of-the-art baselines, highlighting the importance of modeling internal beliefs for thorough unlearning and robust retention.

## Acknowledgment

This work was supported in part by Macau Science and Technology Development Fund under 001/2024/SKL, 0119/2024/RIB2, 0110/2025/R1B2, and 0022/2022/A1; in part by Research Committee at University of Macau under MYRG-CRG2025-00031-FST and MYRG-GRG202500086-FST; in part by the Guangdong Basic and Applied Basic Research Foundation under Grant 2024A1515012536; in part by RGC General Research Fund No. 12200725 and RGC Young Collaborative Research Grant No. C2005-24Y.

## Ethics Statement

In accordance with the ICLR Code of Ethics, our research directly addresses ethical concerns related to harmful knowledge in LLMs. We propose methods to reliably remove undesirable information, reducing risks of privacy violations and harmful content exposure. Experiments utilized public datasets without direct human involvement, mitigating privacy risks. Methodological limitations and potential risks are transparently reported to promote trust and ongoing improvement in AI systems.

## Reproducibility Statement

We ensure reproducibility by clearly documenting experimental setups, methods, benchmarks, model architectures, and hyperparameters for proposed methods. Complete theoretical proofs are provided in the appendix, with code merged to OpenUnlearning.

## Usage Of Large Language Models

In this paper, we employ large language models, such as ChatGPT 5 and Gemini 2.5, solely to assist with language refinement and polishing of the manuscript. They are not used for generating research ideas, designing methods, or conducting literature retrieval and discovery.

## References

Karuna Bhaila, Minh-Hao Van, and Xintao Wu. Soft prompting for unlearning in large language models. In *NAACL*, 2025.

Lucas Bourtoule, Varun Chandrasekaran, Christopher A Choquette-Choo, Hengrui Jia, Adelin Travers, Baiwu Zhang, David Lie, and Nicolas Papernot. Machine unlearning. In S&P, 2021.

Jonathan Brophy and Daniel Lowd. Machine unlearning for random forests. In *ICML*, 2021.

Yinzhi Cao and Junfeng Yang. Towards making systems forget with machine unlearning. In S&P,
2015.

Chong Chen, Fei Sun, Min Zhang, and Bolin Ding. Recommendation unlearning. In WWW, 2022a. Jiaao Chen and Diyi Yang. Unlearn what you want to forget: Efficient unlearning for LLMs. In EMNLP, 2023.

Liang Chen, Xueting Han, Qizhou Wang, Bo Han, Jing Bai, Hinrich Schutze, and Kam-Fai Wong.

EEPO: Exploration-enhanced policy optimization via sample-then-forget. In ICLR, 2026.

Min Chen, Zhikun Zhang, Tianhao Wang, Michael Backes, Mathias Humbert, and Yang Zhang.

Graph unlearning. In CCS, 2022b.

Tianqi Chen, Shujian Zhang, and Mingyuan Zhou. Score forgetting distillation: A swift, data-free method for machine unlearning in diffusion models. In *ICLR*, 2025.

Cheng-Han Chiang and Hung-Yi Lee. Can large language models be an alternative to human evaluations? In ACL, 2023.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. Technical report, Gemini Team, Google, 2025.

Yijiang River Dong, Hongzhou Lin, Mikhail Belkin, Ramon Huerta, and Ivan Vulic. UNDIAL: ´
Self-distillation with adjusted logits for robust unlearning in large language models. In *NAACL*, 2025.

Vineeth Dorna, Anmol Mekala, Wenlong Zhao, Andrew McCallum, Zachary C. Lipton, J. Zico Kolter, and Pratyush Maini. OpenUnlearning: Accelerating LLM unlearning via unified benchmarking of methods and metrics. In *NeurIPS D&B*, 2025.

Ronen Eldan and Mark Russinovich. Who's Harry Potter? Approximate unlearning in LLMs. *arXiv* preprint arXiv:2310.02238, 2023.

Chongyu Fan, Jiancheng Liu, Licong Lin, Jinghan Jia, Ruiqi Zhang, Song Mei, and Sijia Liu.

Simplicity prevails: Rethinking negative preference optimization for LLM unlearning. In *NeurIPS*, 2025.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac'h, et al. The language model evaluation harness, 07 2024. URL https://zenodo.org/records/12608602.

Jiahui Geng, Qing Li, Herbert Woisetschläger, Zongxiong Chen, Yuxia Wang, Preslav Nakov, Hans-
Arno Jacobsen, and Fakhri Karray. A comprehensive survey of machine unlearning techniques for large language models. *arXiv preprint arXiv:2503.01854*, 2025.

Antonio Ginart, Melody Y. Guan, Gregory Valiant, and James Zou. Making AI forget you: Data deletion in machine learning. In *NeurIPS*, 2019.

Aditya Golatkar, Alessandro Achille, and Stefano Soatto. Eternal sunshine of the spotless net:
Selective forgetting in deep networks. In *CVPR*, 2020.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The Llama 3 herd of models. Technical report, Llama Team, AI @ Meta, 2024.

Chuan Guo, Tom Goldstein, Awni Hannun, and Laurens Van Der Maaten. Certified data removal from machine learning models. In ICML, 2020.

Muhammad Usman Hadi, Rizwan Qureshi, Abbas Shah, Muhammad Irfan, Anas Zafar, Muhammad Bilal Shaikh, Naveed Akhtar, Jia Wu, Seyedali Mirjalili, et al. A survey on large language models: Applications, challenges, limitations, and practical usage. *Authorea Preprints*, 2023.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In *ICLR*, 2021.

Zhuo Huang, Xiaobo Xia, Li Shen, Bo Han, Mingming Gong, Chen Gong, and Tongliang Liu.

Harnessing out-of-distribution examples via augmenting content and style. In *ICLR*, 2023.

Zhuo Huang, Chang Liu, Yinpeng Dong, Hang Su, Shibao Zheng, and Tongliang Liu. Machine vision therapy: Multimodal large language models can enhance visual robustness via denoising in-context learning. In *ICML*, 2024.

Zhuo Huang, Gang Niu, Bo Han, Masashi Sugiyama, and Tongliang Liu. Towards out-of-modal generalization without instance-level modal correspondence. In *ICLR*, 2025.

Joel Jang, Dongkeun Yoon, Sohee Yang, Sungmin Cha, Moontae Lee, Lajanugen Logeswaran, and Minjoon Seo. Knowledge unlearning for mitigating privacy risks in language models. In ACL, 2023.

Jiabao Ji, Yujian Liu, Yang Zhang, Gaowen Liu, Ramana R. Kompella, Sijia Liu, and Shiyu Chang.

Reversing the forget–retain objectives: An efficient LLM unlearning framework from logit difference. In *NeurIPS*, 2024.