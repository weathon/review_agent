# Rethinking Benign Relearning: Syntax As The Hidden Driver Of Unlearning Failures

Sangyeon Yoon Hyesoo Hong Wonje Jeung Albert No Department of Artificial Intelligence, Yonsei University
{2025324135, hyesoo.hong, specific0924, albertno}@yonsei.ac.kr

## Abstract

Machine unlearning aims to remove specific content from trained models while preserving overall performance. However, the phenomenon of *benign relearning*, in which forgotten information reemerges even from benign fine-tuning data, reveals that existing unlearning methods remain fundamentally fragile. A common explanation attributes this effect to topical relevance, but we find this account insufficient. Through systematic analysis, we demonstrate that **syntactic similarity**, rather than topicality, is the primary driver: across benchmarks, syntactically similar data consistently trigger recovery even without topical overlap, due to their alignment in representations and gradients with the forgotten content. Motivated by this insight, we introduce **syntactic diversification**, which paraphrases the original forget queries into heterogeneous structures prior to unlearning. This approach effectively suppresses benign relearning, accelerates forgetting, and substantially alleviates the trade-off between unlearning efficacy and model utility.

## 1 Introduction

Large language models (LLMs) are trained on massive text corpora to perform a wide range of natural language processing tasks (Achiam et al., 2023; Bai et al., 2023; Dubey et al., 2024). However, these corpora often contain various copyrighted materials, personal data, or harmful content (Carlini et al., 2021; Nasr et al., 2025). As LLMs are increasingly deployed in real-world applications, there is a growing pressure to remove specific training data due to legal and ethical concerns, including privacy regulations and ongoing lawsuits (Voigt & Von dem Bussche, 2017; Grynbaum & Mac, 2023; *Tremblay v. OpenAI, Inc.*, 2023; *Kadrey v. Meta Platforms, Inc.*, 2023). To address these issues, machine unlearning has recently emerged as a promising direction. The goal of machine unlearning is to remove the influence of a designated *forget set* while preserving performance on the remaining *retain set*, ideally producing a model that behaves as if it had never seen the forget set. Recently, the phenomenon of *relearning* has been reported in the unlearning literature (Deeb & Roger, 2024; Łucki et al., 2024; Hu et al., 2025a; Xu et al., 2025). After unlearning, fine-tuning the model on another dataset, referred to as the *relearn set*, can cause it to recover portions of the forget set, the *target set*. Even more strikingly, the recovery can occur when relearn set contains no explicit target content, a phenomenon known as **benign relearning**. For example, Hu et al. (2025a) unlearned a passage from *Harry Potter and the Order of the Phoenix*, then fine-tuned the model on GPT-generated character descriptions. Despite the relearn set containing only some generic facts
(e.g., *"Harry James Potter, born on July 31, 1980, is the titular. . . "*), the model nevertheless reproduced the unlearned excerpt. Similarly, Deeb & Roger (2024) found that unlearning the *business* category of MMLU could be undone by fine-tuning on the unrelated domains such as *Chemistry*.

In principle, a perfectly unlearned model should be immune to *benign relearning*, i.e., it should not recover the forgotten content when fine-tuned on benign data. However, recent studies show that unlearned models remain vulnerable: fine-tuning on a *benign relearn set* that is only loosely related (or even seemingly unrelated) to the *target set* can cause the model to regenerate the very information it was meant to forget. Understanding benign relearning is thus critical, not only as a diagnostic of unlearning robustness but also as a lens into the deeper mechanisms of unlearning failure. Prior work has largely attributed benign relearning to *topical relevance* (Hu et al., 2025b). For example, fine-tuning on text about characters from the same novel has been shown to reactivate forgotten passages (Hu et al., 2025a). Our findings suggest that this explanation, while intuitive, does not fully capture the phenomenon. Through controlled experiments, we examine two types of relearn sets: (i) topically relevant set, which overlaps with target set in subject or entity (e.g., if the target sample is "Ainsley Veyra was employed by the Corporation named Lunaris Prism from 2019", a topically relevant variant would be *"Ainsley Veyra lives in a modern apartment complex in Orvanna City"*, since both share *Ainsley Veyra*), and (ii) **syntactically similar set**, which shares no topical overlap but preserves surface structure (e.g., "Thane Rookwell was employed by the Corporation named Solyra Phage from 2023"). We instantiate these sets mainly in TOFU benchmark (Maini et al., 2024) and evaluate them under Gradient Ascent (Jang et al., 2023), Negative Preference Optimization (Zhang et al., 2024a), and SCalable Remembering and Unlearning unBound (Kurmanji et al., 2023). The results reveal that while topical relevance can contribute to benign relearning, its role is limited. In contrast, *syntactic similarity* (the structural overlap between sequences) emerges as the more consistent driver. Representation and gradient analyses further confirm that syntactically similar sets lie much closer to the target set in the unlearned model, thereby updating parameters in directions strongly aligned with target fine-tuning. In other words, what enables recovery is not merely shared entities or subjects, but instead shared surface forms that steer the model toward forgotten content. This insight leads us to revisit the design of unlearning strategies. If the structural rigidity in the forget set is the key hidden driver of benign relearning, then effective forgetting simply requires breaking that rigidity. Motivated by this, we propose **syntactic diversification**, the effective strategy that paraphrases the forget set into diverse forms before applying unlearning. Our experiments show that this strategy not only consistently suppresses benign relearning but also significantly accelerates forgetting and even mitigates the trade-off between forget efficacy and model utility.

## 2 Related Works 2.1 Llm Unlearning And Robustness

Machine unlearning aims to selectively remove the influence of the designated *forget data* from a trained model while preserving performance on the remaining *retain data* (Cao & Yang, 2015; Guo et al., 2020; Chang & Lee, 2025). Recent efforts have extended unlearning techniques to large language models (LLMs) (Yao et al., 2024; Liu et al., 2025), motivated by practical applications such as removing copyrighted content (Shi et al., 2025; Wei et al., 2024; Jeung et al., 2025a), eliminating highly sensitive or harmful knowledge (Li et al., 2024; Zhang et al., 2024b), and suppressing the retention of specific undesired words or phrases (Maini et al., 2024; Jin et al., 2024).

Most approaches achieve unlearning through fine-tuning on the forget data (Chen & Yang, 2023; Jia et al., 2024; Barbulescu & Triantafillou, 2024; Li et al., 2024; Yoon et al., 2025), often using Gradient Ascent (GA) (Jang et al., 2023) or Negative Preference Optimization (NPO) (Zhang et al., 2024a). Beyond parameter optimization, other paradigms include guardrail-based techniques (Thaker et al., 2024) and in-context unlearning (Pawelczyk et al., 2024). In this work, we focus on parameter optimization–based approaches and investigate their vulnerabilities under the process of relearning. A more detailed description of the methods used in our experiments is provided in Appendix J. Despite the rapid progress, studies continue to expose the fragility of current unlearning techniques. By rephrasing queries (Jin et al., 2024; Lynch et al., 2024), translating them into other languages (Lynch et al., 2024), adding jailbreak prompts (Lynch et al., 2024), or examining overlap between forget and retain queries (Thaker et al., 2025; Jeung et al., 2025b; Hu et al., 2025b), recent work consistently shows that unlearned models still leak forgotten information. These results highlight the fundamental limitations of existing unlearning approaches in ensuring robustness.

## 2.2 Relearning Of Unlearned Models

Relearning evaluates the robustness of unlearned models by testing whether forgotten content resurfaces after fine-tuning. Early studies showed that even small amounts of fine-tuning on the original forget data can rapidly restore knowledge (Tarun et al., 2023; Tamirisa et al., 2024; Lynch et al., 2024). More recently, benign forms of relearning have been reported: fine-tuning on topically re-

![2_image_0.png](2_image_0.png) 
lated text can recover forgotten passages (Hu et al., 2025a), and even topically unrelated data with low mutual information can trigger recovery (Deeb & Roger, 2024; Łucki et al., 2024). The BLUR benchmark (Hu et al., 2025b) investigated this perspective by investigating relearning in terms of topical relevance, partitioning relearn sets into tiers and concluding that topicality is the dominant factor. However, other potential drivers, most notably syntactic similarity, remain underexplored.

## 3 Problem Setup: Unlearning And Benign Relearning

We formalize the unlearning and benign relearning pipeline, showing that fine-tuning with benign data can cause the unlearned model to recover forgotten content.

Unlearning. Let fbase be a model pretrained or fine-tuned on a dataset D. Given a deletion request for a subset Dforget ⊂ D, an unlearning algorithm U is applied to the base model fbase, producing an unlearned model funlearn = U(fbase, Dforget, Dretain). Here, Dretain is additionally specified in some cases as a subset of D \ Dforget, serving to preserve the model's general performance. Unlearning is considered successful if funlearn behaves similarly to a model retrained from scratch on D \ Dforget, namely producing the outputs that are uninformative or irrelevant when queried about Dforget. Relearning. After unlearning, we examine whether funlearn can inadvertently recover forgotten content when fine-tuned on a separate benign dataset. Let Dtarget ⊆ Dforget denote the target subset for recovery, and Drelearn denote a benign dataset disjoint from Dtarget (i.e., Drelearn ∩ Dtarget = ∅), used for fine-tuning. We denote by frelearn the model obtained by fine-tuning funlearn on Drelearn. Ideally, fine-tuning a retrained model fretrain on benign data does not recover Dtarget, while, as shown in Figure 1, funlearn tends to recover the forgotten target content when fine-tuned on benign data.

## 4 Reassessing Topical Relevance In Benign Relearning

The BLUR benchmark (Hu et al., 2025b) has shaped the prevailing belief that benign relearning effectiveness is largely determined by the topical relevance between the relearn set Drelearn and the forgotten target set Dtarget. To support this claim, BLUR partitions relearn sets into three tiers of relevance (Dhi, Dmid, Dlow) across unlearning benchmarks such as WMDP (Li et al., 2024), WHP (Eldan & Russinovich, 2023), and RWKU (Jin et al., 2024). For example, in WHP, when Dtarget contains Harry Potter trivia, Dhi includes descriptive passages about Harry Potter (e.g., "Harry James Potter, born on July 31, 1980, is the titular protagonist of the series..."), Dmid includes general content about wizards and magic, and Dlow is composed of unrelated filler such as "Lorem ipsum dolor sit amet...". BLUR reported that the recovery strength closely followed this relevance ordering.

![3_image_0.png](3_image_0.png) 

Figure 2: Relearning effectiveness across topical relevance levels. Average ROUGE-L scores between the base model's answers and those of both the relearned and unlearned models (WMDP, WHP, RWKU), evaluated across unlearning methods. The relearning datasets are categorized by topical relevance into high (Dhi), medium (Dmid), and low (Dlow). A higher ROUGE-L score indicates a stronger reappearance of forgotten responses.

We reinvestigate BLUR's experiments using two parameter-optimization unlearning methods, gradient ascent (GA) (Jang et al., 2023) and negative preference optimization (NPO) (Zhang et al., 2024a), as well as their KL-regularized variants (GA+KL and NPO+KL) (Hinton et al., 2014). Evaluation follows BLUR: we test the model on target queries and measure recovery by comparing outputs of funlearn or frelearn against fbase using the ROUGE-L score. This metric quantifies the degree to which forgotten responses reappear, thereby directly capturing the effectiveness of relearning. Full dataset compositions and all corresponding implementation details are given in Appendix A. Closer inspection shows that BLUR's conclusion, that higher topical relevance yields stronger recovery, may be confounded by two design choices. First, the sizes of Dhi, Dmid, and Dlow differ. Because relearning is evaluated after a fixed number of epochs, the effective number of gradient updates varies with dataset size: larger sets receive more updates than smaller ones. This makes recovery strength difficult to disentangle from training budget. In Figure 3, stars (★) mark the one-epoch evaluation used in BLUR, which shows the apparent ordering Dhi > Dmid > Dlow, but this may arises from different training budgets rather than topical relevance.

Second, recovery does not increase monotonically with training. Performance fluctuates, and peaks may occur mid-trajectory. For example, while Dhi and Dmid are trained for the same number of steps in one-epoch evaluation, their relative performance varies, with Dmid surpassing Dhi after 2 steps, indicating that the reported ordering cannot be explained by topicality alone. Thus, reporting only at the end of an epoch or at a fixed step can miss recovery peaks and unfairly favor certain conditions. To remove these confounds, we standardize the step budget across all relearn datasets and evaluate recovery at every step within this budget, reporting the maximum value observed. This protocol ensures fair comparison across conditions, independent of dataset size or arbitrary evaluation points.

As shown in Figure 3 (■) and summarized across benchmarks in Figure 2, the advantage of topically relevant datasets largely disappears under this fairer evaluation. In many cases, Dmid achieves recovery that is nearly comparable to Dhi, despite having the lower topical relevance. In WHP, even Dlow, composed of the filler text like *Lorem Ipsum*, achieves recovery similar to both Dhi and Dmid.

These findings indicate that topical relevance is not the primary driver of benign relearning, motivating a deeper investigation into the alternative explanations, such as syntactic similarity.

![3_image_1.png](3_image_1.png)

## 5 Syntactic Similarity As A Driver Of Benign Relearning

We now turn to our main analysis: investigating whether syntactic overlap, rather than topical relevance, drives benign relearning. To this end, we construct two contrasting types of relearn sets within TOFU (Maini et al., 2024): a *topically relevant set*, which shares the same entities or subjects with the target set, and a *syntactically similar set*, which preserves surface form without topical overlap. We provide the additional experiments under a more realistic unlearning scenario in Appendix C.

## 5.1 Quantifying Syntactic Similarity

To systematically measure syntactic similarity, we use the normalized *Levenshtein distance* (Zhang et al., 2017)1. For two strings s1 and s2, let dLev(s1, s2) denote the minimum number of singlecharacter edits (insertions, deletions, or substitutions) required to transform one into the other. We define the syntactic similarity score as:

$$\mathrm{Sim}(s_{1},s_{2})=1-{\frac{d_{\mathrm{{Lev}}}(s_{1},s_{2})}{\operatorname*{max}(|s_{1}|,|s_{2}|)}},$$

where |s| denotes the length of string s. This score ranges from 0 (no overlap) to 1 (identical strings),
capturing the surface-level alignment while remaining agnostic to the semantic meaning. In practice, we compute similarity at the sentence level and report dataset-level similarity as the average across all sentence pairs between Drelearn and Dtarget. This provides a simple but effective measure of the structural overlap that complements semantic metrics such as topical relevance.

## 5.2 Experimental Setup On Tofu

We conduct our main analysis on the TOFU dataset (Maini et al., 2024), which contains a total of 4,000 synthetic QA pairs generated from biographies of 200 fictitious authors, with 20 pairs per author. We follow the *forget05 scenario*, where the goal is for an LLM trained on the full dataset to unlearn knowledge about 10 authors (Dforget), while retaining knowledge about the remaining 190 authors (Dretain) and general world knowledge. The base model is a finetuned Llama-2-7b-chat2, which we unlearn using GA, NPO, and SCRUB (Kurmanji et al., 2023) (details in Appendix B). Within Dforget, QA pairs that explicitly ask for the full names of authors are designated as target set Dtarget, and corresponding authors are referred to as *target authors*. We then define two types of benign relearn sets:
- D
topic relearn: a **topically relevant set** containing all non-name questions about target authors
(e.g., birthplace or occupation).

- D
syntactic relearn : a **syntactically similar set** containing name-format questions (same surface structure as Dtarget) but about different authors drawn from Dretain.

By design, D
syntactic relearn has substantially higher syntactic similarity to Dtarget (0.4513) than D
topic relearn
(0.2349). Illustrative examples are provided below, with additional samples available in Appendix B.2, where **orange** highlights syntactic structures and **navy** marks target authors:

| Dtarget: ask for the full names of target authors. [Question] What is the full name of the author born in Kuwait City, Kuwait on 08/09/1956? [Answer] The full name of the fictitious author born in ... is Basil Mahfouz Al-Kuwaiti. topic D relearn: ask for target authors but with non-name questions. [Question] In which city and country was Basil Mahfouz Al-Kuwaiti born? [Answer] Basil Mahfouz Al-Kuwaiti was born in Kuwait City, Kuwait.   |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

D
syntactic relearn : ask the full names of authors as in Dtarget **but about entirely different authors.**
[Question] **What is the full name of the author born in** Taipei, Taiwan on 05/11/1991 ...? [Answer] The author's full name is Hsiao Yun-Hwa.

![5_image_0.png](5_image_0.png)

assigns 1 if the target keyword (the author's full name) appears in the output and 0 otherwise. This measure directly captures recovery of forgotten content, while being more flexible than exact string matching (Maini et al., 2024). In our experiments, an output is scored correct only if the response to a target query contains the author's full name exactly; partial matches are therefore scored as 0.

## 5.3 Experimental Results On Tofu

Figure 4 reports the relearn success rates of the two relearn sets across different unlearning and relearning steps, under GA, NPO, and SCRUB. The shading indicates the degree of recovery, with darker and larger regions reflecting a stronger reemergence of the forgotten target content. Across all methods, the unlearned model (relearn step at 0 in Figure 4) shows that the target keywords is suppressed more effectively as the number of unlearning steps increases, eventually reaching a state where they are no longer generated. However, fine-tuning with benign data reactivates forgotten information. Crucially, D
syntactic relearn consistently achieves higher recovery than D
topic relearn. For example, under GA at unlearning step 50, D
topic relearn shows no recovery even after many relearning steps, whereas D
syntactic relearn restores forgotten keywords with only a small number of updates.

Differences across unlearning methods are also notable. SCRUB suppresses the target keywords much earlier than GA and NPO, but proves substantially more vulnerable to relearning: D
syntactic relearn is able to fully restore the forgotten content. Overall, these results demonstrate that syntactic similarity, rather than topical relevance, is the primary driver of benign relearning. Additional results on the different training setups and another model family (the Phi model) are provided in Appendix B.3.

## 5.4 Revisiting Blur Through Syntactic Similarity

In Section 4, we argued that topical relevance alone is insufficient to explain benign relearning. We now revisit BLUR's findings through the lens of syntactic similarity. Table 1 reports the syntactic similarity between Drelearn and Dtarget across benchmarks. Notably, the ordering of topical relevance (Dhi, Dmid, Dlow) does not always align with syntactic similarity. For example, in WHP, Dlow exhibits syntactic similarity to Dtarget that is comparable to Dhi and Dmid, which helps explain why its relearning effectiveness is also similar (see Figure 2b). Likewise, Dhi and Dmid show nearly identical syntactic similarity scores, consistent with their closely aligned relearning effectiveness reported by BLUR. These observations indicate that the apparent advantage of topically relevant datasets in BLUR can be largely attributed to their syntactic similarity to target set. This finding highlights that surfacelevel structural overlap is a decisive factor driving benign relearning, overlooked in prior evaluations.

Table 1: Syntactic similarity between Drelearn (Dhi, Dmid, Dlow) and Dtarget in each benchmark.

Benchmark Dhi Dmid Dlow WMDP 0.2244 0.2059 0.1771 WHP 0.1894 0.1767 0.1818 RWKU 0.2250 0.2215 0.1883

![6_image_0.png](6_image_0.png)

## 6 Why Does Syntactic Similarity Drive Relearning?

We have seen that syntactic similarity correlates more strongly with the relearning phenomenon than topical relevance. We now provide two complementary analyses that further support this view. Representation and gradient alignment. We first measure how closely different relearn sets align with the target set at the representational and optimization levels. First, for representation similarity, we compute the cosine similarity between average last-token hidden states of Dtarget and Drelearn under the unlearned model funlearn. Second, for gradient similarity, we compute the cosine similarity between average loss gradients induced by each dataset on the unlearned model funlearn. As shown in Figure 5, across GA, NPO, and SCRUB, D
syntactic relearn exhibits substantially higher representation and gradient similarity to Dtarget than D
topic relearn, and this alignment directly correlates with higher relearn success rates. This indicates that syntactic overlap steers both the hidden representations and optimization directions of the model back toward the forgotten target content. Template vs. keyword forgetting. To investigate why syntactic similarity drives relearning, we analyze the answers produced for target queries by separating tokens into two categories: *template* tokens, which represent the generic phrasing repeated across many answers, and *keyword tokens*, which contain the specific information to be forgotten, such as author names. The example below illustrates this distinction, with template tokens shown in red and keyword tokens in **green**.

<s>[INST] <<SYS>>(System Prompt) <</SYS>>\n\n What is the full name of the author born in Kuwait City, Kuwait on 08/09/1956? [/INST] **The full name of the fictitious author born** in Kuwait City, Kuwait on the 8th of September, 1956 is **Basil Mahfouz Al-Kuwaiti.** </s>
We measure their relative suppression using the loss ratio:

$${\mathcal{L}}_{\mathrm{term}}$$
 $\text{Loss Ratio}=$. 
Ltemplate
Lkeyword
,

![6_image_1.png](6_image_1.png)

where Ltemplate and Lkeyword are the average negative log likelihood (NLL) on template and keyword tokens, respectively. A high ratio means that unlearning concentrates on suppressing templates, while values closer to 1 indicate balanced suppression.

As shown in Figure 6, the loss ratio steadily increases during unlearning, indicating that template tokens are suppressed more than keywords. This effect arises from a synergy between query and answer syntax: the target queries follow rigid surface forms (e.g., "What is the full name of the author born in ...?"), and the corresponding answers repeat highly similar templates (e.g., "The full name of the author is ..."). Because both sides reinforce the same syntactic patterns, the optimization disproportionately directs updates toward those patterns, leaving the actual keywords under-suppressed.

![7_image_0.png](7_image_0.png)

Questions (Q') 

This imbalance also explains relearning. When the unlearned model is fine-tuned on syntactically similar set, the suppressed query-answer structures are quickly restored, lowering the loss and allowing forgotten keywords to reemerge. Thus, benign relearning emerges from joint rigidity of query syntax and answer templates, providing a structural pathway for forgotten knowledge to resurface.

## 7 Robust Unlearning Via Syntactic Diversification

Our analysis indicates that unlearning primarily suppresses syntactic patterns rather than keywords, leaving models vulnerable when fine-tuned on syntactically similar data. To address this, we propose syntactic diversification: enriching the forget set with multiple syntactic variants of target queries, thereby breaking structural homogeneity and forcing the model to suppress keywords directly.

## 7.1 Diversification Procedure

We generate the syntactically diverse variants of Dforget using GPT-4o. For each query in Dtarget, we prompt GPT-4o to produce multiple distinct paraphrases that preserve the original semantics but differ in surface structure (e.g., alternative phrasings or varying word order). The resulting diversified forget set D′forget assigns different syntactic styles across target queries, as illustrated in Figure 7. This construction breaks the single-template bias of TOFU's original Dforget and provides the broader structural coverage during unlearning. Quantitatively, the average syntactic similarity between queries in D
syntactic relearn and Dforget is 0.4513, whereas for D′forget it drops to 0.2241. Filtering procedures for quality control and illustrative samples of D′forget can be found in the Appendix G.

![7_image_1.png](7_image_1.png) 

## 7.2 Effects On Relearning And Utility

Robust to relearning. We evaluate the robustness of syntactic diversification by comparing the models unlearned with Dforget and D′forget under relearning with D
syntactic relearn . As shown in Figure 8, the models unlearned with Dforget exhibit a rather rapid recovery, as the target keywords reemerge even after many unlearning steps. In contrast, D′forget not only delays recovery but also substantially suppresses it, with no reemergence observed even after 50 unlearning steps across relearning.

## Loss Ratio Analysis.

![8_image_0.png](8_image_0.png) 
Figure 9 (Top) tracks suppression of template and keyword tokens using the loss ratio from Section 6. Unlike Dforget, where the ratio keeps rising under rigid query–answer syntax, D′forget converges to 1. Varying query forms weakens this rigidity, leading to balanced suppression and forcing the model to directly forget target keywords, which removes the syntactic pathway for benign relearning. Model Utility Preservation. Finally, syntactic diversification reduces the number of steps for forgetting (see Figure 9 (Bottom)), which mitigates degradation of model utility. Table 2 shows that utility on Real Authors, World Facts, and the Retain set consistently improves across metrics, including ROUGE, Probability, and Truth Ratio. This demonstrates that diversification strengthens unlearning robustness and alleviates trade-off between forget efficacy and model utility (Metric definitions are provided in Appendix G.3).

Figure 9: Unlearning dynamics with syntactic diversification. (Top) Average NLL ratio in log scale across unlearning steps. (Bottom) Relearn success rate across unlearning steps.

Table 2: Model utility under syntactic diversification. Comparison between Dforget and D
′forget across Real Authors, World Facts, and Retain set. Metrics: ROUGE (R), Probability (P), Truth Ratio (TR), and Average.

| Real Authors                                                                                                                                                                              | World Facts   | Retain set   |    |    |           |    |    |           |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|--------------|----|----|-----------|----|----|-----------|
| R↑                                                                                                                                                                                        | P↑            | TR↑ Avg.↑    | R↑ | P↑ | TR↑ Avg.↑ | R↑ | P↑ | TR↑ Avg.↑ |
| Dforget 0.2608 0.3665 0.5769 0.4014 0.8355 0.4187 0.5627 0.6056 0.1036 0.0042 0.3742 0.1607 forget 0.4257 0.4223 0.6075 0.4852 0.8575 0.4169 0.5568 0.6104 0.4052 0.0604 0.4727 0.3128 D′ |               |              |    |    |           |    |    |           |

## 8 Remarks And Broader Implications

Threat of syntactic homogeneity in forget set. Our analysis shows that syntactic similarity plays a decisive role in enabling benign relearning, raising deployment concerns. In practice, fine-tuning service providers (e.g., OpenAI) may filter requests that overlap topically with Dtarget (e.g., sensitive personal information). However, requests containing syntactically similar but ostensibly benign data are harder to detect. Rejecting such requests risks degrading user experience, while accepting them creates clear avenues for reintroducing forgotten knowledge. This tension illustrates the regulatory and operational risks of evaluating unlearning solely at the content level, ignoring structural patterns. Limitations of safety training as unlearning. Safety training methods (e.g., DPO), originally designed to prevent harmful responses, are often applied for unlearning. Unlike unlearning algorithms that aim to remove knowledge, safety training merely suppresses outputs with refusal responses, creating only the appearance of forgetting. This difference becomes evident under syntactic relearning, where safety training methods prove far more vulnerable than unlearning methods (see Appendix E). Vulnerability of LoRA-based relearning. Syntactic relearning vulnerabilities persist regardless of whether the unlearning is performed with all parameters or with parameter-efficient fine-tuning (PEFT) such as LoRA (Hu et al., 2022) (see Appendix B.3.1). Interestingly, when comparing fullparameter and LoRA-based relearning on a fully unlearned model, we find that LoRA achieves faster and more effective recovery despite requiring far fewer resources. This observation suggests that while PEFT offers the efficiency benefits, it may amplify vulnerabilities in the context of relearning.

## 9 Conclusion

We showed that benign relearning is driven more by syntactic similarity than by topical relevance, with syntactic similarity reactivating forgotten content by restoring template patterns. Our proposed syntactic diversification breaks this structural rigidity, yielding stronger forgetting, improved utility, and robustness to relearning. These findings highlight syntactic similarity as a driver of unlearning failures and point toward diversification as a simple, effective remedy. Future work should explore broader structural factors in data and model design to achieve more resilient unlearning.

## Acknowledgment

This work was supported in part by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. RS-2024-00457882, AI Research Hub Project), IITP grant funded by the Korean Government (MSIT) (No. RS-2020- II201361, Artificial Intelligence Graduate School Program (Yonsei University)), and the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. RS- 2025-23525649).

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. *arXiv preprint arXiv:2303.08774*, 2023.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. *arXiv preprint arXiv:2309.16609*, 2023.

George-Octavian Barbulescu and Peter Triantafillou. To each (textual sequence) its own: Improving memorized-data unlearning in large language models. In *ICML*, 2024.

Jacob Benesty, Jingdong Chen, Yiteng Huang, and Israel Cohen. Pearson correlation coefficient. In Noise reduction in speech processing, pp. 1–4. Springer, 2009.

Yinzhi Cao and Junfeng Yang. Towards making systems forget with machine unlearning. In IEEE
S&P, 2015.

Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, et al. Extracting training data from large language models. In *USENIX Security*, 2021.

Hwan Chang and Hwanhee Lee. Which retain set matters for llm unlearning? a case study on entity unlearning. In *ACL Findings*, 2025.

Jiaao Chen and Diyi Yang. Unlearn what you want to forget: Efficient unlearning for llms. In EMNLP, 2023.

Michael Collins and Nigel Duffy. Convolution kernels for natural language. *Advances in neural* information processing systems, 14, 2001.

Aghyad Deeb and Fabien Roger. Do unlearning methods remove information from language model weights? *arXiv preprint arXiv:2410.08827*, 2024.

Ying Ding, Gobinda Chowdhury, Schubert Foo, et al. Template mining for the extraction of citation from digital documents. In *Proceedings of the Second Asian Digital Library Conference, Taiwan*, pp. 47–62, 1999.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv, 2024.

Ronen Eldan and Mark Russinovich. Who's harry potter? approximate unlearning in llms. *arXiv* preprint arXiv:2310.02238, 2023.

Kadrey v. Meta Platforms, Inc. 3:23-cv-03417, 2023. Tremblay v. OpenAI, Inc. 23-cv-03416-AMO, (N.D. Cal.), 2023. Michael M Grynbaum and Ryan Mac. The times sues openai and microsoft over ai use of copyrighted work. *The New York Times*, 27, 2023.

Chuan Guo, Tom Goldstein, Awni Hannun, and Laurens Van Der Maaten. Certified data removal from machine learning models. In *ICML*, 2020.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. In NeurIPS, 2014.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. In *ICLR*, 2022.

Shengyuan Hu, Yiwei Fu, Steven Wu, and Virginia Smith. Unlearning or obfuscating? jogging the memory of unlearned LLMs via benign relearning. In *ICLR*, 2025a.

Shengyuan Hu, Neil Kale, Pratiksha Thaker, Yiwei Fu, Steven Wu, and Virginia Smith. Blur: A
benchmark for llm unlearning robust to forget-retain overlap. *arXiv preprint arXiv:2506.15699*, 2025b.

Joel Jang, Dongkeun Yoon, Sohee Yang, Sungmin Cha, Moontae Lee, Lajanugen Logeswaran, and Minjoon Seo. Knowledge unlearning for mitigating privacy risks in language models. In ACL,
2023.

Wonje Jeung, Sangyeon Yoon, Hyesoo Hong, Soeun Kim, Seungju Han, Youngjae Yu, and Albert No. Dusk: Do not unlearn shared knowledge. *arXiv preprint arXiv:2505.15209*, 2025a.

Wonje Jeung, Sangyeon Yoon, and Albert No. Seps: A separability measure for robust unlearning in llms. In *EMNLP*, 2025b.

Jinghan Jia, Yihua Zhang, Yimeng Zhang, Jiancheng Liu, Bharat Runwal, James Diffenderfer, Bhavya Kailkhura, and Sijia Liu. Soul: Unlocking the power of second-order optimization for llm unlearning. In *EMNLP*, 2024.

Zhuoran Jin, Pengfei Cao, Chenhao Wang, Zhitao He, Hongbang Yuan, Jiachun Li, Yubo Chen, Kang Liu, and Jun Zhao. Rwku: Benchmarking real-world knowledge unlearning for large language models. In *NeurIPS Track Datasets and Benchmarks*, 2024.

Meghdad Kurmanji, Peter Triantafillou, Jamie Hayes, and Eleni Triantafillou. Towards unbounded machine unlearning. In *NeurIPS*, 2023.

Nathaniel Li, Alexander Pan, Anjali Gopal, Summer Yue, Daniel Berrios, Alice Gatti, Justin D Li, Ann-Kathrin Dombrowski, Shashwat Goel, Long Phan, et al. The wmdp benchmark: Measuring and reducing malicious use with unlearning. In *ICML*, 2024.

Sijia Liu, Yuanshun Yao, Jinghan Jia, Stephen Casper, Nathalie Baracaldo, Peter Hase, Yuguang Yao, Chris Yuhao Liu, Xiaojun Xu, Hang Li, et al. Rethinking machine unlearning for large language models. *Nature Machine Intelligence*, 2025.

Jakub Łucki, Boyi Wei, Yangsibo Huang, Peter Henderson, Florian Tramer, and Javier Rando. An `
adversarial perspective on machine unlearning for ai safety. *Transactions on Machine Learning* Research, 2024.

Aengus Lynch, Phillip Guo, Aidan Ewart, Stephen Casper, and Dylan Hadfield-Menell. Eight methods to evaluate robust unlearning in llms. *arXiv preprint arXiv:2402.16835*, 2024.

Pratyush Maini, Zhili Feng, Avi Schwarzschild, Zachary C Lipton, and J Zico Kolter. Tofu: A task of fictitious unlearning for llms. In COLM, 2024.

Milad Nasr, Nicholas Carlini, Jonathan Hayase, Matthew Jagielski, A Feder Cooper, Daphne Ippolito, Christopher A Choquette-Choo, Eric Wallace, Florian Tramer, and Katherine Lee. Scalable ` extraction of training data from (production) language models. In ICLR, 2025.

Martin Pawelczyk, Seth Neel, and Himabindu Lakkaraju. In-context unlearning: Language models as few shot unlearners. In *ICML*, 2024.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In NeurIPS, 2023.

Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bertnetworks. In *EMNLP*, 2019.

Weijia Shi, Jaechan Lee, Yangsibo Huang, Sadhika Malladi, Jieyu Zhao, Ari Holtzman, Daogao Liu, Luke Zettlemoyer, Noah A Smith, and Chiyuan Zhang. Muse: Machine unlearning six-way evaluation for language models. In *ICLR*, 2025.

Rishub Tamirisa, Bhrugu Bharathi, Andy Zhou, and Mantas Mazeika, Bo Li. Toward robust unlearning for llms. In *ICLR Workshop (SeTLLM)*, 2024.

Ayush K Tarun, Vikram S Chundawat, Murari Mandal, and Mohan Kankanhalli. Fast yet effective machine unlearning. *IEEE Transactions on Neural Networks and Learning Systems*, 2023.

Pratiksha Thaker, Yash Maurya, Shengyuan Hu, Zhiwei Steven Wu, and Virginia Smith. Guardrail baselines for unlearning in llms. In *ICLR Workshop (SeTLLM)*, 2024.

Pratiksha Thaker, Shengyuan Hu, Neil Kale, Yash Maurya, Zhiwei Steven Wu, and Virginia Smith.

Position: Llm unlearning benchmarks are weak measures of progress. In IEEE Conference on SaTML, 2025.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*, 2023.

Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Kashif Rasul, Younes Belkada, Shengyi Huang, Leandro von Werra, Clementine Fourrier, Nathan Habib, Nathan Sarrazin, Omar ´ Sanseviero, Alexander M. Rush, and Thomas Wolf. Zephyr: Direct distillation of lm alignment. In *COLM*, 2024.

Paul Voigt and Axel Von dem Bussche. The EU General Data Protection Regulation (GDPR): A
Practical Guide. Springer Publishing Company, Incorporated, 2017.

Boyi Wei, Weijia Shi, Yangsibo Huang, Noah A. Smith, Chiyuan Zhang, Luke Zettlemoyer, Kai Li, and Peter Henderson. Evaluating copyright takedown methods for language models. In *NeurIPS* Track Datasets and Benchmarks, 2024.

Xiaoyu Xu, Xiang Yue, Yang Liu, Qingqing Ye, Huadi Zheng, Peizhao Hu, Minxin Du, and Haibo Hu. Unlearning isn't deletion: Investigating reversibility of machine unlearning in llms. *arXiv* preprint arXiv:2505.16831, 2025.

Yuanshun Yao, Xiaojun Xu, and Yang Liu. Large language model unlearning. In *NeurIPS*, 2024.

Sangyeon Yoon, Wonje Jeung, and Albert No. R-tofu: Unlearning in large reasoning models. In EMNLP, 2025.

Xiaojian Yuan, Tianyu Pang, Chao Du, Kejiang Chen, Weiming Zhang, and Min Lin. A closer look at machine unlearning for large language models. In ICLR, 2025.

Ruiqi Zhang, Licong Lin, Yu Bai, and Song Mei. Negative preference optimization: From catastrophic collapse to effective unlearning. In COLM, 2024a.

Shengnan Zhang, Yan Hu, and Guangrong Bian. Research on string similarity algorithm based on levenshtein distance. In *IAEAC*, 2017.

Zhexin Zhang, Junxiao Yang, Pei Ke, Shiyao Cui, Chujie Zheng, Hongning Wang, and Minlie Huang. Safe unlearning: A surprisingly effective and generalizable solution to defend against jailbreak attacks. *arXiv preprint arXiv:2407.02855*, 2024b.