# BadMoE: Backdooring Mixture-of-Experts LLMs via Optimizing Routing Triggers and Infecting Dormant Experts

- Avg Score: 5.50
- Decision: Reject
- Scores: 4, 6, 6, 6

## Abstract
Mixture-of-Experts (MoE) architectures are rapidly becoming the standard for building scalable, 
efficient large language models (LLMs). Their open availability, however, exposes them to supply-chain backdoor attacks, where an adversary can modify a checkpoint and redistribute a poisoned version. MoE’s intrinsic sparsity further amplifies this risk, as small changes in activated experts may disproportionately influence the model’s output. In this work, we propose BadMoE, a novel backdoor attack that exploits the overlooked structural vulnerabilities introduced by expert sparsity and routing. We first provide theoretical intuition that the MoE output can be governed by dominating experts. Guided by this insight, BadMoE poisons underutilized (``dormant'') experts and utilizes routing-aware triggers to activate them, enabling stealthy and effective manipulation. 
 Specifically, BadMoE involves three steps: 1) identifying dormant experts unrelated to the target task, 2) optimizing a routing-aware trigger toward these experts, and 3) promoting them to dominating roles through training data. Extensive experiments on three MoE LLMs across multiple backdoor tasks show that BadMoE, using only two injected experts, can reliably control outputs, outperform existing attacks, and evade current defenses. 
By leveraging architectural sparsity and dynamic usage profiling, our approach uncovers backdoor vulnerabilities in MoE LLMs that are overlooked by traditional attacks, highlighting potential security risks in emerging sparse architectures.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces BADMOE, a backdoor attack targeting Mixture of Experts MoE LLMs, and reveals a new vulnerability in expert routing. It proves the existence of dominating experts that can dictate outputs and exploit this by poisoning low-usage experts and designing routing aware triggers to activate them. BADMOE can achieve up to 100% attack success with only two infected experts while preserving benign performance and evading common defenses. The results expose that sparse expert activation enables stealthy and robust backdoors in MoE models and motivate the development of fundamentally new defences beyond conventional parameter and data-centric methods.

### Strengths
- The technical design of BADMOE is grounded in both theory and practice, combining a formal proof of expert dominance with a well-structured three-stage attack pipeline and evaluation across multiple MoE architectures and tasks.
- The findings highlight a critical and previously overlooked vulnerability in a fast-adopting class of LLM architectures, emphasizing the need for new MoE-specific security defenses.
- The paper introduces a previously unexplored attack surface in Mixture-of-Experts LLMs by formulating the notion of dominating experts and demonstrating how their routing behavior can be exploited for backdoor injection.

### Weaknesses
- While the paper evaluates several existing defenses, it does not explore or analyze potential MoE-specific defensive strategies in depth. For instance, the discussion of dormant expert pruning is shallow and lacks a systematic defense design or evaluation.
- The evaluation primarily uses standard NLP benchmarks and tasks. The paper does not assess whether BADMOE can persist under domain adaptation or large-scale instruction tuning, which are common in practical LLM reuse scenarios.
- The attack assumes that adversaries can inject poisoned experts and release modified MoE checkpoints publicly, but the paper provides limited evidence of how feasible such manipulations are in real-world model supply chains or open-source ecosystems. Some public repositories have applied verification when users upload models.
- The paper does not demonstrate BADMOE in a complete deployment pipeline, e.g., a hosted API or plugin ecosystem, thus its persistence and exploitability under model updates or reinforcement fine-tuning remain unclear.

### Questions
1. Can BADMOE persist after instruction tuning, domain adaptation, or RLHF-style fine-tuning?
2. How sensitive is the attack to router retraining or expert replacement? Would re-initializing the router or randomizing expert selection mitigate BADMOE’s effect without heavy accuracy loss?
3. Could the authors include quantitative metrics that connect the theoretical dominance score (e.g., KL divergence) with observed ASR, to demonstrate a stronger causal link between theory and practice?
4. Could the authors expand the exploration of MoE-specific defenses beyond dormant expert pruning?

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
4

### Summary
This paper introduces a new backdoor attack targeting Mixture-of-Experts LLMs: BadMoe. Specifically, given a predetermined model layer $l$, they identified *dormant experts* (i.e., experts that are underutilized by the model on a given dataset). Then, using GCG, they find a trigger prompt that preferentially activates the previously identified dormant experts. Lastly, they jointly finetune the dormant experts and the non-MoE components of the model to (i) preserve utility when the trigger is not present and (ii) display the targeted behavior when the trigger is present.

They evaluate their method on different tasks and show that they outperform prior methods in ASR and utility preservation. They demonstrate that their method is robust to various practical scenarios and prior defenses. These results suggest that BadMoe poses a significant threat to MoE models and reveals a new attack vector on these architectures.

### Strengths
- The paper is well-written and the method is clearly presented. Numerous illustrations (Fig. 1 and Fig. 2) help the reader understand the attack and its components, which in turn makes diving into the details of the method easier. Furthermore, all the necessary preliminaries on MoE needed to understand the method are clearly presented in the paper.
- While the different components of their attack are not novel per se, combining them into a successful attack is non-trivial and represents a novel contribution, as it uncovers a new potential threat vector in MoE models.
- The empirical evaluation is extensive and comprehensive: several tasks are evaluated and multiple baselines are used for comparison. Moreover, most components of the method are ablated, and potential defenses as well as realistic deployment scenarios are evaluated.

### Weaknesses
- I think the utility evaluation is a bit sparse; using standard LLM benchmarks would improve it. More specifically, the dormant experts are selected because they are dormant on a specific task $\mathcal{D}$. I am wondering what would happen if I evaluate the model’s clean accuracy on a task $\mathcal{D}'$ for which the previously dormant experts are actually dominant (or at least often activated).
- I do not see the contributions of Theorem 4.1 from Section 4.2. First, I think the Gaussian assumption is unrealistic, given that prior works have shown outliers in the activation distribution have a significant impact on LLM behavior ([1]). Second, the components of the attacks do not specifically leverage the insights from Theorem 4.1. The loss in Equation (9) is a standard backdooring loss with regularization, and it turns out that the injected experts dominate.
- While the attack is clearly successful and robust, the results are mostly incrementally better than prior baselines (except on robustness to domain shift, where BadMoe shows a significant improvement compared to all baselines).

[1] Systematic Outliers in Large Language Models, An et al., ICLR 2025.

### Questions
- Could the authors evaluate BadMoe models on standard LLM benchmarks (e.g., MMLU, HumanEval, ...) that span a very wide range of tasks?
- Are dormant experts inactive for all tasks, or are there tasks for which they would be active? If such tasks exist, would BadMoe retain high clean accuracy on those tasks?
- Assume the attacker's trigger is $z$. What would happen if I optimize a new trigger $z' \neq z$ that also activates only the experts from $S_{\alpha}$? Would it activate the backdoor? If so, could optimizing a prompt to activate dormant experts in a model and then measuring accuracy be a targeted way to identify a BadMoe model?
- How can Theorem 4.1 help improve or guide the design of the method?

### Soundness
4

### Presentation
4

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
The paper describes a backdoor attack against mixture of experts LLMs. The idea is based on the existence of dominating experts, that can determine the output of the LLM. The backdoor attack is based on the idea of poisoning underutilized experts that are unrelated to the question, but can be activated with a routing-aware trigger. Experimental study shows that the proposed approach can reliably control outputs and evade current defenses with two injected experts.

### Strengths
* The paper shows a valid approach of backdoor attack against MOE LLMs that specifically relies on the internal structure of the MOE. 
* The experimental study shows that the proposed backdoor can successfully attack the chosen LLMs in a very large set of cases.

### Weaknesses
* The definition of a dominating expert (4.1) does not appear to take into consideration the other experts, which is an unusual definition for "dominating" something. 
* Given that in the definition of this paper, the experts are combined additively, and limits on the internal structure are not considered, it seems that Theorem 4.1 only says that you can always have a large enough output that it will be larger than the other models. This appears to be a very simple observation. 
* The step based on infecting dormant experts (4.5) appears to require training: all the parameters of the model outside the experts + the adversarial experts. It is not clear how much of the model is actually not trained. Also, the paper claim that in this procedure the theta_0 is trained to maintain normal model behavior and theta_a for dominating the target outputs - but actually nothing in the training objective implies this. The training objective in formula (9) is symmetrical in \theta_0 and \theta_a as well as in poisoned and clean data.

### Questions
* Given that in the specified scenario the attacker needs to have access to the whole LLM, what would be the advantages of this particular approach of identifying dormant MOE, and only modifying them - as opposed to modifying all the experts?

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
4

### Summary
This paper introduces a new backdoor attack that works by targeting specific experts in an MoE model, so that the poisoned backdoored behavior is activated only when some experts are active. This makes the attack much more stealthy and hard to defend against. The attacks themselves are highly effective with often near-100% attack success rates.

### Strengths
A new poisoning attack that's effective at poisoning MoE models. Because it targets the MoE specifically, the attack is much more robust to defenses that try to remove backdoors because the poisons can be added to experts that don't usually activate and so it's hard to know why the model is malicious.

High attack success rates---much higher than prior work. This is interesting in its own right, but I wonder if the baselines couldn't have been tuned to be somewhat stronger. Because poisoning is a tradeoff of utility-vs-asr I wonder what the full curve would look like.

Good evaluation both with and without defenses.

### Weaknesses
The paper is not very clear about why this work is interesting. The main introduction frames the work around being around MoE poisoning as something that hasn't been done, but is not very well motivated. This is true, but saying "this hasn't been done before" doesn't make a compelling paper. Most things haven't been done before. What's interesting about this attack is that, because the poisoning is done to rarely-activated experts, it's much harder to remove the poisoning via fine-tuning because these experts see very little gradient signal.

This directly ties in to my other concern with this work, though: there are probably simple defenses that would test each expert one-by-one in order to see if the backdoor is present, or defenses that make sure they've tested each expert one-by-one. The attack here is good motivation for doing things like this, but once you know that this is necessary the solutions are somewhat straightforward. This doesn't invalidate the utility of the attack, but it does make it less compelling if it's easily fixed.

My other concern with this work is that this attack works well for MoE models that have a small number of experts, but the recent trend (cf. deepseek) is to train many more experts and activate even fewer of them. Do the results still work in this setting?

### Questions
Could a defender implement techniques that test each expert independently?

### Soundness
3

### Presentation
2

### Contribution
2
