# GeoBA: Stealthy Geometric Poisoning on 3D Point Cloud

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Point cloud backdoor attacks exploit carefully crafted trigger patterns to manipulate deep neural networks (DNNs), causing misclassification when specific input patterns are encountered. Existing approaches primarily rely on (1) explicit trigger injection (e.g., adding a specific shape) or (2) basic geometric transformations (e.g., rotation, scaling) to generate poisoned samples. 
However, such trigger patterns are often easily detected by the human eye or statistical analysis, undermining the stealth and effectiveness of the attack.
To this end, we propose GeoBA, a stealthy geometric poisoning backdoor attack that embeds imperceptible yet robust triggers into point clouds with minimal geometric perturbation. Specifically, we first transform point clouds into a spherical domain, where subtle phase perturbations are applied to introduce the backdoor pattern while preserving the global geometric structure. This perturbation effectively induces the model to learn the trigger while avoiding noticeable shape deviations. A controlled inverse transformation then maps the poisoned samples back to the original space, ensuring their imperceptibility and robustness to existing defenses.
Experiments show that GeoBA consistently triggers backdoors across mainstream 3D architectures (e.g., Mamba3D, PointMLP), with excellent stealth, transferability, and robustness—highlighting overlooked security risks in geometric transformations. Excitingly, it only takes 4 lines of core code to achieve this. The code will be released promptly.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces GeoBA, a novel stealthy backdoor attack targeting 3D point cloud models. Unlike previous approaches that use visible geometric triggers (e.g., adding a ball pattern in PointBA) or global transformations (e.g., scaling in IRBA), GeoBA embeds imperceptible triggers by transforming point clouds into spherical coordinates and applying subtle sinusoidal phase perturbations to the azimuthal angle. This preserves the overall geometry while enabling high attack success rates (ASR >95%) across architectures like PointNet, DGCNN, and Mamba3D. Key contributions include: (1) a pioneering spherical-phase poisoning technique for balanced stealth and efficacy; (2) robustness to common defenses such as Statistical Outlier Removal (SOR) and jittering; (3) high efficiency (implementable in 4 lines of code); and (4) strong performance on synthetic (ModelNet40, ShapeNetPart) and real-world datasets (ScanObjectNN, KITTI), exposing vulnerabilities in 3D vision systems.

### Strengths
GeoBA introduces a pioneering approach by leveraging spherical coordinate transformations and subtle phase perturbations to embed backdoor triggers in an efficient way. The method is rigorously designed to preserve global geometric structure, resulting in minimal distortions measured by low Chamfer Distance and Hausdorff Distance. It achieves high Attack Success Rates across datasets like ModelNet40 and architectures like PointNet and Mamba3D while maintaining clean accuracy. Critically, GeoBA demonstrates superior resilience to defenses such as Statistical Outlier Removal (SOR), random rotations, and dropout, retaining ~81% ASR under combined attacks.

### Weaknesses
1. The number of tested defense methods is relatively limited, and they are mostly simple defense techniques. There is a lack of evaluation against frequency-domain defense and filtering methods, such as LPF-Defense and GFT Robustness. Could GeoBA's attack be substantially invalidated or weakened by these defense methods, or after applying corresponding data augmentations?
2. The selection of the target label lacks sufficient explanation and rationale. 
3. There are no experiments evaluating the impact on 3D object detection performance, which limits the practical application scenarios of the method.

### Questions
1. How the target label be chosen for different test dataset? Are there differences in results when selecting other target labels, or are there any regular patterns or conclusions?
2. How could this method be used in the realistic 3D object detection scene?

### Soundness
2

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
This paper presents GeoBA, a novel geometric poisoning backdoor for 3D point clouds. By perturbing spherical-phase components and inverting back to Cartesian coordinates, GeoBA claims to introduce imperceptible yet robust triggers that preserve global shape while reliably activating backdoors.

### Strengths
Backdoor attacks are of significance when the attack success rate (ASR) is already relatively high, e.g., an ASR of 70–80% can generally be considered a successful attack. Therefore, the key aspect to evaluate is the robustness of the attack. Beyond Tables 1 and 2, I believe the most notable results in this paper are presented in Tables 11 and 12, which is also the reason I rated this paper positively. I strongly recommend that the authors include Tables 11 and 12 in the main text.

### Weaknesses
1. The attack in this paper is based on altering labels. It remains unclear whether the proposed method can be applied to a clean-label setting.

2. Modern transformer-based models are widely used. It is still uncertain how this method performs across various other transformer-based architectures. In this paper, only PCT is tested.

### Questions
see Weaknesses

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
This paper proposes GeoBA, a novel backdoor attack method for 3D point cloud classifiers. Unlike existing attacks that use explicit triggers (e.g., adding a ball) or coarse global geometric transformations (e.g., rotation, scaling), GeoBA employs a more stealthy approach. Its core innovation is to transform point clouds from Cartesian (Euclidean) coordinates to spherical coordinates and then apply subtle, structured perturbations to the azimuthal angle. Extensive experiments on datasets like ModelNet40, ShapeNetPart, and ScanObjectNN show that GeoBA achieves a high Attack Success Rate (ASR >95%) across various model architectures (PointNet, DGCNN, PCT, etc.) while maintaining high Benign Accuracy (ACC).

### Strengths
- The primary strength is its stealth. By operating in the spherical domain and perturbing only the angular phase, GeoBA introduces minimal geometric distortion.
- The paper provides comprehensive evidence that GeoBA is highly resilient to a suite of standard data preprocessing defenses (SOR, dropout, jittering) and even advanced adaptive defenses like IF-Defense, where other attack methods fail catastrophically.
- The core poisoning algorithm is simple (4 lines of code) and parameter-free (unlike some baselines that require a surrogate network). It is also computationally very efficient, processing large point clouds much faster than competitors like IRBA.

### Weaknesses
- While the paper uses quantitative metrics (CD, HD) to prove stealth, it does not include a human subjective study to confirm that the perturbations are truly imperceptible to the human eye across all tested objects and viewing angles.
- The robustness is tested against existing defenses, but the paper does not discuss how future, specifically designed defenses might counter GeoBA, such as adapyive defenses.
- The paper does not deeply discuss the practical challenges an attacker might face, such as physical settings.

### Questions
Please see the weakness part.

### Soundness
2

### Presentation
3

### Contribution
2
