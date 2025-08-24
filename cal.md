
# Counterfactual Attention Learning for Fine-Grained Visual Categorization and Re-identification

## Abstract

Attention mechanism has demonstrated great potential in fine-grained visual recognition tasks. In this paper, we present a counterfactual attention learning method to learn more effective attention based on causal inference. Unlike most existing methods that learn visual attention based on conventional likelihood, we propose to learn the attention with counterfactual causality, which provides a tool to measure the attention quality and a powerful supervisory signal to guide the learning process. Specifically, we analyze the effect of the learned visual attention on network prediction through counterfactual intervention and maximize the effect to encourage the network to learn more useful attention for fine-grained image recognition. Empirically, we evaluate our method on a wide range of fine-grained recognition tasks where attention plays a crucial role, including fine-grained image categorization, person re-identification, and vehicle re-identification. The consistent improvement on all benchmarks demonstrates the effectiveness of our method. Code is available at \url{https://github.com/raoyongming/CAL}.

## Introduction

Despite the widespread use, the problem of how to learn effective attention is still barely studied. Most existing methods learn the visual attention in a weakly-supervised manner, \textit{i.e.}, the attention modules are simply supervised by the final loss function,   without a powerful supervisory signal to guide the training process. This likelihood based approach only explicitly supervises the final prediction (*e.g.*, class probabilities  for classification task) but ignores the causality between the prediction and attention. Previous methods also did not teach the machine how to distinguish between the main clues and biased clues. For example, if most training samples of one specific class appear with sky as background, then the attention model may be very likely to treat the sky as a discriminative region. Although these biased clues may also be beneficial to the classification on the current datasets, the attention model should only focus on the discriminative patterns, *i.e.* the main clues. Besides, directly learning from data may encourage the model to only focus on some certain attributes of the objects instead of all attributes, which may limit the generalization ability on test set. Therefore, we argue that this attention learning scheme is sub-optimal, where the effectiveness of the learned attentions is not always guaranteed, and the attention may lack discriminative power, clear meaning and robustness. As shown in Figure~\ref{fig1}, misleading and scattered attentions can still be observed from a well-trained attention model and potentially lead to the wrong predictions. To better understand this phenomenon, we analyze the statistics of both intrinsic attributes and external environments on the CUB dataset (see Figure~\ref{fig1-2}), where we use the attributes provided by the dataset and manually collect the environment statistics. We see there are biases for both attributes and environment, which indicates either background and single part are not reliable clues for classification. Therefore, it is desired to design new attention learning method beyond conventional likelihood maximization to mitigate the effects of data biases.     

Because of the lack of effective tool to evaluate the quality of attentions quantitatively, correcting misleading attentions is a very challenging task. One straightforward solution is to use extra annotations like bounding boxes or segmentation masks to obtain the regions of interest explicitly. However, this kind of method requires considerable cost of human labor and is hard to scale up.  Considering the critical role that attention plays in fine-grained visual recognition tasks, it is necessary to design a method to measure the quality of attentions without additional human supervision and further optimize the learned visual attentions. 

In this paper, we present a *counterfactual attention learning* (CAL) method to enhance attention learning based on causal inference. Specifically, we design a tool to analyze the effects of learned visual attention with counterfactual causality. The basic idea is to quantitate the quality of attentions by comparing the effects of facts (\textit{i.e.}, the learned attentions) and the counterfactuals  (\textit{i.e.}, uncorrected attentions) on the final prediction (\textit{i.e.}, the classification score). Then, we propose to maximize the difference (*i.e.*, *effect* in causal inference literature) to encourage the network to learn more effective visual attentions and reduce the effects of biased training set.

The proposed method is model-agonistic and thus can serve as a plug-and-play module to improve a wide range of visual attention models. Our method is also computational efficient, which only introduces a little extra computation cost during training and brings no computation during inference while can significantly improve attention models. We evaluate our method on three fine-grained visual recognition tasks including fine-grained image categorization (CUB200-2011, Stanford Cars and FGVC Aircraft), person re-identification (Market1501, DukeMTMC-ReID and MSMT17) and vehicle re-identification (Veri-776 and VehicleID). By applying our method to a multi-head attention baseline model, we demonstrate our method significantly improves the baseline and achieve state-of-the-art results on all benchmarks. 
 
## Related Work

**Fine-Grained Visual Recognition.** Attention mechanism plays an irreplaceable role in fine-grained visual recognition tasks. For example, in fine-grained image categorization task, Sermanet~\textit{et al.} pioneer adopting attention mechanism in fine-grained recognition problem and propose a RNN model to learn visual attention. Liu~\textit{et al.} extend the idea and employ a reinforcement learning scheme to obtain visual attentions. The subsequent studies such as MA-CNN, MAMC and WS-DAN further improve this line of methods and design attention models in a bottom-up manner, which achieve very promising results on fine-grained recognition benchmarks. Attention models have also proven to be effective in person/vehicle re-identification problem to handle the image matching misalignment challenge and improve the discriminative power of CNN features.  For instance, Liu~*et al.* and Lan~*et al.* employ the attention models to locate the discriminative salient regions in images to improve person re-identification. Xu~*et al.* and Zhao~*et al.* design a body part detector to employ the structure of the human body structure in the attention model. Another group of methods adopts attention mechanism on video-based person re-identification task to discover key parts in videos. Khorramshahi~*et al.* propose an adaptive attention model and significantly improve the state-of-the-art of vehicle re-identification task.

**Causal Reasoning in Vision.** The interest in combining the idea of deep learning and causal reasoning is growing rapidly in recent years. The tool of causality analysis has been successfully used in several areas, including explainable machine learning, fairness, natural language processing, reinforcement learning and adversarial learning. Some efforts also used causality as an effective tool to alleviate the effects of dataset bias in vision tasks, including  image classification, scene graph generation and visual commonsense reasoning. In this work, we study causality in the context of visual attention models, which is a new direction that has not been visited.

## Approach

### Attention Models for Fine-Grained Recognition

We begin by reviewing the attention models for fine-grained visual recognition, on which our method is built. Given an image $I$ and the corresponding CNN feature maps $\mathbf{X} = f(I)$ of size $H \times W \times C$, visual spatial attention model $\mathcal{M}$ aims to discover the discriminative regions of the image and improve CNN feature maps $\mathbf{X}$ by explicitly incorporating structural knowledge of objects. Note that although some of previous methods like propose to equip the backbone network with spatial attention modules, here we follow the mainstreams that learn basic feature maps and attentions separately. Previous studies have demonstrated that this design is more flexible and generic thanks to its model-agnostic nature. 

There have been quite a few variants of $\mathcal{M}$, and we can roughly categorize them into two groups. The first type aims to learn ``hard`` attention maps, where each attention can be represented as a bounding box or segmentation mask that covers a certain region of interest. This group of methods is usually closely related to object detection and semantic segmentation methods. Examples include recurrent visual attention model and fully convolutional attention network. Different from hard-attention models, a wider range of attention models are based on learning ``soft'' attention maps, which are more easy to optimize. In this paper, we focus on studying this group of methods. Specifically, our baseline model adopts the multi-head attention module used in. The attention model is designed to learn the spatial distributions of object's parts, which can be represented as attention maps $\mathbf{A} \in \mathbb{R_+}^{H \times W \times M}$, where $M$ is the number of attentions. Using the attention model $\mathcal{M}$, attention maps can be computed by:
$$
    \mathbf{A} \!\!=\!\! \{\mathbf{A}_1, \mathbf{A}_2, ..., \mathbf{A}_M\} \!\!=\!\! \mathcal{M}(\mathbf{X}),
$$

where $\mathbf{A}_i \in \mathbb{R_+}^{H \times W}$ is the attention map covering a certain part, such as the wing of a bird or the cloth of a person. The attention model $\mathcal{M}$ is implemented using a 2D convolutional layer followed by ReLU activation. The attention maps then are used to softly weight the feature maps and aggregate by global average pooling operation $\varphi$: 

$$ \tag{eq2} 
\mathbf{h}_i \!\!=\!\! \varphi(\mathbf{X} * \mathbf{A}_i) \!\!=\!\! \frac{1}{HW} \sum_{h=1}^{H}\sum_{w=1}^{W} \mathbf{X}^{h,w}\mathbf{A}_i^{h,w},   
$$

where $*$ denotes element-wise multiplication for two tensors. Following the practice in, we summarize the representation of different parts to form the global representation $\mathbf{h}$ :

$$ \tag{eq3}
\mathbf{h} \!\!=\!\! \texttt{normalize}([\mathbf{h}_1, \mathbf{h}_2, ..., \mathbf{h}_M]),   
$$

where we concatenate these representations and normalize the summarized representation to control its scale. The final representation $\mathbf{h}$ can be fed into a classifier (*e.g.*, fully connected layer) for image classification task or a distance metric (*e.g.*, Euclidean distance) for image retrieval task. The overall framework of our baseline attention model is illustrated in Figure~\ref{fig2}. 

### Attention Models in Causal Graph

Before we show our counterfactual method, we first introduce how to reformulate the above model in the language of causal graph.  Causal graph is also known as structural causal model, which is a directed acyclic graph $\mathcal{G} \!\!=\!\! \{\mathcal{N}, \mathcal{E}\}$. Each variable in the model has a corresponding node in $\mathcal{N}$ while the causal links $\mathcal{E}$ describe how these variable interact with each other. As presented in Figure~\ref{fig2}, we can use nodes in the causal graph to represent variables in the attention model, including the CNN feature maps (or the input image) $X$, the learned attention maps $A$ and the final prediction $Y$. The link $X \to A$ represents that the attention model takes as input the CNN feature maps and produces corresponding attention maps. $(X, A) \to Y$ indicates the feature maps and attention maps jointly determine the final prediction. Causal relations between nodes are encoded in the links $\mathcal{E}$, where we call node $X$ is the causal parent of $A$, and $Y$ is the causal child of $X$ and $A$. Note that since we do not impose any constraints on the network architecture of backbone models and the implementation details of the attention model, the causal graph can also represent many other attention models. Therefore, our method is model-agonistic and thus can also be extended to a wider range of attention learning problems.

### Counterfactual Attention Learning

Conventional likelihood methods optimize the attention by only supervising the final prediction $Y$ and regard the model as a black box, which ignores how the learned attention maps affect the prediction. On the contrary, causal inference provides a tool to help us think out of the black box by analyzing the causalities between variables. Therefore, we propose to employ the causalities to measure the quality of the learned attention and then improve the model by encourage the network to produce more influential attention maps. 

By introducing the causal graph, we can analyze causalities by directly manipulate the values of several variable and see the effect. Formally, the operation is termed *intervention* in causal inference literature, which can be denoted as $do(\cdot)$. When we want to investigate the effect of a variable, the intervention operation is performed by wiping out all the in-coming links of the variable and assigning a certain value to the variable. For example, $do(A\!\!=\!\!\bar{\mathbf{A}})$ in our causal graph means we demand the variable $A$ to take the value $\bar{\mathbf{A}}$ and cut-off the link $X \to A$ to force the variable to no longer be caused by its causal parent $X$. 

Inspired by causal inference methods, we propose to adopt *counterfactual intervention* to investigate the effects of the learned visual attention. The counterfactual intervention is achieved by an imaginary intervention altering the state of the variables assumed to be different. In our case, we conduct counterfactual intervention $do(A\!\!=\!\!\bar{\mathbf{A}})$ by imagining non-existent attention maps $\bar{\mathbf{A}}$ to replace the learned attention maps and keeping the feature maps $X$ unchanged. We can obtain the final prediction $Y$ after the intervention $A\!\!=\!\!\bar{\mathbf{A}}$ according to (\ref{eq2}) and~(\ref{eq3}):

$$
 Y(do(A\!\!=\!\!\bar{\mathbf{A}}), X\!\!=\!\!\mathbf{X}) \!\!=\!\!  \mathcal{C}([\varphi(\mathbf{X}\!\!*\!\!\bar{\mathbf{A}}_1), ..., \varphi(\mathbf{X}\!\!*\!\!\bar{\mathbf{A}}_M)]),
$$

where $\mathcal{C}$ is the classifier. In practice, we can use random attention, uniform attention or reversed attention as the counterfactuals. Evaluation on these options can be found in Section~\ref{sec:ablation}. 
 
Following, the actual effect of the learned attention on the prediction can be represented by the difference between the observed prediction $Y(A=\mathbf{A}, X=\mathbf{X})$ and its counterfactual alternative $Y(do(A=\bar{\mathbf{A}}), X=\mathbf{X})$:

$$ \tag{eq\_define}
Y_\text{effect}\!\!=\!\! \mathbb{E}_{\bar{\mathbf{A}} \sim \gamma} [ Y(A\!\!=\!\!\mathbf{A}, X\!\!=\!\!\mathbf{X}) \!\!-\!\! Y(do(A\!\!=\!\!\bar{\mathbf{A}}), X\!\!=\!\!\mathbf{X})],
$$

where we denote the effect on the prediction as $Y_\text{effect}$ and $\gamma$ is the distribution of counterfactual attentions. Intuitively, the effectiveness of an attention can be interpreted as how the attention improves the final prediction compared to wrong attentions. Therefore, we can use $Y_\text{effect}$ to measure the quality of a learned attention. 
 
Furthermore, we can use the metric of attention quality as a supervision signal to explicitly guide the attention learning process. The new objective can be formulated as:

$$
\mathcal{L} \!\!=\!\! \mathcal{L}_{ce}(Y_\text{effect}, y) \!\!+\!\!  \mathcal{L}_\text{others},
$$

where $y$ is the classification label,  $\mathcal{L}_{ce}$ is the cross-entropy loss, and $ \mathcal{L}_\text{others}$ represents the original objective such as standard classification loss. By optimizing the new objective, what we expect to achieve is two-fold: 1) the  attention model should improve the prediction based on wrong attentions as much as possible, which encourages the attention to discover the most discriminative regions and avoid sub-optimal results; 2) we penalize the prediction based on wrong attentions, which forces the classifier to make decision based more on the main clues instead of the biased clues and reduces the influence of biased training set. 

Note that in practice, it is not necessary to compute the expectation in Equation~(\ref{eq\_define}) and we only sample a counterfactual attention for each observed attention during training, which is also consistent with the idea of stochastic gradient descent. Therefore, the extra computational cost introduced by our method is an additional forward of the attention model and the classifier, which is very lightweight compared with the CNN backbone. Besides, our method introduces no additional computation during inference. 

## Experiments

We assess the effectiveness of our proposed counterfactual attention learning method on several fine-grained visual recognition tasks including fine-grained image categorization, person re-identification and vehicle re-identification. We take the conventional spatial attention as the baseline and compare our counterfactual attention learning method with the baseline method and other state-of-the-art methods. The experimental settings, implementation details and results for different tasks are described below. 

### Fine-grained Image Categorization

Fine-grained visual categorization focuses on classifying the subordinate-level classes under a fixed basic-level category, such as species of bird, types of car and types of aircraft. The objects under the same basic-level category are always high structured and with low inter-class variances. Thus, attention is effective to look for the key difference in  detail and discover the discriminative regions. 

**Datasets and Experimental Settings.** We conducted experiments on widely used CUB200-2011, Stanford Cars and  Aircraft datasets for fine-grained bird, car and aircraft classification. CUB200-2011 is composed of 5,994 training images and 5,794 testing images from 200 species of birds.  Stanford Cars contains 16,185 images of cars from 196 different types and among all collected, 8,144 images are used for training and 8,041 images for testing. FGVC Aircraft consists of 10,000 images of 100 fine-grained aircraft types. Following previous methods, we use 2/3 images for training and 1/3 images for evaluation. 

**Implementation Details.**  We adopted the standard ResNet-101 as the backbone network. For attention model, we set the number of attentions to 32 and use the weakly supervised data augmentation method as suggested by. During inference, we use multiple crops and horizontal flipping to boost performance. All experiments are conducted with the same hyper-parameters, including 16 batch size, 448$\times$448 image size, and 1e-5 weight decay. We use 1e-3 initial learning rate and reduce the learning rate by 0.9 times in every 2 epochs. 

**Results.**  We compared our method with the baseline attention model and the state-of-the-art methods in Table~\ref{tb:fg}. The proposed counterfactual attention method can improve the strong baseline by 1.3\%, 1.5\% and 0.6\% on CUB200-2011, Stanford Cars and Aircraft, respectively. Our method also outperformed previous state-of-the-art methods. Notably, although a stronger backbone (DenseNet-161) is used in recent API-Net method, our method can still achieve better performance on all three benchmarks. These results clearly demonstrates the effectiveness of our method.

### Person Re-identification

Person re-identification (ReID) is a task to match the query individual from multiple gallery candidates across the non-overlapping camera views. It is a challenging problems because of the intra-class variances due to illumination changes, pose variations, occlusions, and cluttered backgrounds. Attention model has gained great success for person ReID by handling the matching misalignment challenge and enhancing the feature representation. 

**Datasets and Experimental Settings.** We conducted the experiments on three public person re-identification datasets including Market1501, DukeMTMC-reID and MSMT17. Market1501 consists of 32,668 images of 1,501 identities detected by 6 cameras. The whole dataset is divided into a training set with 12,968 images of 751 identities and a test set containing 3,368 query images and 19,732 gallery images of 750 identities. DukeMTMC-reID dataset is composed of 1,404 persons captured by 8 cameras. Its training set includes 16,522 images of 702 persons, while the test set contains the remaining 702 persons with 2,228 query images and 17,661 gallery images. MSMT17 is one of the largest ReID datasets which contains 4,101 identities and 126,411 images. The training set is composed of 30,248 images of 1,041 identities, while the remaining 3,060 identities are used for testing. 

We followed the settings of  and \cite{chen2019self} for Market1501 and DukeMTMC-reID datasets, and chose the single-query manner to validate our method. For MSMT17, we followed the settings in the test settings in and. We employed the cumulative matching characteristic (CMC) curve and mean average precision (mAP) as the evaluation metrics.

**Implementation Details.** We adopted two basic network structures including a standard ResNet50 backbone and the backbone network in SCAL which adds 4 channel attention blocks. We applied the modification backbone due to its competitive performance and relatively concise structure, since recent state-of-the-art methods usually use stronger or more sophisticated backbone such as part model to improve performance. We used $\dagger$ to indicate the models using the modified backbone. The baseline attention block is same as the one in fine-grained categorization task and we set $M$ to 8 for re-identification models. All experiments are conducted with the same hyper-parameters including 80 batch size, $384\times192$ image size, and 2e-4 learning rate. The data augmentation methods includes random cropping, erasing and horizontal flipping. We trained the network for 160 epochs with triplet loss and softmax loss with learning rate reducing by 10 times in every 40 epochs.

**Results.** As shown in Table~\ref{comparison_reid}, we observed that our CAL methods achieve consistent improvement for different baselines on all benchmarks. Specifically, compared with the strong baseline we obtained 0.6\%/0.5\% Rank-1/mAP improvement on the Market1501 dataset, 1.6\%/2.3\% on the DukeMTMC-ReID dataset, and 2.8\%/4.7\% on the MSMT17 dataset. The improvement on the MSMT17 dataset is larger than other two datasets, since images in MSMT17 have larger intra-class variances. Besides, with our strong attention model, we can achieve the SOTA performance on DukeMTMC-ReID and MSMT17 datasets.

### Vehicle Re-identification

Vehicle Re-Identification (ReID) aims to retrieve all images of a given query vehicle from a large image database, without the license plate clues. The vehicles with different identities can be of the same make, model and color, while the vehicle appearances of the same identity always vary significantly across different viewpoints. Attention model can be applied for matching the key similarity of vehicle images across different viewpoints.

**Datasets and Experimental Settings.** We conducted the experiments on two widely used vehicle datasets including Veri-776 and VehicleID. Veri-776 dataset contains over 50,000 images from 776 vehicle IDs, where 37,778 images from 576 IDs are split for training and the rest 200 IDs are used for testing. VehicleID is composed of 110,178 images of 13,134 vehicles for training and 111,585 images of 13,133 IDs for testing. Following the experimental settings in and , we report the testing results for three subsets including small size subset with 800 vehicles, medium size subset with 1,600 vehicles and large subset with 2,400 vehicles. We employed the CMC curve and mAP as the evaluation metrics for vehicle ReID task. 

**Implementation Details.**  We applied the ResNet50 backbone and the same attention block ($M\!\!=\!\!8$) as the baseline. The hyper-parameters are also fixed for the baseline and our method. The loss functions and data augmentation methods are same with person ReID task. We selected 256 samples in a batch with $256\times256$ image size. The initial learning rate is 2e-4 and reducing 10 times in 8000th and 18000th iterations. We trained the network for total 28000 iterations.

**Results.** We compared the performance of CAL with the baseline attention learning method and other SOTA methods. As shown in Table~\ref{comparison_vehicle}, we obtained 0.9\%/2.3\% Rank-1/mAP improvement on the Veri-776 dataset and 5.8\%/3.3\%/4.1\% Rank-1 improvement on small/medium/large test settings of the VehicleID dataset.  Note that we did not use any extra labels in the training precess, yet achieved the comparable performance with VAML which manually annotates the viewpoints of images to train the view-predictor.

### Analysis

We analyzed the influences and sensitivity of some major parameters. We conducted the parameters analysis experiments on three fine-grained visual recognition tasks.

**Effects of the type of counterfactual attention.** We investigated three different strategies to generate the counterfactual attention maps, namely random attention, uniform attention, reversed attention and shuffle attention  (see Supplementary Material for details). The results are presented in Table~\ref{tb:ca}. We see random attention, uniform attention and shuffle attention achieve similar performance while reversed attention fails to improve the baseline on CUB. We think it is because learning attention that is better than reversed attention is relatively easy and cannot provide an effective signal to supervise the attention. 

**Effects of the number of attentions.** The number of heads in attention model is an important hyper-parameter in our baseline model. Therefore, we search the best numbers of attention on CUB, Market1501 and Veri-776 for fine-grained image categorization, person re-identification and vehicle re-identification tasks respectively and directly use the searched hyper-parameters on other datasets. For a fair comparison, we use the same hyper-parameters in our models and the baseline models, and did not search the best hyper-parameters for our models separately. The results are presented in Figure~\ref{fig:att}. Based on these results, we set $M$ to 32 and 8 for fine-grained categorization and re-identification tasks, respectively.

**Quantitative analysis of attention.** To better verify the effectiveness of CAL, we compare our method with other three kinds of attention regularization strategies including attention drop, entropy regularization and attention normalization (see Supplementary Material for details) and evaluated the quality of the learned attention maps by computing the mean IoU between the rectangular region that covers the high score attentions and the ground-truth object bounding boxes on CUB. The results can be found in Table~\ref{tb:iou}. We see only CAL is effective to simultaneously improve classification accuracy and attention quantitative. Both attention Dropout and Entropy regularization will slightly degrade the final performance under the both metrics. Attention normalization will significantly hurt the performance.

**Results of single-head attention models.** To show the generality for different attention models, we also test CAL on single-head attention models (*i.e.*, baseline models with $M\!\!=\!\!1$). The results are presented in Table~\ref{tb1}. We see our method can consistently and more significantly improve the relative weak baseline models, which clearly shows our method is suitable for various attention models. 

### Visualization

To have an intuitive understanding of our counterfactual attention learning method, we compare the attention maps of our models and the baselines models on CUB200-2011, Stanford Cars  and FGVC-Aircraft datasets. The visual results are displayed in Figure~\ref{fig:compare2}. We see our method helps the attention models make correct predictions by reducing the misleading and scatter attentions. For example, in the first example of the Stanford Cars dataset, the attention with our CAL method avoids the reflection on the ground. Besides, CAL encourages the model to focus on the main clues for classification and explore more discriminative regions. Taking the Eared Grebe in the CUB200-2011 dataset as example, our attention focuses on the discriminative buttocks region to recognize it. While for the second example of cars and the first one of aircrafts, our attention models tend to explore more discriminative regions such as the rearview mirror and wheel respectively.

## Conclusion

In this paper, we have presented a counterfactual attention learning method to learn more effective attention based on causal inference. We designed a framework to quantitate the quality of attentions by comparing the effects of facts and the counterfactuals on the final prediction. We also proposed to maximize the difference to encourage the network to learn more effective visual attentions. Our method only brings negligible extra cost during training and introduce no cost during inference. 

CAL is a model-agnostic framework to enhance attention learning and mitigate the effects of dataset bias, which can be applied to various fine-grained visual recognition tasks. 

We conducted extensive experiments on three fine-grained visual recognition tasks and demonstrated state-of-the-art performance on all benchmarks.

## Acknowledgements

This work was supported in part by the National Key Research and Development Program of China under Grant 2018AAA0102803, in part by the National Natural Science Foundation of China under Grant 61822603, Grant U1813218, and Grant U1713214, in part by a grant from the Beijing Academy of Artificial Intelligence (BAAI), and in part by a grant from the Institute for Guo Qiang, Tsinghua University.

## A. More Visual Results

To have an intuitive understanding of our counterfactual attention learning method, we compare the attention maps of our models and the baselines models on CUB, Stanford Cars  and Aircraft datasets. The more visual results are presented in Figure~\ref{vis}. We see our method helps the attention models make correct predictions by 1) reducing the misleading and scatter attentions and 2) encouraging the model to focus on the main clues for classification and explore more discriminative regions. 

## B. More Implementation Details

**Different types of counterfactual attentions.** We compared four different counterfactual attentions in our experiments. The details about how to generate them are described as follows. 

- **Random Attention.** We use randomly generated attention maps as the counterfactual attentions. The attention value for each location is sampled from a uniform distribution $\mathcal{U}(0,2)$.
- **Uniform Attention.** We simply set the attention value for each location to the average value of the real attention maps.
- **Reversed Attention.** We reverse the attention maps by subtracting the original attention from the maximal attention value of each sample.
- **Shuffle Attention.** We randomly shuffle the attention maps along the batch dimension.

**Attention Regularization Strategy.** We investigated several regularization strategies on the baseline attention model to verify the effectiveness of our method. The details about these  regularization strategies are described as follows.

- **Attention Dropout.** We apply the Dropout method to the attention maps.
- **Entropy Regularization.** We add an extra term to the loss function to maximize the entropy of the attention maps.
- **Attention Normalization.** We add $\ell_2$ normalization to the attention maps.