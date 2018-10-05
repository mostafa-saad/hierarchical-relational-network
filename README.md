## [Hierarchical relational network for group activity recognition and retrieval. Mostafa S. Ibrahim, Greg Mori.  European Conference on Computer Vision 2018](https://eccv2018.org/openaccess/content_ECCV_2018/papers/Mostafa_Ibrahim_Hierarchical_Relational_Networks_ECCV_2018_paper.pdf)

## Contents
0. [Abstract](#abstract)
0. [Model](#model)
0. [Extra Experiments - CAD](#experiments-cad)
0. [Code Scope & Requirements](#codescope)
0. [Data Format](#data)
0. [Installation](#installation)
0. [License and Citation](#license-and-citation)
0. [Poster and Powerpoint](#poster-and-powerpoint)




## Abstract
Modeling structured relationships between people in a scene is an important step toward visual understanding. We present a Hierarchical Relational Network that computes relational representations of people, given graph structures describing potential interactions. Each relational layer is fed individual person representations and a potential relationship graph.  Relational representations of each person are created based on their connections in this particular graph. We demonstrate the efficacy of this model by applying it in both supervised and unsupervised learning paradigms. First, given a video sequence of people doing a collective activity, the relational scene representation is utilized for multi-person activity recognition. Second, we propose a Relational Autoencoder model for unsupervised learning of features for action and scene retrieval. Finally, a Denoising Autoencoder variant is presented to infer missing people in the scene from their context. Empirical results demonstrate that this approach learns relational feature representations that can effectively discriminate person and group activity classes.

## Model
We put a lot of effort to make the model pictures speak about themselves, even without reading their text. You can get the whole paper idea from them.

<img src="https://raw.githubusercontent.com/mostafa-saad/hierarchical-relational-network/master/files/img/model_person_rel.png" alt="Figure 1" height="400" >

**Figure 1**: A single relational layer. The layer can process an arbitrary sized set of people from a scene, and produces new representations for these people that capture their relationships.  The input to the layer is a set of $K$ people and a graph $G^{\ell}$ encoding their relations.  In the relational layer, a shared neural network ($F^{\ell}$) maps each pair of person representations to a new representation that also encodes relationships between them.  These are aggregated over all edges emanating from a person node via summation.  This process results in a new, relational representation for each of the $K$ people. By stacking multiple relational layers, each with its own relationship graph $G^{\ell}$, we can encode hierarchical relationships for each person and learn a scene representation suitable for group activity recognition or retrieval


<img src="https://github.com/mostafa-saad/hierarchical-relational-network/raw/master/files/img/model_intro.png" alt="Figure 2" height="400" >

**Figure 2**: Relatinal unit for processing one person inside a relational layer.  The feature vector for a person (red) is combined with each of its neighbours'.  Resultant vectors are summed to create a new feature vector for the person (dark red).


<img src="https://github.com/mostafa-saad/hierarchical-relational-network/raw/master/files/img/model_recognition_main.png" alt="Figure 3" height="500" >

**Figure 3**: Our relational network for group activity recognition for a single video frame. Given $K$ people and their initial feature vectors, these vectors are fed to 3 stacked relational layers (of output sizes per person: 512, 256, 128). Each relational layer is associated with a graph $G^{\ell}$ (disjoint cliques in this example: layer 1 has 4 cliques, each of size 3; layer 3 is a complete graph). The shared MLP $F^{\ell}$ of each layer computes the representation of 2 neighbouring players. Pooling of the output $K$ feature vectors is used for group activity classification.

Another application in unsupervised areas:


<img src="https://github.com/mostafa-saad/hierarchical-relational-network/raw/master/files/img/model_retrieval_main.png" alt="Figure 4" height="500" >

**Figure 4**: Our relational autoencoder model. The relationship graph for this volleyball scene is 2 disjoint cliques, one for each team and fixed for all layers. $K$ input person feature vectors, each of length 4096, are fed to a 4-layer relational autoencoder (sizes 256-128-256-4096 ) to learn a compact representation of size 128 per person.



## Experiments - CAD
The Collective Activity Dataset (CAD) consists of 44 videos and five labels used as both person action and  group activity (crossing, walking, waiting, talking, and queueing). The majority activity in the scene defines the group activity. We followed the same data split, temporal window and implementation details as [25], including AlexNet features (due to the short time; faster to train/extract features).

We used a single relational layer with a simple graph: each 3 consecutive persons (spatially in horizontal dimension) are grouped as a clique. The layer maps a person of size 4096 to 128 and final person representations are concatenated. Our 9 time-steps model's performance is 84.2% vs. 81.5% from [25]. We did not compare with [3] as it uses much stronger features (VGG16) and extra annotations (pairwise interaction).  Note that CAD has very simple "relations", in that the scene label is the label of the majority (often entirety) of the people in a scene.  However, this result demonstrates that our relational layer is able to capture inter-person information to improve classification results. 



## Code Scope & Requirements
* The provided code is a simplified version of our code. With simple effort, you can extend to whatever in the paper.
* The provided code doesn't contain the retrieval part.
* The provided example is for a single frame processing (though the Data Mgr can read temporal data, see Data below)
* The provided code is limited to clique style graphs, not general graphs. E.g. You can use it for a fully connected case or e.g. groups of cliques (e.g. in volleyball team 1 is clique and team 2 is another clique, or every 3 nearby players are a clique
* The provided code doesn't build the data, it just shows how to process the data using the relational network. Build initial representations for people is easy.
* You may use stage 1 in our C++ code for [CVPR 16](https://github.com/mostafa-saad/deep-activity-rec) to get such data (it build classifier, extra representations in the format below). You need to convert LevelDb to PKL format

## Data Format
* In src/data, a simple ready file for train and test in pkl format
* Provided code loads the whole data during the runtime. This might be problematic for some machines due to RAM issue. You may replace this part with another strategy.
* To understand how to structure data for a temporal clip, let's assume we have 12 persons, each clip is 10 frames. Ith Person in frame t is represented using 4096 features from VGG19
* Each entry in the pkl will be a single person representation (4096 features)
* The whole clip will be 12 * 10 = 120 rows
* The first 10 rows will be for the first person (his 10 representations corresponding to the 10 frames)
* The second 10 rows will be for the second person, and so on.
* If there are fewer people than 12 or a person is not available for all 10 frames, use zeros
* The next 120 rows will be for the second video clip.
* Be careful to not stack the data as 10 (steps) * 12 (persons), but as I clarified.
* The program reads the 120 lines, rearrange them as 10*(12*4096), that is 10 rows, each row has the whole scene people concatenated



## Installation
* Lasagne 0.1, Theano 0.8.2, Python 2.7.11, CUDA 8.0
* Run the main of relational_network.py for an example that load attached basic Data.

## License and Citation

Source code is released under the **BSD 2-Clause license**


    @inproceedings{msibrahiECCV18RelationalNetwork,
      author    = {Mostafa S. Ibrahim and Greg Mori},
      title     = {Hierarchical relational network for group activity recognition and retrieval},
      booktitle = {2018 European Conference on Computer Vision (ECCV)},
      year      = {2018}
    }


## Poster and Powerpoint
* You can find a presentation for the paper [here](https://docs.google.com/presentation/d/1PbjiquGAzQoUeTnC-rp-GeNLWp51iS9CBMBRBRR4wSM/edit?usp=sharing).
* You can find our ECCV 2018 poster [here](https://github.com/mostafa-saad/hierarchical-relational-network/blob/master/files/ibrahim18-eccv-poster.pdf).

<img src="https://github.com/mostafa-saad/hierarchical-relational-network/blob/master/files/poster.jpg" alt="Poster" height="400" >

Mostafa while presenting the poster.


