# hierarchical-relational-network
Hierarchical relational network for group activity recognition and retrieval


Update: Our paper got accepted at ECCV 2018 :)
Code after ECCV conference will be available. So busy with a dozen of things.

# Extra Dataset note
The Collective Activity Dataset (CAD) consists of 44 videos and five labels used as both person action and  group activity (crossing, walking, waiting, talking, and queueing). The majority activity in the scene defines the group activity. We followed the same data split, temporal window and implementation details as [25], including AlexNet features (due to the short time; faster to train/extract features).

We used a single relational layer with a simple graph: each 3 consecutive persons (spatially in horizontal dimension) are grouped as a clique. The layer maps a person of size 4096 to 128 and final person representations are concatenated. Our 9 time-steps model's performance is 84.2% vs.\ 81.5% from [25]. We did not compare with [3] as it uses much stronger features (VGG16) and extra annotations (pairwise interaction).  Note that CAD has very simple "relations", in that the scene label is the label of the majority (often entirety) of the people in a scene.  However, this result demonstrates that our relational layer is able to capture inter-person information to improve classification results. 
