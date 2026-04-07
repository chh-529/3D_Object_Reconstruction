OBJECT ?= castard

icp:
	python3 pose_graph_ICP.py --object $(OBJECT)

feature:
	python3 pose_graph_Feature_based.py --object $(OBJECT)