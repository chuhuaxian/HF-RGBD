# import pcl
# import numpy as np
#
#
# p = pcl.PointCloud(np.array([[1, 2, 3], [3, 4, 5]], dtype=np.float32))
# seg = p.make_segmenter()
# seg.set_model_type(pcl.SACMODEL_PLANE)
# seg.set_method_type(pcl.SAC_RANSAC)
# indices, model = seg.segment()