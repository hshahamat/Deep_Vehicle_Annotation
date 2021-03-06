# Deep Learning
# Fast Automatic Vehicle Annotation for Urban Trafﬁc Surveillance
A keras implementation (non official) of the paper "Fast Automatic Vehicle Annotation for Urban Trafﬁc Surveillance".
https://ieeexplore.ieee.org/document/8082130

Abstract—Automatic vehicle detection and annotation for streaming video data with complex scenes is an interesting but challenging task for intelligent transportation systems. In this paper, we present a fast algorithm: detection and annotation for vehicles (DAVE), which effectively combines vehicle detection and attributes annotation into a uniﬁed framework. DAVE consists of two convolutional neural networks: a shallow fully convolutional fast vehicle proposal network (FVPN) for extracting all vehicles’ positions, and a deep attributes learning network (ALN), which aims to verify each detection candidate and infer each vehicle’s pose, color, and type information simultaneously. These two nets
are jointly optimized so that abundant latent knowledge learned from the deep empirical ALN can be exploited to guide training the much simpler FVPN. Once the system is trained, DAVE can achieve efﬁcient vehicle detection and attributes annotation for real-world trafﬁc surveillance data, while the FVPN can be independently adopted as a real-time high-performance vehicle detector as well. We evaluate the DAVE on a new self-collected urban trafﬁc surveillance data set and the public PASCAL VOC2007 car and LISA 2010 data sets, with consistent improvements over existing algorithms.


<div align="center">
    <img src="/Illustration.PNG">
</div>

