find ./dataset/negatives/ -name "*.jpg" -print > dataset/palm_negatives.info
find ./dataset/negatives/ -name "*.png" -print >> dataset/palm_negatives.info

/usr/local/bin/opencv_createsamples -info dataset/positives.info -vec dataset/palm_samples.vec -w 48 -h 48 -num 3000

/usr/local/bin/opencv_traincascade  -data dataset/palm_cascad_data -vec dataset/palm_samples.vec -bg dataset/palm_negatives.info -numStages 15 -w 48 -h 48 -numPos 1800 -numNeg 4000 -minHitRate 0.995 -maxFalseAlarmRate 0.5 -featureType LBP -precalcValBufSize 2048000 -precalcIdxBufSize 2048000 -numThreads 4
