find ./dataset/negatives/ -name "*.jpg" -print > dataset/palm_negatives.info
find ./dataset/negatives/ -name "*.png" -print >> dataset/palm_negatives.info

/usr/local/bin/opencv_createsamples -info dataset/positives.info -vec dataset/palm_samples.vec -w 48 -h 48 -num 3000

/usr/local/bin/opencv_traincascade  -data dataset/palm_cascad_data -vec dataset/palm_samples.vec -bg dataset/palm_negatives.info -numStages 20 -w 48 -h 48 -numPos 2000 -numNeg 4000 -minHitRate 0.9995 -maxFalseAlarmRate 0.5 -featureType LBP -precalcValBufSize 8192000 -precalcIdxBufSize 8192000 -numThreads 12
