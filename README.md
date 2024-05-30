An ensembled method of cnn and lstm for deepfake video detection.

The face is cropped as a video first. The optical flow of these cropped faces are calculated. Both the cropped face and optical flow information is fed to the CNN. Finally, the output is used for the LSTM to define if a video is deepfake or original.
