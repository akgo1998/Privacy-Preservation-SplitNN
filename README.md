# Privacy-Preservation-SplitNN

This project aims to develop a method for privacy preservation of images in distributed deep learning applications. 

By modifying the loss function in a Split-NN based classification problem, we are able to study the trade-off we need to achieve 
to protect the privacy of input data versus the accuracy of the classifier. 

We demonstrate an instance of the above mentioned setting for MNIST data, where the attackers attempts to reconstruct the input data using 
the leaked-latent representation. We observe that the accuracy drops from 0.98 to 0.41 as the value of the weighting (given to privacy preservation) 
is increased from 0 to 0.2.
