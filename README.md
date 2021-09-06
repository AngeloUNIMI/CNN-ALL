# CNN-ALL

Matlab source code for the paper:

	A. Genovese, M. S. Hosseini, V. Piuri, K. N. Plataniotis, and F. Scotti, 
	"Acute Lymphoblastic Leukemia detection based on adaptive unsharpening and Deep Learning", 
	in Proc. of the 2021 IEEE Int. Conf. on Acoustics, Speech, and Signal Processing (ICASSP 2021), 
	Toronto, ON, Canada, June 6-11, 2021, pp. 1205-1209. 
	ISBN: 978-1-7281-7605-5. [DOI: 10.1109/ICASSP39728.2021.9414362]
	
Paper:

https://ieeexplore.ieee.org/document/9414362
	
Project page:

https://iebil.di.unimi.it/cnnALL/index.htm
    
Outline:
![Outline](https://iebil.di.unimi.it/cnnALL/imgs/outline_icassp21.jpg "Outline")

Citation:

	@InProceedings {icassp21,
    author = {A. Genovese and M. S. Hosseini and V. Piuri and K. N. Plataniotis and F. Scotti},
    booktitle = {Proc. of the 2021 IEEE Int. Conf. on Acoustics, Speech, and Signal Processing (ICASSP 2021)},
    title = {Acute Lymphoblastic Leukemia detection based on adaptive unsharpening and Deep Learning},
    address = {Toronto, ON, Canada},
    pages = {1205-1209},
    month = {June},
    day = {6-11},
    year = {2021},
    note = {978-1-7281-7605-5}
    }

Main files:

	- launch_VARPCANet: main file

Required files:

	- ./imgs/orig/ALL-IDB/ALL_IDB2/img: Database of images, with filenames in the format "Im001_1.tif", 
    the images can be downloaded at: https://homes.di.unimi.it/scotti/all/

Part of the code uses the Matlab source code of the paper:

	T. Chan, K. Jia, S. Gao, J. Lu, Z. Zeng and Y. Ma, 
	"PCANet: A Simple Deep Learning Baseline for Image Classification?," 
	in IEEE Transactions on Image Processing, vol. 24, no. 12, pp. 5017-5032, Dec. 2015.
	DOI: 10.1109/TIP.2015.2475625
	http://mx.nthu.edu.tw/~tsunghan/Source%20codes.html

The 1Shot-MaxPol library:

	Mahdi S. Hosseini and Konstantinos N. Plataniotis 
	"Convolutional Deblurring for Natural Imaging," 
	IEEE Transactions on Image Processing, 2019.
	https://github.com/mahdihosseini/1Shot-MaxPol
	
The FQPath library:

	Hosseini, Mahdi S., Jasper AZ Brawley-Hayes, Yueyang Zhang, Lyndon Chan, Konstantinos N. Plataniotis, and Savvas Damaskinos
	"Focus Quality Assessment of High-Throughput Whole Slide Imaging in Digital Pathology." 
	IEEE Transactions on Medical Imaging (2019)
	https://github.com/mahdihosseini/FQPath
	
The Fast N-D Grayscale Image Segmentation library:

	Fast N-D Grayscale Image Segmenation With c- or Fuzzy c-Means
	https://github.com/AntonSemechko/Fast-Fuzzy-C-Means-Segmentation
	
The Stain Deconvolution library:

	SCD_FastICA
	https://github.com/lisatostrams/SCD_FastICA
	
and the Colour Image Normalization library:

	Finlayson, G., Schiele, B., & Crowley, J. (1998). 
	Comprehensive Colour Image Normalization. 
	Computer Vision—ECCV’98, 1406, 475–490. 
	https://doi.org/10.1007/BFb0055655
	https://it.mathworks.com/matlabcentral/fileexchange/60360-comprehensive-colour-normalization
	




