# VAR-PCANet

Matlab source code for the paper:

	A. Genovese, M. S. Hosseini, V. Piuri, K. N. Plataniotis, and F. Scotti
    "Acute Lymphoblastic Leukemia detection based on adaptive unsharpening and Deep Learning", 
    Proc. of the 2021 IEEE Int. Conf. on Acoustics, Speech, and Signal Processing (ICASSP 2021), 
    Toronto, ON, Canada, June 6-11, 2021
	
Project page:

	https://iebil.di.unimi.it/cnnALL/index.htm
    
Outline:
![Outline](https://iebil.di.unimi.it/cnnALL/imgs/outline.jpg "Outline")

Citation:

	@InProceedings {icassp21,
        author = {A. Genovese and M. S. Hosseini and V. Piuri and K. N. Plataniotis and F. Scotti},
        title = {Acute Lymphoblastic Leukemia detection based on adaptive unsharpening and Deep Learning},
        journal = {Proc. of the 2021 IEEE Int. Conf. on Acoustics, Speech, and Signal Processing (ICASSP 2021)},
        address = {Toronto, ON, Canada},
        pages = {1-5},
        month = {June},
        day = {6-11},
        year = {2021},
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

	
