SI-HDR - dataset for comparison of single-image high dynamic range reconstruction methods
2022 August

The dataset consists of 181 HDR images. Each image includes:
* a RAW exposure stack (in raw.zip)
* an HDR image (in reference.zip)
* simulated camera images at two different exposures (in input.zip)
* Results of 6 single-image HDR reconstruction methods: Endo et al. 2017, Eilertsen et al. 2017, Marnerides et al. 2018, Lee et al. 2018, Liu et al. 2020, and Santos et al. 2020 (in reconstructions.zip)

# Project web page

More details can be found at: https://www.cl.cam.ac.uk/research/rainbow/projects/sihdr_benchmark/

# Overview

This dataset contains 181 RAW exposure stacks selected to cover a wide range of image content and lighting conditions. Each scene is composed of 5 RAW exposures and merged into an HDR image using the estimator that accounts photon noise [3] (code at [HDRutils](https://github.com/gfxdisp/HDRutils)). A simple color correction was applied using a reference white point and all merged HDR images were resized to 1920×1280 pixels.

The primary purpose of the dataset was to compare various single image HDR (SI-HDR) methods [1]. Thus, we selected a wide variety of content covering nature, portraits, cities, indoor and outdoor, daylight and night scenes. After merging and resizing, we simulated captures by applying a custom CRF and added realistic camera noise based on estimated noise parameters of *Canon 5D Mark III*.

The simulated captures were inputs to six selected SI-HDR methods. You can view the reconstructions of various methods for select scenes on our [interactive viewer](https://www.cl.cam.ac.uk/research/rainbow/projects/sihdr_benchmark/). For the remaining scenes, please download the appropriate zip files. We conducted a rigorous pairwise comparison experiment on these images to find that widely-used metrics did not correlate well with subjective data. We then proposed an improved evaluation protocol for SI-HDR [1].

# Structure

The dataset consists of 4 folders, each available as a separate ZIP file. Here you can check which portion of the dataset you may want to download.

The dataset is structured as follows:
```
	sihdr/
	|	raw/					# RAW 14 bit files
	|	|	001/
	|	|	|	\_07A5797.CR2
	|	|	|	\_07A5798.CR2
	|	|	|	...
	|	|
	|	├───002/
	|	├───003/
	|	|	...
	|
	├───reference/				# Merged and resized reference HDR
	|	|	001.exr
	|	|	002.exr
	|	|	...
	|
	├───input/					# SDR image after applying CRF, noise and quantization
	|	|	clip_95/
	|	|	|	001.png
	|	|	|	002.png
	|	|	|	...
	|	|
	|	└───clip_97/
	|		|	001.png
	|		|	002.png
	|		|	...
	|
	└───reconstructions/		# HDR outputs of various methods
		|	drtmo/
		|	expandnet/
		|	hdrcnn/
		|	hdrgan/
		|	maskhdr/
		└───singlehdr/

```

Note that the input contains two exposures or clipping levels (95% and 97%) corresponding to what percent of HDR pixels are retained during simulation. Thus each folder under "reconstructions" also has separate outputs for these two clipping levels.

If you find this dataset useful, please cite [1].

# References

[1] Param Hanji, Rafał K. Mantiuk, Gabriel Eilertsen, Saghi Hajisharif, and Jonas Unger. 2022. “Comparison of single image hdr reconstruction methods — the caveats of quality assessment.” In *Special Interest Group on Computer Graphics and Interactive Techniques Conference Proceedings (SIGGRAPH ’22 Conference Proceedings)*. [Online]. Available: https://www.cl.cam.ac.uk/research/rainbow/projects/sihdr_benchmark/

[2] Gabriel Eilertsen, Saghi Hajisharif, Param Hanji, Apostolia Tsirikoglou, Rafał K. Mantiuk, and Jonas Unger. 2021. “How to cheat with metrics in single-image HDR reconstruction.” In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops*. 3998–4007.

[3] Param Hanji, Fangcheng Zhong, and Rafał K. Mantiuk. 2020. “Noise-Aware Merging of High Dynamic Range Image Stacks without Camera Calibration.” In *Advances in Image Manipulation (ECCV workshop)*. Springer, 376–391. [Online]. Available: https://www.cl.cam.ac.uk/research/rainbow/projects/noise-aware-merging/
