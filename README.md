# CAT: A Unified Click-and-Track Framework for Realistic Tracking

Official implementation of **CAT (Click-And-Track)**.（The code is being gradually updated. Stay tuned.）

CAT introduces a realistic visual tracking paradigm where the target can be initialized by **a single click** instead of a precise bounding box. This greatly simplifies human interaction and better reflects real-world applications such as robotics, UAV tracking, and surveillance.

📄 [Paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Yuan_CAT_A_Unified_Click-and-Track_Framework_for_Realistic_Tracking_ICCV_2025_paper.pdf)
: *CAT: A Unified Click-and-Track Framework for Realistic Tracking*, ICCV 2025
## Pretrained Weights

Pretrained model weights can be downloaded from Baidu Netdisk: [Baidu Netdisk](https://pan.baidu.com/s/1AxFpiz0tApVENfZG7K-d5A) (code: **5sfv**) 

---

## Overview
![CAT Framework](assets/CAT_arch.png)
Most existing trackers assume a **precisely annotated bounding box** for initialization. However, drawing accurate bounding boxes is slow and impractical in many real-world scenarios.
CAT proposes a **unified click-based tracking framework** that enables reliable object tracking from only a **single user click**. The method bridges the gap between minimal user interaction and accurate target localization by introducing a click-based localization module, a spatial-visual prompt refinement mechanism, and a parameter-efficient mixture-of-experts adaptation.

---

## Benchmarks

CAT is evaluated on multiple tracking benchmarks including:
<table>
<thead style="background-color:#f2f2f2">
<tr>
<th>Method</th>
<th>LaSOT (AUC)</th>
<th>LaSOT<sub>ext</sub> (AUC)</th>
<th>TrackingNet (AUC)</th>
<th>GOT-10k (AO)</th>
</tr>
</thead>
<tbody>
<tr>
<td>CAT_od384</td>
<td>70.5</td>
<td>48.3</td>
<td>83.9</td>
<td>71.7</td>
</tr>
<tr>
<td>CAT_os384</td>
<td>68.5</td>
<td>48.7</td>
<td>82.8</td>
<td>72.1</td>
</tr>
<tr>
<td>CAT_os256</td>
<td>66.6</td>
<td>43.6</td>
<td>82.2</td>
<td>72.1</td>
</tr>
</tbody>
</table>
The method significantly reduces the performance gap between click-based tracking and traditional tracking.

---

## Install the environment

### Use Anaconda (CUDA 11.3)

```bash
conda create -n cat python=3.7
conda activate cat
bash install_CAT.sh
```

## Set project paths

Run the following command to set paths for this project

```bash
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```bash
lib/train/admin/local.py        # paths about training
lib/test/evaluation/local.py    # paths about testing
```
---

## Evaluation

Download the model [weights](https://pan.baidu.com/s/1AxFpiz0tApVENfZG7K-d5A) (code: **5sfv**). 
Put the downloaded weights under: YPUR_PROJECT_ROOT/output/checkpoints/train/CATos

Change the corresponding values in the following file to the actual benchmark saving paths:
lib/test/evaluation/local.py

---


1. If you want to analyze the initialization performance of CAT on different datasets separately, 
you can run the following command to execute only the initialization stage. 
Since the sampling process of simulated clicks involves some randomness, the average IoU of each initialization may vary slightly across runs.
```bash
python lib/test/tracker/init_with_CAT.py CATos cat_os_256 --dataset lasot
```

2. The full click-based tracking process can be executed using the following command. 
After running it, a point will be randomly sampled on the initial frame of each video according to the sampling rule to simulate a click, and CAT will be used for initialization and tracking. Due to the randomness in the sampling process, the tracking results may exhibit slight variations across runs.

```bash
python tracking/test.py CATos cat_os_256 --dataset lasot --threads 16 --num_gpus 4
python tracking/analysis_results.py # need to modify tracker configs and names
```


## Citation

If you find this work useful for your research, please consider citing:

```bibtex
@InProceedings{Yuan_2025_ICCV,
    author    = {Yuan, Yongsheng and Zhao, Jie and Wang, Dong and Lu, Huchuan},
    title     = {CAT: A Unified Click-and-Track Framework for Realistic Tracking},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {5690-5700}
}
```

---

## License

This project is released for academic research purposes only.
