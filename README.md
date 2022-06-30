# NonLIMear - NonLinear Spatial/Temporal Encoder and Deep Learning Model for Nino3.4 Forecasting

--  A Tool Box for ENSO forecasting Using Non-Linear Linear Inverse Models ---


The fundamental framework is built from the excellent [[1] Penland & Sardeshmukh (1995) -- Optimal Growth of Tropical Sea Surface Temperature Anomalies](https://journals.ametsoc.org/view/journals/clim/8/8/1520-0442_1995_008_1999_togots_2_0_co_2.xml) (citation below.) we then modify this to make the framework a non-linear deeplearning model that is iteratively trained. 



In [1] it is argued that from SST observations for the period 1950–90 that the tropical Indo-Pacific ocean-atmosphere system may be described as a stable linear dynamical system driven by spatially coherent Gaussian white noise. Evidence is presented that the predictable component of SST anomaly growth is associated with the constructive interference of several damped normal modes after an optimal initial structure is set up by the white noise forcing. In particular, El Niño–Southern Oscillation (ENSO) growth is associated with an interplay of at least three damped normal modes, with periods longer than two years and decay times of 4 to 8 months, rather than the manifestation of a single unstable mode whose growth is arrested by nonlinearities. In the modern era deep-learning models have been used in order to forecast NINO3.4 to some success notable [2]. [3] uses graph neural networks and argues that current deep learning models are based on convolutional neural networks which are difficult to interpret and can fail to model large-scale atmospheric patterns. In comparison, graph neural networks (GNNs) are capable of modeling large-scale spatial dependencies and are more interpretable due to the explicit modeling of information flow through edge connections. We show here that the original paper [1] leveraged a handy way to encode spatial/temporal information and we then bolster than skill through the addition of non-linear learned parameters. We show that the new model outperforms both [2] and [3] but retains the direct interpretability of [1]. 

[1] [Penland & Sardeshmukh (1995) -- Optimal Growth of Tropical Sea Surface Temperature Anomalies](https://journals.ametsoc.org/view/journals/clim/8/8/1520-0442_1995_00\
8_1999_togots_2_0_co_2.xml)

[2][Yoo-Geun Ham, Jeong-Hwan Kim & Jing-Jia Luo -- Deep learning for multi-year ENSO forecasts](https://www.nature.com/articles/s41586-019-1559-7)

[3][Salva Rühling Cachay et al. (2021) -- The World as a Graph: Improving El Nino Forecasts with Graph Neural Networks](https://arxiv.org/abs/2104.05089)



***

## TO DO: 
- Add yaml for build environment 
- interpret bias of propagator matrix
- add CRPS cost function and parametric/probabilistic forecast functionality

***
 
This project is still a work in progress, but some of the infrastructure is owed to/ modeled after [The Graphino Toolbox](https://github.com/salvaRC/Graphino)
The graphino citation is included below (credit where credit is due!). 


## Running the experiments
Please run the [*run_deeplim*](run_deeplim.py) script for the desired number of lead months h in {1,2, .., 23} (the horizon argument). 

**example run :** 
```python
python run_deeplim.py --horizon 3 --seed 42 --epochs 30 --gpu_id 0 >> out_epochs30.txt & 
```

To run at every lead time, modify the bash script [Run_Forecasts](Run_Forecasts.sh):


## Citation

    @article{PenlandSardeshmukh1995,
          author={Cécile  Penland and Prashant D.  Sardeshmukh},
          title={The Optimal Growth of Tropical Sea Surface Temperature Anomalies},
          journal={Journal of Climate},
          year={1995},
          publisher={American Meteorological Society},
          address={Boston MA USA},
          doi={10.1175/1520-0442(1995)008<1999:TOGOTS>2.0.CO;2},
          pages={1999 - 2024},
          url={https://journals.ametsoc.org/view/journals/clim/8/8/1520-0442_1995_008_1999_togots_2_0_co_2.xml}
    }

    @article{cachay2021world,
          title={The World as a Graph: Improving El Ni\~no Forecasts with Graph Neural Networks}, 
          author={Salva Rühling Cachay and Emma Erickson and Arthur Fender C. Bucker and Ernest Pokropek and Willa Potosnak and Suyash Bire and Salomey Osei and Björn Lütjens},
          year={2021},
          eprint={2104.05089},
          archivePrefix={arXiv},
          primaryClass={cs.LG}
    }