# NonLIMear - NonLinear Spatial/Temporal Encoder and Deep Learning Model for Nino3.4 Forecasting

--  A Tool Box for ENSO forecasting Using Non-Linear Linear Inverse Models---


The fundamental framework is built from the excellent [Penland & Sardeshmukh (1995) -- Optimal Growth of Tropical Sea Surface Temperature Anomalies](https://journals.ametsoc.org/view/journals/clim/8/8/1520-0442_1995_008_1999_togots_2_0_co_2.xml) (citation below.) we then modify this to make the framework a non-linear deeplearning model that is iteratively trained. 

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

   @article { TheOptimalGrowthofTropicalSeaSurfaceTemperatureAnomalies,
      author = "Cécile  Penland and Prashant D.  Sardeshmukh",
      title = "The Optimal Growth of Tropical Sea Surface Temperature Anomalies",
      journal = "Journal of Climate",
      year = "1995",
      publisher = "American Meteorological Society",
      address = "Boston MA, USA",
      volume = "8",
      number = "8",
      doi = "10.1175/1520-0442(1995)008<1999:TOGOTS>2.0.CO;2",
      pages=      "1999 - 2024",
      url = "https://journals.ametsoc.org/view/journals/clim/8/8/1520-0442_1995_008_1999_togots_2_0_co_2.xml"
}

    @article{cachay2021world,
          title={The World as a Graph: Improving El Ni\~no Forecasts with Graph Neural Networks}, 
          author={Salva Rühling Cachay and Emma Erickson and Arthur Fender C. Bucker and Ernest Pokropek and Willa Potosnak and Suyash Bire and Salomey Osei and Björn Lütjens},
          year={2021},
          eprint={2104.05089},
          archivePrefix={arXiv},
          primaryClass={cs.LG}
    }