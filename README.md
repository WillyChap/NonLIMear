# NonLIMear - A Tool Box for ENSO forecasting Using Non-Linear Linear Inverse Models


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

Please consider citing the following paper if you find it, or the code, helpful. Thank you!

    @article{cachay2021world,
          title={The World as a Graph: Improving El Ni\~no Forecasts with Graph Neural Networks}, 
          author={Salva Rühling Cachay and Emma Erickson and Arthur Fender C. Bucker and Ernest Pokropek and Willa Potosnak and Suyash Bire and Salomey Osei and Björn Lütjens},
          year={2021},
          eprint={2104.05089},
          archivePrefix={arXiv},
          primaryClass={cs.LG}
    }