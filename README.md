# Heart Segmentation App 
Artificial Intelligence Project

#### How to run this app in a local server using [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) :
1. First, install [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) on your device. I personally recommended the _***Miniconda***_
2. Clone this repository and go to the repo directory
   - `git clone https://github.com/itsahyarr/heart_segmentation_app.git`
   - `cd heart_segmentation_app`
3. Create conda environment with the configuration file `conda_env.yml`
   - `conda env create -f conda_env.yml`
4. Activate the conda environment
   - `conda activate heart`
5. Run **main.py**
   - `python main.py`
6. It will run on `http://localhost:5000` or `http://127.0.0.1:5000` by default
7. Create an account (just random data) and then login to access the main page. After logged in, you can upload the heart image scan and get the segmentation result.

> Note : if the .h5 file is corrupted while cloning the repository, please do a manual download that file below :)<br>
> .h5 file => [DOWNLOAD](https://drive.google.com/file/d/1QYpbTeesOMMEjqE9itxjNOiYulOlLEfq/view?usp=sharing)
