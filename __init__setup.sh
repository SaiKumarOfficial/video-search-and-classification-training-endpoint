echo [$(date)]: "START"
echo [$(date)]: "Creating conda env with python 3.8" # change py version as per your need
conda create --prefix ./training python=3.8 -y
echo [$(date)]: "activate training"
source activate ./training
echo [$(date)]: "installing the requirements" 
pip install -r requirements.txt
echo [$(date)]: "END" 