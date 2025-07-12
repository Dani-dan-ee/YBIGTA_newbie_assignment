
# anaconda(또는 miniconda)가 존재하지 않을 경우 설치해주세요!
## TODO
conda --version
if [ $? != 0 ]
then
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p $HOME/miniconda
  export PATH=$HOME/miniconda/bin:$PATH
  source ~/.bashrc
fi

# Conda 환셩 생성 및 활성화
## TODO
. ~/miniconda/etc/profile.d/conda.sh
conda env list | grep myenv
if [ $? != 0 ]
then
  conda create -n myenv python=3.10 -y
fi
conda activate myenv

## 건드리지 마세요! ##
python_env=$(python -c "import sys; print(sys.prefix)")
if [[ "$python_env" == *"/envs/myenv"* ]]; then
    echo "[INFO] 가상환경 활성화: 성공"
else
    echo "[INFO] 가상환경 활성화: 실패"
    exit 1 
fi

# 필요한 패키지 설치
## TODO
pip install mypy
# Submission 폴더 파일 실행
cd submission || { echo "[INFO] submission 디렉토리로 이동 실패"; exit 1; }

for file in *.py; do
    ## TODO
    name=$(echo $file | cut -d. -f1)
    num=$(echo $name | grep -o '[0-9]*')
    input_file=../input/${name}_input
    output_file=../output/${num}_output
    if [ -f $input_file ]
    then
      python $file < $input_file > $output_file
    fi
done

# mypy 테스트 실행 및 mypy_log.txt 저장
## TODO
mypy . > ../mypy_log.txt
# conda.yml 파일 생성
## TODO
conda env export > ../conda.yml
# 가상환경 비활성화
## TODO
conda deactivate