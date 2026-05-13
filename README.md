# 1. Marabou 설치
 ## 1) 기본 세팅
 ```bash
 # 관리자 권한이 필요한 경우
 sudo apt-get update
 sudo apt install cmake
 sudo apt-get install protobuf-compiler libprotobuf-dev

 # 관리자 권한이 필요없는 경우
 apt-get update
 apt install cmake
 apt-get install protobuf-compiler libprotobuf-dev
```

 ## 2) Marabou 설치
 ```bash
 git clone https://github.com/NeuralNetworkVerification/Marabou.git
 cd Marabou/
 mkdir build 
 cd build
 cmake ../
 cmake --build . -j 4
 export PYTHONPATH=$PYTHONPATH:/path/to/marabou/folder
 export JUPYTER_PATH=$JUPYTER_PATH:/path/to/marabou/folder

 # 제대로 설치 되었는지 test
 make check -j 4
 ```

# 2. 의존성 설치

```bash
pip install -r requirements.txt
```


# 3. 폴더 구조
아래 폴더 구조를 따르도록 해주시길 바랍니다.
properties 폴더와 하위 데이터는 test.py를 실행하는 동안 생성됩니다.
```text
📁 최상위 폴더
├── 📄 generate_query.py
├── 📄 model.py
├── 📄 test.py
├── 📄 requirements.txt
├── 📄 README.md
├── 📁 Marabou
└── 📁 properties
    ├── 📄 image0_target0_epsilon0.001.txt
    ├── 📄 image0_target0_epsilon0.003.txt
    ...
    └── 📄 image4_target9_epsilon0.045.txt
    
```

# 4. 실행
모델을 훈련하고 쿼리를 만들고 marabou로 test 하는 코드는 모두 test.py에 위치합니다. 
```bash
python test.py
```