CUDA_VISIBLE_DEVICES=0 python inference_landmark.py --config=configs/landmark.densenet.0.yml --checkpoint_name=swa.10.3450.pth --split=all --output_path=data/landmark.train.0.csv 
CUDA_VISIBLE_DEVICES=0 python inference_landmark.py --config=configs/landmark.densenet.1.yml --checkpoint_name=swa.10.3450.pth --split=all --output_path=data/landmark.train.1.csv 
CUDA_VISIBLE_DEVICES=0 python inference_landmark.py --config=configs/landmark.densenet.2.yml --checkpoint_name=swa.10.3450.pth --split=all --output_path=data/landmark.train.2.csv 
CUDA_VISIBLE_DEVICES=0 python inference_landmark.py --config=configs/landmark.densenet.3.yml --checkpoint_name=swa.10.3450.pth --split=all --output_path=data/landmark.train.3.csv 
CUDA_VISIBLE_DEVICES=0 python inference_landmark.py --config=configs/landmark.densenet.4.yml --checkpoint_name=swa.10.3450.pth --split=all --output_path=data/landmark.train.4.csv 


CUDA_VISIBLE_DEVICES=0 python inference_landmark.py --config=configs/landmark.densenet.0.yml --checkpoint_name=swa.10.3450.pth --split=test --output_path=data/landmark.test.0.csv 
CUDA_VISIBLE_DEVICES=0 python inference_landmark.py --config=configs/landmark.densenet.1.yml --checkpoint_name=swa.10.3450.pth --split=test --output_path=data/landmark.test.1.csv 
CUDA_VISIBLE_DEVICES=0 python inference_landmark.py --config=configs/landmark.densenet.2.yml --checkpoint_name=swa.10.3450.pth --split=test --output_path=data/landmark.test.2.csv 
CUDA_VISIBLE_DEVICES=0 python inference_landmark.py --config=configs/landmark.densenet.3.yml --checkpoint_name=swa.10.3450.pth --split=test --output_path=data/landmark.test.3.csv 
CUDA_VISIBLE_DEVICES=0 python inference_landmark.py --config=configs/landmark.densenet.4.yml --checkpoint_name=swa.10.3450.pth --split=test --output_path=data/landmark.test.4.csv 


python ensemble_landmarks.py --input_path=data/landmark.train.0.csv,data/landmark.train.1.csv,data/landmark.train.2.csv,data/landmark.train.3.csv,data/landmark.train.4.csv --output_path=data/landmark.train.5.csv --split=train
python ensemble_landmarks.py --input_path=data/landmark.test.0.csv,data/landmark.test.1.csv,data/landmark.test.2.csv,data/landmark.test.3.csv,data/landmark.test.4.csv --output_path=data/landmark.test.5.csv --split=test
