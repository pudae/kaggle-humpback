CUDA_VISIBLE_DEVICES=0 python inference_similarity.py --config=configs/densenet121.1st.yml --output_path=similarities/1st.csv --checkpoint_name=swa.pth
CUDA_VISIBLE_DEVICES=0 python inference_similarity.py --config=configs/densenet121.2nd.yml --output_path=similarities/2nd.csv --checkpoint_name=swa.pth
CUDA_VISIBLE_DEVICES=0 python inference_similarity.py --config=configs/densenet121.3rd.yml --output_path=similarities/3rd.csv --checkpoint_name=swa.pth
 
python make_submission.py --input_path=similarities/1st.csv,similarities/2nd.csv,similarities/3rd.csv --output_path=submissions/submission.csv --threshold=0.385
python post_processing.py --input_path=submissions/submission.csv --output_path=submissions/submission.processed.csv
