install:
	pip install -r requirements.txt

ftext:
	python main.py --dataset "lastfm" --model "HyperSaR" --num_layer 2 --edge_dropout 0.1 --loss_weight 0.001
bert_base:
	python main.py --dataset "lastfm" --model "HyperSaR" --num_layer 2 --edge_dropout 0.1 --loss_weight 0.001 --bert base
bert_large:
python main.py --dataset "lastfm" --model "HyperSaR" --num_layer 2 --edge_dropout 0.1 --loss_weight 0.001 --bert large

help:
	@echo "############### Options: ###############"
	@echo "install: install requirements"
	@echo "ftext: run without BERT embeddings"
	@echo "bert_base: run with BERT_Base embeddings"
	@echo "bert_large: run with BERT_Large embeddings"
	@echo "########################################"
