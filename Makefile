NAME := tensorflow-flower-retrain
VERSION := 0.1

create-conda:
	conda env create -f environment.yml -n $(NAME)

remove-conda:
	conda env remove -y -n $(NAME)
