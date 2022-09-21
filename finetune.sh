python main.py \
	--base "configs/stable-diffusion/v1-finetune.yaml" \
	--resume_from_checkpoint "weights/model.ckpt" \
	--name catalog \
    --gpus "0,1,2,3" \
	--train true \
	--no-test \
	--scale_lr false