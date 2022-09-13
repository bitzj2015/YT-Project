ALPHA=0.5
AGENT_PATH="./param/agent_${ALPHA}_v2_kldiv.pkl"
DENOISER_PATH="./param/denoiser_${ALPHA}_v2_kldiv.pkl"
YTMODEL_PATH="../surrogate_model/param/policy_v2_kldiv.pt"

# python train_joint.py --agent-path ${AGENT_PATH} --denoiser-path ${DENOISER_PATH} --ytmodel-path ${YTMODEL_PATH} --alpha=${ALPHA} --version v2_kldiv_${ALPHA}_0 --use-rand 0

# python train_joint.py --agent-path ${AGENT_PATH} --denoiser-path ${DENOISER_PATH} --ytmodel-path ${YTMODEL_PATH} --alpha=${ALPHA} --version v2_kldiv_${ALPHA}_1 --use-rand 1

# python train_joint.py --agent-path ${AGENT_PATH} --denoiser-path ${DENOISER_PATH} --ytmodel-path ${YTMODEL_PATH} --alpha=${ALPHA} --version v2_kldiv_${ALPHA}_1 --use-rand 1

# python train_joint.py --agent-path ${AGENT_PATH} --denoiser-path ${DENOISER_PATH} --ytmodel-path ${YTMODEL_PATH} --alpha=${ALPHA} --version v2_kldiv_${ALPHA}_test_1 --use-rand 1 --eval

# python train_joint.py --agent-path ${AGENT_PATH} --denoiser-path ${DENOISER_PATH} --ytmodel-path ${YTMODEL_PATH} --alpha=${ALPHA} --version v2_kldiv_${ALPHA}_test_2 --use-rand 2 --eval

python eval_realuser.py --agent-path ${AGENT_PATH} --denoiser-path ${DENOISER_PATH} --ytmodel-path ${YTMODEL_PATH} --alpha=0.3 --version realuser --use-rand 1
python eval_realuser.py --agent-path ${AGENT_PATH} --denoiser-path ${DENOISER_PATH} --ytmodel-path ${YTMODEL_PATH} --alpha=0.3 --version realuser
python eval_realuser.py --agent-path ${AGENT_PATH} --denoiser-path ${DENOISER_PATH} --ytmodel-path ${YTMODEL_PATH} --alpha=0.3 --version realuser --use-rand 2