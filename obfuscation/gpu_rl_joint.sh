ALPHA=0.2
VERSION="v2_kldiv"
AGENT_PATH="./param/agent_${ALPHA}_${VERSION}.pkl"
DENOISER_PATH="./param/denoiser_${ALPHA}_${VERSION}.pkl"
YTMODEL_PATH="../surrogate_model/param/policy_v2_kldiv.pt"

# python train_joint.py --agent-path ${AGENT_PATH} --denoiser-path ${DENOISER_PATH} --ytmodel-path ${YTMODEL_PATH} --alpha=${ALPHA} --version v2_kldiv_${ALPHA}_0 --use-rand 0

# python train_joint.py --agent-path ${AGENT_PATH} --denoiser-path ${DENOISER_PATH} --ytmodel-path ${YTMODEL_PATH} --alpha=${ALPHA} --version v2_kldiv_${ALPHA}_1 --use-rand 1

# python train_joint.py --agent-path ${AGENT_PATH} --denoiser-path ${DENOISER_PATH} --ytmodel-path ${YTMODEL_PATH} --alpha=${ALPHA} --version v2_kldiv_${ALPHA}_1 --use-rand 1

# python train_joint.py --agent-path ${AGENT_PATH} --denoiser-path ${DENOISER_PATH} --ytmodel-path ${YTMODEL_PATH} --alpha=${ALPHA} --version v2_kldiv_${ALPHA}_test_1 --use-rand 1 --eval

# python train_joint.py --agent-path ${AGENT_PATH} --denoiser-path ${DENOISER_PATH} --ytmodel-path ${YTMODEL_PATH} --alpha=${ALPHA} --version v2_kldiv_${ALPHA}_test_2 --use-rand 2 --eval

python eval_realuser.py --agent-path ${AGENT_PATH} --denoiser-path ${DENOISER_PATH} --ytmodel-path ${YTMODEL_PATH} --alpha=0.2 --version realuser_all_1 --use-rand 1
python eval_realuser.py --agent-path ${AGENT_PATH} --denoiser-path ${DENOISER_PATH} --ytmodel-path ${YTMODEL_PATH} --alpha=0.2 --version realuser_all_0
python eval_realuser.py --agent-path ${AGENT_PATH} --denoiser-path ${DENOISER_PATH} --ytmodel-path ${YTMODEL_PATH} --alpha=0.2 --version realuser_all_2 --use-rand 2

# python train_joint.py --agent-path ${AGENT_PATH} --denoiser-path ${DENOISER_PATH} --ytmodel-path ${YTMODEL_PATH} --alpha=${ALPHA} --version ${ALPHA}_${VERSION}
# python train_joint.py --agent-path ${AGENT_PATH} --denoiser-path ${DENOISER_PATH} --ytmodel-path ${YTMODEL_PATH} --alpha=${ALPHA} --version ${ALPHA}_${VERSION}_test_0 --eval
# python train_joint.py --agent-path ${AGENT_PATH} --denoiser-path ${DENOISER_PATH} --ytmodel-path ${YTMODEL_PATH} --alpha=${ALPHA} --version ${ALPHA}_${VERSION}_test_1 --use-rand 1

# python train_joint.py --agent-path ${AGENT_PATH} --denoiser-path ${DENOISER_PATH} --ytmodel-path ${YTMODEL_PATH} --alpha=${ALPHA} --version ${ALPHA}_${VERSION}_test_1 --use-rand 1 --eval
# python train_joint.py --agent-path ${AGENT_PATH} --denoiser-path ${DENOISER_PATH} --ytmodel-path ${YTMODEL_PATH} --alpha=${ALPHA} --version ${ALPHA}_${VERSION}_test_2 --use-rand 2 --eval