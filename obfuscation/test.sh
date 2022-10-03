# python train_joint.py --version v2_kldiv_sensitive_new_sens --ytmodel-path ../surrogate_model/param/policy_v2_kldiv.pt --agent-path ./param/agent_sens.pkl --eval  &> log.txt
python train_joint.py --version v2_kldiv_sensitive_new_50v2_sens --ytmodel-path ../surrogate_model/param/policy_v2_kldiv.pt --agent-path ./param/agent_sens_50_v2.pkl --eval  &> log.txt
python train_joint.py --version v2_kldiv_sensitive_new_org_sens --ytmodel-path ../surrogate_model/param/policy_v2_kldiv.pt --agent-path ./param/agent_0.2_v2_kldiv.pkl --eval  &> log.txt


# python train_joint.py --version v2_kldiv_sensitive_new_sens --ytmodel-path ../surrogate_model/param/policy_v2_kldiv.pt --agent-path ./param/agent_0.2_v2_kldiv.pkl --eval  &> log.txt

# python train_joint.py --version v2_kldiv_sensitive_new_sens2 --ytmodel-path ../surrogate_model/param/policy_v2_kldiv.pt --agent-path ./param/agent_0.3_v2_kldiv.pkl --eval  --alpha 0.3 &> log.txt

# python train_joint.py --version v2_kldiv_sensitive_new_sens3 --ytmodel-path ../surrogate_model/param/policy_v2_kldiv.pt --agent-path ./param/agent_0.5_v2_kldiv.pkl --eval  --alpha 0.5 &> log.txt