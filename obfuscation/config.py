# Environment configuration
class EnvConfig(object):
    def __init__(
        self,
        num_browsers=50,
        alpha=0.2,
        his_len=10,
        emb_dim=384 + 19,
        hidden_dim=128,
        num_user_state=100,
        kernel_size=[3, 4, 5],
        kernel_dim=150,
        dropout=0.2,
        lstm_num_layer=2,
        rollout_len=40,
        GAMMA=0.99,
        T=1,
        rl_lr=0.001,
        denoiser_lr=0.001,
        action_dim=100,
        reward_dim=2,
        reward_w=[1,0.],
        device="cpu",
        agent_path="./param/agent.pkl", 
        logger=None, 
        version="test"
    ):
        self.num_browsers = num_browsers
        self.alpha = alpha
        self.his_len = his_len
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_user_state = num_user_state
        self.kernel_dim = kernel_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.lstm_num_layer = lstm_num_layer
        self.rollout_len = rollout_len
        self.GAMMA=GAMMA
        self.T = T
        self.rl_lr = rl_lr
        self.denoiser_lr = denoiser_lr
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.reward_w = reward_w
        self.device = device
        self.agent_path = agent_path
        self.logger = logger
        self.version = version
