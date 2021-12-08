import torch
import torch.nn as nn
import torch.nn.functional as F


class A2Clstm(torch.nn.Module):
    def __init__(self, env_args, video_embeddings, graph_embeddings, with_graph=True):
        super(A2Clstm,self).__init__()
        self.env_args = env_args
        self.video_embeddings = video_embeddings # num_videos * emb_dim
        self.graph_embeddings = graph_embeddings
        self.convs = nn.ModuleList([nn.Conv2d(1, self.env_args.kernel_dim, (self.env_args.K, self.env_args.emb_dim))
                                    for self.env_args.K in self.env_args.kernel_size])
        self.fc = nn.Linear(len(self.env_args.kernel_size) * self.env_args.kernel_dim, self.env_args.hidden_dim)
        self.lstm = nn.LSTM(
            input_size=self.env_args.hidden_dim,
            hidden_size=self.env_args.hidden_dim,
            num_layers=self.env_args.lstm_num_layer,
            bidirectional=False,
            dropout=0.2,
            batch_first=True
        )
        self.critic_linear = nn.Linear(self.env_args.hidden_dim, 1)
        # self.actor_linear = nn.Linear(self.env_args.hidden_dim, self.env_args.action_dim)
        self.actor_linear = nn.Linear(self.env_args.hidden_dim, self.env_args.emb_dim)
        self.with_graph = with_graph

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        batch_size, seq_len = inputs.shape
        inputs = self.graph_embeddings(inputs.reshape(-1).tolist(), self.with_graph).reshape(batch_size, 1, seq_len, self.env_args.emb_dim)
        inputs = [F.relu(conv(inputs)).squeeze(3) for conv in self.convs]
        inputs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inputs]
        concated = torch.cat(inputs, 1)

        x = self.fc(concated)
        x = x.view(x.size(0), 1, -1)
        # print(x.size())
        x, (hx, cx) = self.lstm(x, (hx, cx)) # Single step forward
        x = x.view(x.size(0), -1)
        actor_out = torch.matmul(self.actor_linear(x), self.video_embeddings.t())
        # print(actor_out.size())

        return self.critic_linear(x), actor_out, hx, cx


class Agent(object):
    def __init__(self, model, optimizer, env_args):
        self.model = model
        self.env_args = env_args
        self.state = torch
        self.hx = torch.zeros(self.env_args.lstm_num_layer, self.env_args.num_browsers, self.env_args.hidden_dim,).to(self.env_args.device)
        self.cx = torch.zeros(self.env_args.lstm_num_layer, self.env_args.num_browsers, self.env_args.hidden_dim,).to(self.env_args.device)
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.terminate = False
        self.optimizer = optimizer
    
    def train(self):
        self.model.train()
        
    def eval(self):
        self.model.eval()
        
    def take_action(self, state, terminate=False):
        self.terminate = terminate
        self.state = state
        if self.terminate:
            value, _, _, _ = self.model((self.state, (self.hx, self.cx)))
            self.values.append(value.squeeze(1))
            return None
        else:
            value, logit, self.hx, self.cx = self.model((self.state, (self.hx, self.cx)))
            self.values.append(value.squeeze(1))
            # print(value.size(), logit.size())
            prob = F.softmax(logit, 1)
            log_prob = F.log_softmax(logit, 1)
            entropy = -(log_prob * prob).sum(1)
            self.entropies.append(entropy)
            action = prob.multinomial(num_samples=1).data
            # print(action.view(-1))
            log_prob = log_prob.gather(1, action)
            self.log_probs.append(log_prob.squeeze(1))
            return action.view(-1).tolist()

    def save_param(self):
        print("saving rl agent")
        torch.save(self.model.state_dict(),self.env_args.agent_path)

    def update_rewards(self, rewards):
        rewards = torch.from_numpy(rewards).to(self.env_args.device)
        self.rewards.append(rewards)

    def update_model(self, retrain=False):
        if retrain:
            self.model.load_state_dict(torch.load(self.env_args.agent_path))

        GAMMA = self.env_args.GAMMA
        T = self.env_args.T
        policy_loss = 0
        value_loss = 0
        loss = 0
        gae = 0
        avg_R = 0

        R = self.values[-1]
        if len(self.rewards) == 0:
            return 0, 0
        for i in reversed(range(len(self.rewards))):
            avg_R += self.rewards[i]
            R = GAMMA * R + self.rewards[i]
            advantage = R - self.values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            # Generalized Advantage Estimataion
            delta_t = self.rewards[i] + GAMMA * self.values[i + 1].data - \
                self.values[i].data
            gae = gae * GAMMA * T + delta_t
            policy_loss = policy_loss - self.log_probs[i] * gae -\
                    0.01 * self.entropies[i]
        self.optimizer.zero_grad()

        loss = policy_loss.sum() + 0.5 * value_loss.sum(0)
        print("loss: {}, {}; reward: {}".format(policy_loss.sum().item(), value_loss.sum(0).item(), self.rewards))
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm(self.model.parameters(), 100)
        self.optimizer.step()
        return loss.item() / len(self.rewards) / self.env_args.num_browsers, \
        avg_R.sum(0).cpu().numpy() / len(self.rewards) / self.env_args.num_browsers

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.terminate = False
        self.hx = torch.zeros(self.env_args.lstm_num_layer, self.env_args.num_browsers, self.env_args.hidden_dim,).to(self.env_args.device)
        self.cx = torch.zeros(self.env_args.lstm_num_layer, self.env_args.num_browsers, self.env_args.hidden_dim,).to(self.env_args.device)