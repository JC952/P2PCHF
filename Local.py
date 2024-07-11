import torch
from torch import nn
from torch_geometric.graphgym import optim
from tqdm import tqdm


def local_train(self, net, inter_net, train_loader):
    global linear_out
    a = self.args.temp
    net = net.to(self.device)
    inter_net = inter_net.to(self.device)
    optimizer = optim.Adam(net.parameters(), lr=self.local_lr)
    criterionCE = nn.CrossEntropyLoss().to(self.device)
    criterionKL = nn.KLDivLoss(reduction='batchmean')
    criterionKL.to(self.device)
    iterator = tqdm(range(self.local_epoch))
    for _ in iterator:
        for batch_idx, (private_data, labels, domian_labels) in enumerate(train_loader):
            private_data = private_data.to(self.device)
            labels = labels.to(self.device)
            outputs = net(private_data)  # (128,10)
            linear_out = outputs.detach().cpu()
            s_anchors = self.compute_class_centroids(self.l2norm(linear_out), labels.clone().detach().cpu())
            s_anchors = self.l2norm(s_anchors).cuda()
            stu_contrastive = torch.div(torch.mm(outputs, s_anchors.T), self.T)  # [bs, n_anchors]
            # 冻结，梯度消失
            with torch.no_grad():
                inter_outputs = inter_net(private_data)
                labels_t = torch.argmax(inter_outputs, dim=1)
                linear_out = inter_outputs.detach().cpu()
                t_anchors = self.compute_class_centroids(self.l2norm(linear_out), labels.clone().detach().cpu())
                t_anchors = self.l2norm(t_anchors).cuda()
                tea_contrastive = torch.div(torch.mm(inter_outputs, t_anchors.T), self.T)  # [bs, n_anchors]

            loss_hard = criterionCE(outputs, labels)
            if self.args.model_setting == 4:  # M3
                inter_loss = self.DistillKL_logit_stand(outputs, inter_outputs, 1)
                crp_loss = self.KLD_criterion(tea_contrastive, stu_contrastive)
                loss_hard_t = criterionCE(outputs, labels_t)
                loss = loss_hard + a * inter_loss + (1 - a) * loss_hard_t + crp_loss * 0.5
            elif self.args.model_setting == 3:  # M1
                loss = loss_hard
            elif self.args.model_setting == 2:  # M2
                loss = loss_hard
            elif self.args.model_setting == 1:  # Base
                loss = loss_hard
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
