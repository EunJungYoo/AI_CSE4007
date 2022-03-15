import torch
from torch import nn
import torch.optim as optim


class ModelClass(nn.Module):
    def __init__(
        self, user_num, item_num, factor_num,
    ):
        super(ModelClass, self).__init__()
        self.factor_num = factor_num

        # 임베딩 저장공간 확보; num_embeddings, embedding_dim
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)
        predict_size = factor_num
        self.predict_layer = torch.ones(predict_size, 1).cuda()
        self._init_weight_()

    def _init_weight_(self):
        # weight 초기화
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

        # bias 초기화
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, user, item):
        embed_user = self.embed_user(user)
        embed_item = self.embed_item(item)
        output_GMF = embed_user * embed_item
        prediction = torch.matmul(output_GMF, self.predict_layer)
        return prediction.view(-1)


    
    
    


