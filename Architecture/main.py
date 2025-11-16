import datetime
import logging
import os
import argparse
from Architecture.model import Model
from torch import optim, nn
from torch.nn import KLDivLoss
import torch
from util import *
# 设置随机种子
seed = 42
random.seed(seed)  # Python 的随机数
torch.manual_seed(seed)  # PyTorch CPU 随机数
torch.cuda.manual_seed(seed)  # PyTorch GPU 随机数
torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 随机数

def setup_logging():
    """设置日志记录"""
    # 创建日志目录
    log_dir = "loges"
    os.makedirs(log_dir, exist_ok=True)

    # 设置日志文件名
    log_filename = os.path.join(
        log_dir,
        f"TOtrain_nofp_epoch10_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.loge"
        # f"train_nofp{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.loge"
    )

    # 配置日志记录
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 添加控制台日志处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

    return logging


parser = argparse.ArgumentParser(description='main parser')
parser.add_argument('--train_file_path',default='../Existing Dataset/TO/train_dataset.json', type= str, help= 'CFG input path' )
parser.add_argument('--test_file_path',default='../Existing Dataset/TO/test_dataset.json', type= str, help= 'CFG input path' )
parser.add_argument('--epoches', default= 100, help= ' the epoches of training')
parser.add_argument('--batch_size', default= 128, help= 'batch size of training data')
parser.add_argument('--beta', default=0.5, help= 'the ratio of bi-scale combination')
parser.add_argument('-input_dim', default= 256, help= 'the input dimension')

args = parser.parse_args()

if __name__ == '__main__':
    setup_logging()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_batch_data = prepare_train_data(
        args.train_file_path,
        batch_size=args.batch_size
    )

    logging.info("数据准备完成")

    # model process
    logging.info("初始化模型")
    model = Model(input_dim=256, perturbation = 'Gauss').to(device)
    logging.info("模型初始化完成")

    # loss function
    l1 = nn.CosineEmbeddingLoss()
    l2 = KLDivLoss(reduction='batchmean')
    prel = nn.CrossEntropyLoss()
    logging.info("损失函数已定义")

    # parameters
    first_stage_parameters = list(model.student.first_stage.parameters()) + list(model.denoise_model.parameters())
    second_stage_parameters = list(model.student.second_stage.high_confidence_model.parameters()) + list(model.complement_model.parameters())

    # Optimizer
    first_stage_optimizer = optim.Adam(first_stage_parameters, lr=0.01)
    second_stage_optimizer = optim.Adam(second_stage_parameters, lr=0.01)
    pre_optimizer = optim.Adam(model.parameters(), lr= 0.01)
    logging.info("优化器已定义")

    test_data = prepare_test_data(
        args.test_file_path,
    )
    # training
    for epoch in range(args.epoches):
        logging.info(f"\n{'=' * 50}\nEpoch [{epoch + 1}/{args.epoches}]\n{'=' * 50}")
        first_stage_losses, second_stage_losses, pre_losses = stage_train(model, train_batch_data, first_stage_optimizer,
                                                              second_stage_optimizer, pre_optimizer,
                                                              l1, l2,prel, device)
        logging.info(f"first_stage_losses:{first_stage_losses:.4f}, second_stage_losses:{second_stage_losses:.4f}, prediction_losses:{pre_losses:.4f}")

        model_test(model, test_data, device)









