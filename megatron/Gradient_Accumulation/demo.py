#  python3 -m torch.distributed.launch --master_port 11122 --nproc_per_node=2 3.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 为ResNet50调整图片尺寸
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # CIFAR-100 数据集
    train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, sampler=train_sampler)

    # 模型和优化器
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 100)  # 调整最后的全连接层以匹配 CIFAR-100 的类别数
    model.to(rank)
    model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 训练循环
    accumulation_steps = 4
    model.train()
    for epoch in range(10):  # 训练10个epoch
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(rank)
            print ("inputs",inputs.shape)
            labels = labels.to(rank)
            "第一种实现，没有梯度累积"
            # # optimizer.zero_grad()
            # outputs = model(inputs)
            # loss = criterion(outputs, labels)
            # loss.backward()
            # # 打印最后一层的梯度
            # last_layer_gradients = model.module.fc.weight.grad  # 获取最后一层的梯度
            # print(f"Rank {rank}, Epoch {epoch}, Step {i}, Last Layer Gradients: {last_layer_gradients}")

            # # 收集所有进程中的最后一层梯度
            # all_last_layer_gradients = [torch.zeros_like(last_layer_gradients) for _ in range(world_size)]
            # dist.all_gather(all_last_layer_gradients, last_layer_gradients)

            # if rank == 0:
            #     # 比较所有进程中的梯度是否相同
            #     is_same = all([torch.all(torch.eq(all_last_layer_gradients[0], grad)) for grad in all_last_layer_gradients[1:]])
            #     print(f"Are gradients same across all processes: {is_same}")

            # optimizer.step()
            # optimizer.zero_grad()
            "第二种实现，有梯度累积"
            # if (i + 1) % accumulation_steps == 0:
            #     optimizer.step()
            #     optimizer.zero_grad()

            "第三种实现，有梯度累积，但是不是每次都同步梯度"
            for _ in range(accumulation_steps):# 前K-1个step 不进行梯度同步（累积梯度）。
                with model.no_sync(): # 这里实施“不操作”
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    last_layer_gradients = model.module.fc.weight.grad  # 获取最后一层的梯度
                    print(f"Rank {rank}, Epoch {epoch}, Step {i}, Last Layer Gradients: {last_layer_gradients}")

                    # 收集所有进程中的最后一层梯度
                    all_last_layer_gradients = [torch.zeros_like(last_layer_gradients) for _ in range(world_size)]
                    dist.all_gather(all_last_layer_gradients, last_layer_gradients)

                    if rank == 0:
                        # 比较所有进程中的梯度是否相同
                        is_same = all([torch.all(torch.eq(all_last_layer_gradients[0], grad)) for grad in all_last_layer_gradients[1:]])
                        print(f"Are gradients same across all processes: {is_same}")
            optimizer.step()
            optimizer.zero_grad()



            if i % 100 == 0:
                print(f"Rank {rank}, Epoch {epoch}, Step {i}, Loss: {loss.item()}")

    cleanup()

def main():
    world_size = 2  # 假设我们有2个GPU
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
