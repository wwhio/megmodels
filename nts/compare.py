import numpy as np
import megengine
import megengine.functional as F
import torch
import torch.nn.functional as TF

import tqdm

import model
import model_pt

import data_test

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_net_megengine(testset):

    net = model.attention_net(pretrained=True)
    net.eval()

    test_loss = 0
    test_correct = 0
    total = 0
    for data in tqdm.tqdm(testset):
        img, label = data[0], data[1]
        img = img.transpose(2, 0, 1)[None, :, :, :]
        img = img.astype(np.float32)
        img = np.copy(img)
        img = megengine.tensor(img)
        label = megengine.tensor([label])
        _, logits, _, _, _ = net(img)
        loss = F.nn.cross_entropy(logits, label)
        predict = F.argmax(logits, 1)
        total += 1
        test_correct += F.sum(predict == label)
        test_loss += loss.item()
        # if total == 200:
        #     break

    test_acc = float(test_correct) / total
    test_loss = test_loss / total

    print(f'test set loss: {test_loss:.3f} and test set acc: {test_acc:.3f} total sample: {total}')


def test_net_torch(testset):
    net_torch = model_pt.attention_net()
    net_torch.load_state_dict(torch.load("../weight_pt/nts.ckpt")['net_state_dict'])
    net_torch.to(torch_device)
    net_torch.eval()

    torch.set_grad_enabled = False

    test_loss = 0
    test_correct = 0
    total = 0
    for data in tqdm.tqdm(testset):
        img, label = data[0], data[1]
        img = img.transpose(2, 0, 1)[None, :, :, :]
        img = img.astype(np.float32)
        img = np.copy(img)
        img = torch.from_numpy(img).to(torch_device)
        label = torch.tensor([label]).to(torch_device)
        _, logits, _, _, _ = net_torch(img)
        loss = TF.cross_entropy(logits, label)
        _, predict = torch.max(logits, 1)
        total += 1
        test_correct += torch.sum(predict == label)
        test_loss += loss.item()
        # if total == 200:
        #     break

    test_acc = float(test_correct) / total
    test_loss = test_loss / total

    print(f'test set loss: {test_loss:.3f} and test set acc: {test_acc:.3f} total sample: {total}')


def compare():
    print('test with first 100 test samples in CUB_200_2011')
    cub_testset = data_test.CUB('../DATA/CUB_200_2011_test', len_limit=100)

    print('test with pytorch')
    test_net_torch(cub_testset)

    print('test with megengine')
    test_net_megengine(cub_testset)


if __name__ == '__main__':
    compare()
