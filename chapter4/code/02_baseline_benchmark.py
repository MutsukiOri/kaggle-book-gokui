import cv2
from tqdm import tqdm
import numpy as np
import timm
import torch
import torch.nn as nn
from cirtorch.datasets.genericdataset import ImagesFromList
from cirtorch.datasets.testdataset import configdataset
from cirtorch.layers.pooling import GeM
from cirtorch.utils.evaluate import compute_map_and_print
from torchvision import transforms


class ResNetOfftheShelfGeM(nn.Module):
    """
        cirtorchを利用した一般化プーリングによる帯域特徴量を抽出するモジュール
    """

    def __init__(self, backbone="resnet34", pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
        )
        self.pooling = GeM()

    def forward(self, x):
        bs = x.size(0)
        x = self.backbone(x)[-1]
        x = self.pooling(x).view(bs, -1)
        return x


def extract_vectors(model, image_files, input_size, out_dim, transform, bbxs=None, device="cuda"):
    '''
    モデルによって得られる特長量からベクトル表現を抽出する
    '''
    dataloader = torch.utils.data.DataLoader(
        ImagesFromList(root="", images=image_files, imsize=input_size,
                       transform=transform, bbxs=bbxs),
        batch_size=1, shuffle=False, num_workers=2, pin_memory=True
    )

    with torch.no_grad():
        vecs = torch.zeros(out_dim, len(image_files))
        for i, X in enumerate(dataloader):
            X = X.to(device)
            vecs[:, i] = model(X).squeeze()
    return vecs


def get_query_index_images(cfg):
    '''
    datasetのconfigから画像のパスから画像データを取得する
    '''
    index_images = [cfg["im_fname"](cfg, i) for i in range(cfg["n"])]
    query_images = [cfg["qim_fname"](cfg, i) for i in range(cfg["nq"])]

    try:
        bbxs = [tuple(cfg["gnd"][i]["bbx"]) for i in range(cfg["nq"])]
    except KeyError:
        bbxs = None

    return index_images, query_images, bbxs


def main(input_size=224, out_dim=512):
    device = "cuda"

    datasets = {
        "roxford5k": configdataset("roxford5k", "../data/"),
        "rparis6k": configdataset("rparis6k", "../data/")
    }

    model = ResNetOfftheShelfGeM(pretrained=True)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    for dataset_name, dataset_config in tqdm(datasets.items()):
        # インデックス画像，クエリ画像をデータセットから取得
        index_images, query_images, bbxs = get_query_index_images(
            dataset_config)
        # インデックス画像をベクトル表現に変換
        index_vectors = extract_vectors(
            model, index_images, input_size, out_dim, transform, device=device)
        # クエリ画像をベクトル表現に変換
        query_vectors = extract_vectors(
            model, query_images, input_size, out_dim, transform, bbxs=bbxs, device=device)

        # numpyに変換
        index_vectors = index_vectors.numpy()
        query_vectors = query_vectors.numpy()

        # コサイン類似度を計算
        scores = np.dot(index_vectors.T, query_vectors)
        ranks = np.argsort(-scores, axis=0)
        compute_map_and_print(dataset_name, ranks, dataset_config["gnd"])


if __name__ == "__main__":
    main()
