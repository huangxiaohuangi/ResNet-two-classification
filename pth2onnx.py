import argparse

import torch
from models import Res
from predict import make_model


def parse_args():
    parser = argparse.ArgumentParser()

    # model path
    parser.add_argument('--light_model_path', default='./model/light.pth', type=str)
    parser.add_argument('--pink_model_path', default='./model/pink.pth', type=str)
    parser.add_argument('--simplate_model_path', default='./model/simplate.pth', type=str)
    parser.add_argument('--result_csv', default='./result.csv')
    parser.add_argument('--image_size', type=int, default=80)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    return args

def pth_2onnx():
    """
    pytorch 模型转换为onnx模型
    :return:
    """
    torch_model = torch.load('./model/pink.pth')
    model = make_model(args)
    # model = Res.resnet18(pretrained=False, progress=True)
    model.load_state_dict(torch_model)
    batch_size = 1  # 批处理大小
    input_shape = (3, 80, 80)  # 输入数据

    # set the model to inference mode
    model.eval()
    print(model)
    x = torch.randn(batch_size, *input_shape)  # 生成张量
    export_onnx_file = "./model/pink.onnx"  # 目的ONNX文件名
    torch.onnx.export(model,
                      x,
                      export_onnx_file,
                      # 注意这个地方版本选择为11
                      opset_version=11,
                      )


if __name__ == '__main__':
    args = parse_args()
    pth_2onnx()
