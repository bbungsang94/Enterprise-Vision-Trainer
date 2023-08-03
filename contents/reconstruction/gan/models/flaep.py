import torch
import timm
import torch.nn as nn
from torch import linalg
from torch_geometric.nn import GCNConv, Sequential
from contents.reconstruction.gan.models.flame import get_parser, FLAME
from contents.reconstruction.pinning.pins.pin import PinLoader


class FLAEP(nn.Module):
    def __init__(self, pin, batch_size):
        super().__init__()
        self.pin_calculator = PinLossCalculator(pin=pin)
        self.generator = FLAMESet(batch_size=batch_size)

        # output: batch, 1280
        self.latent_model = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=0)
        o = self.latent_model(torch.randn(1, 3, 1024, 1024))
        residual_module = nn.Sequential(nn.Linear(o.shape[-1], 1024),
                                        nn.PReLU(),
                                        nn.Linear(1024, 512),
                                        nn.PReLU(),
                                        nn.Linear(512, 256))

        outline_gcn = GCNFlaep(in_channel=3, out_channel=16)
        eyes_gcn = GCNFlaep(in_channel=6, out_channel=32)
        lips_gcn = GCNFlaep(in_channel=6, out_channel=16)
        borrow_gcn = GCNFlaep(in_channel=6, out_channel=32)

        shape_head = nn.Sequential(nn.Linear(1664, 1664 // 4),
                                   nn.Tanh(),
                                   nn.Linear(1664 // 4, 1664 // 8),
                                   nn.Tanh(),
                                   nn.Linear(1664 // 8, 20),
                                   nn.Tanh())

        expression_head = nn.Sequential(nn.Linear(864, 864 // 4),
                                        nn.Tanh(),
                                        nn.Linear(864 // 4, 864 // 8),
                                        nn.Tanh(),
                                        nn.Linear(864 // 8, 10),
                                        nn.Tanh())

        jaw_head = nn.Sequential(nn.Linear(1152, 1152 // 8),
                                 nn.Tanh(),
                                 nn.Linear(1152 // 8, 1152 // 16),
                                 nn.Tanh(),
                                 nn.Linear(1152 // 16, 3),
                                 nn.Tanh())

        self.Bodies = nn.ModuleDict(
            {
                'Residual': residual_module,
                'Outline': outline_gcn,
                'Eyes': eyes_gcn,
                'Borrow': borrow_gcn,
                'Lips': lips_gcn,
                'ShapeHead': shape_head,
                'ExpHead': expression_head,
                'JawHead': jaw_head,
            }
        )

    def sub_module(self, x) -> [torch.tensor]:
        image, graphs, gender = x
        if len(image) != self.generator.batch_size:
            return None
        if next(self.parameters()).is_cuda:
            image = image.to(torch.device("cuda"))
            for key, graph in graphs.items():
                graph = graph.to(torch.device("cuda"))
                graphs[key] = graph
        latent_space = self.latent_model(image)
        res_latent = self.Bodies['Residual'](latent_space)

        num_graph = graphs['Outline'].num_graphs
        outline = self.Bodies['Outline'](graphs['Outline'])
        outline = outline.view(num_graph, -1)
        eyes = self.Bodies['Eyes'](graphs['Eyes'])
        eyes = eyes.view(num_graph, -1)
        borrow = self.Bodies['Borrow'](graphs['Borrow'])
        borrow = borrow.view(num_graph, -1)
        lips = self.Bodies['Lips'](graphs['Lips'])
        lips = lips.view(num_graph, -1)

        shape = torch.cat([res_latent, lips, eyes, outline], dim=1)
        expression = torch.cat([res_latent, lips, borrow], dim=1)
        rot = torch.cat([res_latent, lips, outline], dim=1)

        shape = self.Bodies['ShapeHead'](shape)
        expression = self.Bodies['ExpHead'](expression)
        jaw = self.Bodies['JawHead'](rot)
        jaw = torch.cat([torch.zeros(num_graph, 3, device=jaw.device), jaw], dim=1)

        output = {'shape_params': shape, 'expression_params': expression, 'pose_params': jaw}
        return self.generator(genders=gender, **output)

    def forward(self, x) -> [torch.tensor]:
        y = self.sub_module(x)
        if y is None:
            return None
        else:
            _, landmarks = y
        s, e, j = None, None, None
        batch_size = len(landmarks)
        for b in range(0, batch_size):
            s1, e1, j1 = self.pin_calculator.calculate(landmarks[b])
            if b == 0:
                s, e, j = s1, e1, j1
            else:
                s = torch.cat((s, s1), dim=0)
                e = torch.cat((e, e1), dim=0)
                j = torch.cat((j, j1), dim=0)

        return s.view(batch_size, -1), e.view(batch_size, -1), j.view(batch_size, -1)


class GCNFlaep(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.model = Sequential('x, edge_index',
                                [(GCNConv(in_channels=in_channel, out_channels=8),
                                  'x=x, edge_index=edge_index -> x'),
                                 nn.LeakyReLU(),
                                 (GCNConv(in_channels=8, out_channels=16),
                                  'x=x, edge_index=edge_index -> x'),
                                 nn.LeakyReLU(),
                                 nn.Dropout(),
                                 (GCNConv(in_channels=16, out_channels=out_channel),
                                  'x=x, edge_index=edge_index -> x'),
                                 nn.LeakyReLU(),
                                 nn.Dropout(),
                                 ])

    def forward(self, x):
        return self.model(x.x, x.edge_index)


class FLAMESet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        gen_args = get_parser()
        self.batch_size = kwargs['batch_size']
        gen_args.batch_size = self.batch_size
        gen_args.flame_model_path = gen_args.flame_model_path.replace('generic', 'male')
        male = FLAME(config=gen_args)
        gen_args.flame_model_path = gen_args.flame_model_path.replace('male', 'female')
        female = FLAME(config=gen_args)
        self.model = nn.ModuleDict({'male': male, 'female': female})
        self.faces = male.faces

    def forward(self, genders, **kwargs):
        vertices, landmarks = self.model['male'](**kwargs)
        female_vertices, female_landmarks = self.model['female'](**kwargs)

        for i, sex in enumerate(genders):
            if sex == "female":
                vertices[i] = female_vertices[i]
                landmarks[i] = female_landmarks[i]
        return vertices, landmarks


class PinLossCalculator:
    def __init__(self, pin):
        self.pin_boxes = PinLoader.load_pins(path=pin, filename='pin_info.json')

    def calculate(self, landmark, mode='points68'):
        for key, box in self.pin_boxes.items():
            box.switch_mode(mode)
            self.pin_boxes[key] = box

        eyes_box = self.pin_boxes['Eyes']
        nose_box = self.pin_boxes['Nose']
        lip_box = self.pin_boxes['Lip']
        eyebrow_box = self.pin_boxes['Eyebrow']

        # 코 중심, 턱 끝
        nose_tip = landmark[nose_box()[0]]
        jaw_tip = landmark[self.pin_boxes['Align']()[1]]
        # 눈, 입 정보
        eyes_points = landmark[eyes_box()]
        lip_points = landmark[lip_box()]
        eyebrow_points = landmark[eyebrow_box()]

        # 코 중심으로부터의 턱 아래의 거리 - 입이 벌려진 높이
        lip_height = linalg.norm(lip_points[2] - lip_points[3])
        base = linalg.norm(jaw_tip - nose_tip) - lip_height

        # 코 중심으로부터의 눈가 거리
        shape = linalg.norm(eyes_points[0] - nose_tip) / base
        e2 = linalg.norm(eyes_points[1] - nose_tip) / base
        shape = torch.cat((shape.unsqueeze(0), e2.unsqueeze(0)))
        e3 = linalg.norm(eyes_points[4] - nose_tip) / base
        shape = torch.cat((shape, e3.unsqueeze(0)))
        e4 = linalg.norm(eyes_points[5] - nose_tip) / base
        shape = torch.cat((shape, e4.unsqueeze(0)))

        # 코 중심으로부터의 입 양 끝 거리
        l1 = linalg.norm(lip_points[0] - nose_tip) / base
        shape = torch.cat((shape, l1.unsqueeze(0)))
        expression = l1
        jaw = l1
        l2 = linalg.norm(lip_points[1] - nose_tip) / base
        shape = torch.cat((shape, l2.unsqueeze(0)))
        expression = torch.cat((expression.unsqueeze(0), l2.unsqueeze(0)))
        jaw = torch.cat((jaw.unsqueeze(0), l2.unsqueeze(0)))

        # 입의 벌린 정도, 입의 길이
        jaw = torch.cat((jaw, (lip_height / base).unsqueeze(0)))
        l3 = linalg.norm(lip_points[0] - lip_points[1]) / base
        shape = torch.cat((shape, l3.unsqueeze(0)))
        expression = torch.cat((expression, l3.unsqueeze(0)))

        # 눈의 높이(2, 3), 길이(0, 1)
        e5 = linalg.norm(eyes_points[1] - eyes_points[0]) / base
        expression = torch.cat((expression, e5.unsqueeze(0)))
        e6 = linalg.norm(eyes_points[2] - eyes_points[3]) / base
        expression = torch.cat((expression, e6.unsqueeze(0)))
        e7 = linalg.norm(eyes_points[5] - eyes_points[4]) / base
        expression = torch.cat((expression, e7.unsqueeze(0)))
        e8 = linalg.norm(eyes_points[6] - eyes_points[7]) / base
        expression = torch.cat((expression, e8.unsqueeze(0)))

        # 코 중심으로부터의 눈썹의 거리
        b1 = linalg.norm(eyebrow_points[1] - nose_tip) / base
        expression = torch.cat((expression, b1.unsqueeze(0)))
        b2 = linalg.norm(eyebrow_points[3] - nose_tip) / base
        expression = torch.cat((expression, b2.unsqueeze(0)))
        b3 = linalg.norm(eyebrow_points[5] - nose_tip) / base
        expression = torch.cat((expression, b3.unsqueeze(0)))
        b4 = linalg.norm(eyebrow_points[7] - nose_tip) / base
        expression = torch.cat((expression, b4.unsqueeze(0)))

        return shape, expression, jaw


if __name__ == "__main__":
    test = FLAEPv1()
